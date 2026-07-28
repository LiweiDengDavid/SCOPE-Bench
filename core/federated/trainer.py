import copy
import random
from collections import defaultdict
from time import time

import torch

from ..base.trainer import TrainerBase
from ..evaluation.ranking import assert_finite_scores, topk_from_scores
from ..evaluation.export import (
    include_scores as export_includes_scores,
    is_enabled as export_is_enabled,
    write_recommendations,
)
from ..training.factory import Components
from ..utils.metrics import extract_target_metric
from ..utils.training import artifact_token
from .aggregator import FederatedAggregator
from .parameter_split import ParameterSplit
from .personal_state import PersonalStateManager
from .triplet_logger import TripletLogger
from ..runtime.monitor import get_system_status


def sample_clients(
    clients, sample_strategy="random", sample_ratio=0.1, previous_clients=None
):
    """Sample clients for a federated training round."""
    if sample_ratio >= 1:
        return clients.copy()

    sample_count = max(1, int(len(clients) * sample_ratio))
    available_clients = clients
    if previous_clients:
        available_clients = list(set(clients) - set(previous_clients))
        if len(available_clients) < sample_count:
            available_clients = clients

    if sample_strategy == "random":
        return random.sample(available_clients, min(sample_count, len(available_clients)))
    raise ValueError(f"Invalid sample strategy: {sample_strategy}")


class FederatedTrainer(TrainerBase):
    """Federated Learning Trainer

    Inherits from TrainerBase and implements federated learning training logic.
    """

    def __init__(self, config, model):
        super(FederatedTrainer, self).__init__(config, model)

        # Extract federated parameters
        from ..config import extract_federated_params
        federated_params = extract_federated_params(config)
        self.local_epochs = federated_params['local_epochs']
        self.clients_sample_ratio = federated_params['clients_sample_ratio']
        self.clients_sample_strategy = federated_params['clients_sample_strategy']

        agg = federated_params['aggregation_method']
        if agg != "fedavg":
            raise ValueError(
                f"Unsupported aggregation_method: '{agg}'. "
                f"Currently only 'fedavg' is implemented."
            )

        self.client_models = {}
        self.best_client_models = {}

        self.last_participants = None
        # Resume state includes this round-level NaN early-stop counter.
        self.consecutive_nan_rounds = 0
        # Resolve the static federated parameter classification once (shared /
        # personal / row-personal / server-grad, + orphan & server-grad checks).
        # Mirror the resolved names onto plain trainer attributes (the runtime
        # read surface used throughout this class and by the eval/resume paths).
        self.split = ParameterSplit(model, self.config, self.logger)
        self.shared_param_names = self.split.shared_param_names
        self.personal_param_names = self.split.personal_param_names
        self.row_personal_param_names = self.split.row_personal_param_names
        self.server_grad_param_names = self.split.server_grad_param_names
        self.split_aware_federation = self.split.split_aware
        # Eval-side personal-state swapping lives in a collaborator that operates
        # on this trainer's client_models cache.
        self.personal = PersonalStateManager(model, self.split)
        # Server-side aggregation stages updates until a round is known valid.
        self.aggregator = FederatedAggregator(model, self.split, self.device, self.config)

        from ..data.features import setup_federated_features
        setup_federated_features(config, self)

        # Create reusable client model and optimizer (one-time cost)
        self._init_client_model()
        self.triplet_logger = TripletLogger(self.config, artifact_token)

    def _init_client_model(self):
        """Create a reusable client model instance and optimizer.

        Instead of deepcopy(self.model) per client per round, we create ONE
        client model at init and reuse it via load_state_dict (in-place copy).
        This eliminates the dominant per-client allocation cost.
        """
        self._client_model = copy.deepcopy(self.model)
        self._client_model = self._client_model.to(self.device)

        # Share multimodal features (not per-client copies)
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            self._client_model.v_feat = self.v_feat
        if hasattr(self, 't_feat') and self.t_feat is not None:
            self._client_model.t_feat = self.t_feat

        # Optionally compile client model for reduced kernel launch overhead.
        # Requires CUDA + C++ compiler (Linux GPU servers). Disable on Windows/CPU.
        if self.config["compile_client_model"] and self.device.type == "cuda":
            self._client_model = torch.compile(
                self._client_model, mode='reduce-overhead'
            )
            self.logger.info("Client model compiled with torch.compile (reduce-overhead)")

        self._client_optimizer = Components.create_optimizer(
            self._client_model, self.config, params=None
        )

        # torch.compile wraps the model in an OptimizedModule that prefixes every
        # state_dict()/named_parameters() key with '_orig_mod.' while sharing the
        # SAME Parameter objects. Route all state-dict / param-lookup / load_state_dict
        # operations through the underlying (uncompiled) module so the name-keyed
        # federated lookups (shared/personal/delta) and the round-snapshot loads keep
        # matching; forward()/calculate_loss still run through the compiled wrapper.
        self._client_state_module = getattr(self._client_model, "_orig_mod", self._client_model)

        # Cache param lookup dicts: Parameter objects are stable across rounds,
        # only their .data changes via load_state_dict(assign=False).
        self._client_named_params = dict(self._client_state_module.named_parameters())
        self._client_named_buffers = dict(self._client_state_module.named_buffers())

    def _personal_state_snapshot(self, user_id, name, tensor):
        # Delegate row-vs-whole personal tensor handling to PersonalStateManager.
        if self.personal.is_row_personal(name):
            user_idx = self.personal.personal_user_index(user_id)
            return tensor[user_idx].detach().cpu().clone()
        return tensor.detach().cpu().clone()

    def _copy_personal_state_tensor(self, name, target, tensor, user_id) -> None:
        if self.personal.is_row_personal(name):
            user_idx = self.personal.personal_user_index(user_id)
            row = self.personal.stored_personal_row(target, tensor, user_idx)
            target.data[user_idx].copy_(row.to(target.device))
            return
        target.data.copy_(tensor.to(target.device))

    # ------------------------------------------------------------------
    # Client training helpers (merged from runtime.py)
    # ------------------------------------------------------------------

    def _restore_client_personal_state(self, client_model, user_id):
        """Restore per-client personal parameters via direct tensor copy.

        Uses .data.copy_() instead of a full state_dict round-trip, avoiding
        the overhead of "read all → modify few → write all".
        """
        if not self.split_aware_federation:
            return client_model

        personal_state = self.client_models.get(user_id)
        if not personal_state:
            return client_model

        named_params = self._client_named_params
        named_buffers = self._client_named_buffers
        for name, tensor in personal_state.items():
            p = named_params.get(name)
            if p is None:
                p = named_buffers.get(name)
            if p is not None:
                self._copy_personal_state_tensor(name, p, tensor, user_id)
        return client_model

    def _train_client(self, client_model, client_optimizer, client_loader, user=None, epoch_idx=None):
        client_losses = []
        client_model.train()
        for epoch in range(self.local_epochs):
            client_loss = 0
            for batch_idx, batch in enumerate(client_loader):
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
                elif torch.is_tensor(batch):
                    batch = batch.to(self.device)

                if self.config["save_training_triplets"]:
                    self.triplet_logger.record(
                        epoch_idx,
                        user,
                        epoch,
                        batch_idx,
                        batch,
                        client_model,
                    )

                client_model, client_optimizer, loss = self._train_one_batch(
                    batch, client_model, client_optimizer
                )

                if self._check_nan(loss):
                    self.logger.info(
                        "NaN Loss exists at the [Batch:%s of %s-th Inner Epoch at %s-th user of %s-th outer loop]",
                        batch_idx,
                        epoch,
                        user if user is not None else "unknown",
                        epoch_idx if epoch_idx is not None else "unknown",
                    )
                    # The per-user loader is cached across rounds and resets its
                    # pointers only on exhaustion; returning mid-iteration would
                    # leave it mid-stream, so this client's next sampled round
                    # would silently train a truncated first epoch (and divide a
                    # partial loss sum by the full batch count). Reset explicitly
                    # so the NaN round abort leaves the loader state unchanged.
                    client_loader.pr = 0
                    client_loader.inter_pr = 0
                    return None

                client_loss += loss.item()

            client_losses.append(client_loss / len(client_loader))
            tol = float(self.config["tolerance"])
            if (
                epoch > 0
                and abs(client_losses[-1] - client_losses[-2]) / (client_losses[-1] + 1e-6)
                < tol
            ):
                break

        avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
        return {"loss": avg_loss}

    def _stage_personal(self, user_id, client_model):
        """Snapshot one client's personal params (row-personal stored as a
        single row) for staging by the aggregator. None when not split-aware.
        Unwraps a compiled client so state_dict keys are unprefixed."""
        if not (self.split_aware_federation and self.personal_param_names):
            return None
        client_state = getattr(client_model, "_orig_mod", client_model).state_dict()
        return {
            key: self._personal_state_snapshot(user_id, key, client_state[key])
            for key in self.personal_param_names
            if key in client_state
        }

    # ------------------------------------------------------------------
    # Best-state tracking (override base to include client_models)
    # ------------------------------------------------------------------

    def _update_best(self, valid_result, test_result, epoch_idx):
        updated = super()._update_best(valid_result, test_result, epoch_idx)
        if updated:
            self.best_client_models = copy.deepcopy(self.client_models)
        return updated

    def evaluate_final_test(self, test_data, extra_state_pairs=None):
        pairs = [] if extra_state_pairs is None else list(extra_state_pairs)
        pairs.append(("best_client_models", "client_models"))
        return super().evaluate_final_test(test_data, extra_state_pairs=pairs)

    # ------------------------------------------------------------------
    # Training resume: extend the base full-state payload with federated state
    # ------------------------------------------------------------------

    def _resume_state_dict(self, completed_epochs: int, train_data) -> dict:
        """Add federated runtime state to the base resume payload.

        The base captures the runtime global model + best-tracking + RNG. Federated
        additionally needs the runtime per-client personal params (all clients),
        their val-best partner, the client-sampling anti-repeat memory, and the
        NaN-round counter. Per-round accumulators are round-local and are not
        serialized.
        """
        state = super()._resume_state_dict(completed_epochs, train_data)
        # Federated training never steps the base global optimizer/scheduler
        # (per-client optimizers are built ephemerally in _set_client and cleared
        # per client), so the base optimizer payload is not applicable here.
        state["optimizer_state_dict"] = None
        state["lr_scheduler_state_dict"] = None
        state["client_models"] = copy.deepcopy(self.client_models)
        state["best_client_models"] = copy.deepcopy(self.best_client_models)
        state["last_participants"] = self.last_participants
        state["consecutive_nan_rounds"] = self.consecutive_nan_rounds
        return state

    def _restore_resume_state(self, checkpoint: dict, train_data) -> None:
        super()._restore_resume_state(checkpoint, train_data)
        self.client_models = checkpoint["client_models"]
        self.best_client_models = checkpoint["best_client_models"]
        self.last_participants = checkpoint["last_participants"]
        self.consecutive_nan_rounds = checkpoint["consecutive_nan_rounds"]

    def _checkpoint_payload(self) -> dict:
        """Add per-client personal parameters to the inherited base checkpoint.

        The base save_checkpoint config block reads config['is_federated'] /
        config['is_sequential'], which equal True/False in any federated run, so
        only the client_models entry is federated-specific.
        """
        return {"client_models": self.best_client_models}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self, train_data, valid_data=None, test_data=None):
        """Federated training entry point."""
        self.training_start_time = time()
        return self._fit_federated(train_data, valid_data, test_data)

    def _fit_federated(self, train_data, valid_data, test_data):
        """Federated training that respects explicit shared/personal parameter splits."""
        if not self.requires_training:
            self.logger.info("Skipping federated training because require_training=false")

            valid_result = self.evaluate(valid_data, idx=0) if valid_data else {}
            test_result = self.evaluate(test_data, is_test=True, idx=1) if test_data else {}

            if valid_result:
                self.best_valid_result = valid_result
                self.best_valid_score = extract_target_metric(
                    valid_result, self.valid_metric
                )
            if test_result:
                self.best_test_result = test_result

            self.best_epoch = 0
            self.best_model_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
            self.best_client_models = copy.deepcopy(self.client_models)

            self._log_training_summary(self.best_epoch)
            return self.best_valid_score, self.best_valid_result, self.best_test_result

        import math

        # self.consecutive_nan_rounds persists across resume (restored from the
        # checkpoint, else 0 from __init__); the round loop starts at start_epoch.
        for iteration in range(self.start_epoch, self.max_epochs):
            iteration_start_time = time()
            train_loss, _ = self._train_epoch(train_data, iteration)
            iteration_time = time() - iteration_start_time

            loss_is_nan = math.isnan(train_loss) if isinstance(train_loss, float) else False

            participant_count = 0 if self.last_participants is None else len(self.last_participants)
            self.logger.info(
                f"[Epoch {iteration + 1}/{self.max_epochs}][Train] "
                f"Time: {iteration_time:.2f}s, "
                f"{get_system_status()}, "
                f"Clients: {participant_count}, "
                f"Loss: {'NaN' if loss_is_nan else f'{train_loss:.4f}'}"
            )

            # NaN round → count toward early stopping patience
            if loss_is_nan:
                self.consecutive_nan_rounds += 1
                self.cur_step += 1
                if self.consecutive_nan_rounds >= self.stopping_step:
                    self.logger.error(
                        "[Round %d] %d consecutive NaN rounds — force early stop",
                        iteration + 1, self.consecutive_nan_rounds,
                    )
                    break
                continue
            else:
                self.consecutive_nan_rounds = 0

            should_evaluate = self._should_eval(iteration + 1, self.max_epochs)
            if should_evaluate:
                self._eval_and_update(valid_data, test_data, iteration)

                if self._patience_exhausted():
                    self.logger.info(f"Early stopping at iteration {iteration + 1}")
                    break
            else:
                self.logger.debug(f"Skipping evaluation at iteration {iteration + 1}")

            # Save only after the round is finalized: global/personal state is
            # committed and accumulators are cleared, so resume starts at the
            # next round boundary.
            self._maybe_save_resume(iteration + 1, train_data)

        self._log_training_summary(self._get_best_epoch())
        return self.best_valid_score, self.best_valid_result, self.best_test_result


    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        """Train for one federated round.

        Returns:
            Average loss and list of user losses
        """
        # Propagate epoch index to models that use dynamic loss weighting
        if hasattr(self.model, 'current_epoch'):
            self.model.current_epoch = epoch_idx

        if not self.requires_training:
            return 0.0, []

        sampled_clients = sample_clients(
            list(train_data.user_set),
            self.clients_sample_strategy,
            self.clients_sample_ratio,
            previous_clients=self.last_participants,
        )

        self.last_participants = sampled_clients

        # Improvement 1: snapshot global state ONCE per round — all clients
        # share this immutable snapshot instead of calling state_dict() N times.
        self._round_global_state = self.model.state_dict()

        # For split-aware models, pre-build a shared-only state dict so each
        # client only restores shared params + buffers (Improvement 2).
        if self.split_aware_federation:
            personal_set = set(self.personal_param_names)
            self._round_shared_state = {
                k: v for k, v in self._round_global_state.items()
                if k not in personal_set
            }

        # Initialize GPU running sums + the server-gradient delta reference for
        # this round (the aggregator owns the accumulator lifecycle).
        self.aggregator.begin_round(self._round_global_state)

        total_loss, user_losses = 0, []
        nan_client_count = 0
        for user in sampled_clients:
            client_loader = train_data.loaders[user]
            client_model, client_optimizer = self._set_client(user, epoch_idx)

            client_losses = self._train_client(
                client_model, client_optimizer, client_loader,
                user=user, epoch_idx=epoch_idx,
            )
            if client_losses is None:
                nan_client_count += 1
                continue

            if not isinstance(client_losses, dict):
                raise TypeError("Client training must return a dict of loss statistics")
            loss_val = client_losses["loss"]

            total_loss += float(loss_val)
            user_losses.append(float(loss_val))

            n_k = len(client_loader.dataset)
            staged = self._stage_personal(user, client_model)
            self.aggregator.accumulate(
                user, self._client_named_params, self._client_named_buffers, staged, n_k
            )

        if nan_client_count > 0:
            n_sampled = len(sampled_clients)
            self.logger.warning(
                "[Round %d] %d/%d clients produced NaN loss — discarding round, "
                "global model and personal state unchanged",
                epoch_idx + 1, nan_client_count, n_sampled,
            )
            # Abort: discard accumulators and staged personal updates without
            # changing global or per-client state.
            self.aggregator.abort()
            self._round_global_state = None
            if hasattr(self, '_round_shared_state'):
                self._round_shared_state = None
            return float("nan"), []

        # Commit: write the aggregated global model + commit the staged personal
        # updates together (half-commit is structurally unrepresentable).
        committed_personal = self.aggregator.finalize()
        if committed_personal:
            self.client_models.update(committed_personal)
        self._update_hyperparams(epoch_idx)

        # Release round-scoped snapshots
        self._round_global_state = None
        if hasattr(self, '_round_shared_state'):
            self._round_shared_state = None

        n_successful = len(user_losses) if user_losses else 1
        return total_loss / n_successful, user_losses

    def _set_client(self, user_id, epoch_idx):
        """Set up client model and optimizer via in-place state swap.

        Uses a single reusable _client_model instance instead of deepcopy per
        client.  load_state_dict(assign=False) copies data in-place, keeping
        nn.Parameter object identity — so the optimizer's param_groups remain
        valid and we only need to clear its running state.

        Optimization: for split-aware models with cached personal state, only
        restore shared params + buffers from the round snapshot, then overlay
        the client's personal params — skipping the redundant global→personal
        copy that would be immediately overwritten.

        INVARIANT (fast path, row-personal tables): only the returning client's
        OWN row is restored; every other row still holds whatever the
        previously-trained client of this round left there (only the slow path
        full-loads round-start state). The "immediately overwritten" claim in
        the optimization rationale is therefore true for the own row only. This
        is sound iff calculate_loss never reads row-personal rows other than
        the client's own — e.g. a full-table L2 over the user-embedding matrix
        would mix another client's private mid-round rows into every gradient.
        Models declaring row-personal params via get_personal_parameters()
        must honor this contract (see docs/reference/interfaces.md).
        """
        if self.split_aware_federation and user_id in self.client_models:
            # Fast path: load only shared params + buffers, then overlay this
            # client's personal state (row-personal: own row only — see the
            # INVARIANT in the docstring above).
            self._client_state_module.load_state_dict(self._round_shared_state, strict=False)
            self._restore_client_personal_state(self._client_model, user_id)
        else:
            # Full load: non-split-aware or first-time client (no cached personal state)
            self._client_state_module.load_state_dict(self._round_global_state, strict=True)

        # Sync non-parameter attributes not covered by state_dict
        if hasattr(self.model, 'current_epoch'):
            self._client_model.current_epoch = self.model.current_epoch

        # Clear optimizer state (fresh per client, matching original behavior)
        self._client_optimizer.state = defaultdict(dict)

        # Sync optimizer LR with config (handles FedRAP/MMFedRAP per-round decay)
        current_lr = float(self.config['learning_rate'])
        for pg in self._client_optimizer.param_groups:
            pg['lr'] = current_lr

        return self._client_model, self._client_optimizer

    def _train_one_batch(self, batch, client_model, client_optimizer):
        """Train one batch."""
        client_optimizer.zero_grad(set_to_none=True)
        loss = client_model.calculate_loss(batch)
        loss.backward()
        if self.clip_grad_norm and self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), self.clip_grad_norm)
        client_optimizer.step()

        return client_model, client_optimizer, loss

    def _check_nan(self, loss):
        """Check if loss is NaN."""
        return torch.isnan(loss).any() or torch.isinf(loss).any()

    def _update_hyperparams(self, epoch_idx):
        """Update hyperparameters — override in subclasses."""
        pass

    # ------------------------------------------------------------------
    # Federated evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_federated(self, eval_data, idx=0, write_export=False):
        """Federated evaluation with per-user personal state restoration.

        PersonalStateManager.eval_session() owns snapshot/merge/per-user-swap/
        restore; this method drives ranking through the shared evaluation core.
        """
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            self.model.v_feat = self.v_feat
        if hasattr(self, 't_feat') and self.t_feat is not None:
            self.model.t_feat = self.t_feat

        from ..utils.training import prepare_batch

        topk_batches = []
        topk_score_batches = []
        score_batches = []
        component_batches = {}
        topk = max(self.config["topk"])
        export_enabled = write_export and export_is_enabled(self.config)
        export_scores = export_enabled and export_includes_scores(self.config)

        with self.personal.eval_session(self.client_models, self.split_aware_federation) as session:
            self.model.eval()
            for user_id, user_loader in eval_data:
                session.apply_user(user_id)

                for batch in user_loader:
                    batch = prepare_batch(batch, self.device)
                    scores = self.model.full_sort_predict(batch)
                    assert_finite_scores(scores, self.model)
                    score_components = self._full_sort_score_components(batch)

                    mask_matrix = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
                    topk_indices = topk_from_scores(
                        scores, mask_matrix, topk, score_components=score_components
                    )
                    topk_batches.append(topk_indices)
                    if export_scores:
                        topk_score_batches.append(scores.gather(1, topk_indices).detach().cpu())
                    if self.config["save_recommendation_scores"]:
                        score_batches.append(scores.detach().cpu())
                    self._append_score_components(component_batches, score_components)

            self._save_recommendation_scores(score_batches, eval_data, idx=idx)
            self._save_recommendation_score_components(
                component_batches, eval_data, idx=idx
            )
            result = self.evaluator.evaluate(topk_batches, eval_data, idx=idx)
            if export_enabled:
                paths = write_recommendations(
                    self.config,
                    eval_data.get_eval_users(),
                    torch.cat(topk_batches, dim=0),
                    torch.cat(topk_score_batches, dim=0) if export_scores else None,
                    idx=idx,
                    metrics=result,
                    user_count=eval_data.dataset.user_num,
                    item_count=eval_data.item_num,
                )
                self.logger.info("Recommendation export saved: %s", paths)
        # eval_session.__exit__ restores the global model exactly.
        return result

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0, write_export=False):
        """Evaluation delegated to federated-specific implementation."""
        del is_test
        return self._evaluate_federated(
            eval_data,
            idx=idx,
            write_export=write_export,
        )

    def get_client_cache_info(self) -> dict:
        """Return size information about the per-client model cache."""
        total_params = sum(
            sum(p.numel() for p in state.values() if torch.is_tensor(p))
            for state in self.client_models.values()
            if isinstance(state, dict)
        )
        return {
            "client_models_size": len(self.client_models),
            "cached_parameters_count": total_params,
        }
