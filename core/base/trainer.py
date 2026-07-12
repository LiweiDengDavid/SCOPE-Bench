# coding: utf-8
 

"""
TrainerBase - Unified Training Interface
========================================

Simplified training logic following RecBole's approach with automatic paradigm detection.
Uses unified utility classes to eliminate code duplication across training paradigms.
"""

import logging
import os
from abc import ABC
import copy
from time import time

import numpy as np
import torch

from ..config import (
    extract_evaluation_params,
    extract_training_params,
)
from ..evaluation.ranking import assert_finite_scores, topk_from_scores
from ..evaluation.export import (
    include_scores as export_includes_scores,
    is_enabled as export_is_enabled,
    write_recommendations,
)
from ..evaluation.topk_kernel import compute_item_pop_count
from ..utils.metrics import extract_target_metric
from ..utils.training import (
    prepare_batch as _prepare_batch,
    train_epoch,
    dict2str,
    early_stopping,
    capture_rng_state,
    restore_rng_state,
    artifact_token,
)
from ..runtime.logger import TrainLogger


class TrainerBase(ABC):
    """Unified Training Interface
    
    Automatically handles different training paradigms with reduced code duplication.
    Uses unified utility classes following RecBole's design principles.
    """
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = logging.getLogger("nexusrec")
        
        # Use unified configuration utilities to eliminate parameter extraction duplication
        training_params = extract_training_params(config)
        evaluation_params = extract_evaluation_params(config)
        
        # Extract training parameters using unified utilities
        self.device = torch.device(training_params['device'])
        self.max_epochs = training_params['max_epochs']
        self.eval_step = training_params['eval_step']
        self.eval_enabled = training_params['eval_enabled']
        self.skip_eval_during_training = training_params['skip_eval_during_training']
        self.stopping_step = training_params['stopping_step']
        self.early_stopping = training_params['early_stopping']
        self.clip_grad_norm = training_params['clip_grad_norm']
        
        # Extract evaluation parameters using unified utilities
        self.valid_metric = evaluation_params['valid_metric']
        self.valid_metric_bigger = evaluation_params['valid_metric_bigger']
        
        # Detect training paradigm
        self.is_multimodal = config["is_multimodal_model"]
        self.requires_training = self._resolve_requires_training(config)
        
        # Create optimizer and scheduler using factory
        self._optimizer_param_ids = ()
        self.optimizer = None
        self.lr_scheduler = None
        self._refresh_optimizer(force=True)
        
        # Create evaluator
        self.evaluator = self._create_evaluator(config)
        
        # Best result tracking
        self.best_valid_score = -float('inf') if self.valid_metric_bigger else float('inf')
        self.best_valid_result = {}
        self.best_test_result = {}
        self.cur_step = 0
        self.best_epoch = 0

        # Resume: epoch/round to start the fit loop from (0 = fresh run). Set by
        # load_training_state when config["resume_training"] and a checkpoint exists.
        self.start_epoch = 0

        # Best model state for storing validation-optimal model
        self.best_model_state = None

        # Training session info for logging
        self.training_start_time = None
        self.model_name = config["model"]
        self.dataset_name = config["dataset"]
        self.last_recommendation_export_paths = []

    @staticmethod
    def _resolve_requires_training(config):
        """Resolve whether the model requires gradient-based training."""
        value = config["require_training"]
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no"}
        return bool(value)
    
    def _create_evaluator(self, config):
        """Create the standard Top-K evaluator."""
        from ..evaluation.evaluator import TopKEvaluator

        return TopKEvaluator(config)

    def _get_optimizer_params(self):
        """Resolve the current trainable parameter set from the model.

        Models may expose get_optimizer_params() to control which params (and in
        which phase) are optimized — e.g. RecVAE alternates encoder/decoder. The
        result is a flat parameter iterable; only requires_grad params are kept.
        """
        optimizer_params_getter = getattr(self.model, "get_optimizer_params", None)
        if callable(optimizer_params_getter):
            params = list(optimizer_params_getter())
        else:
            params = list(self.model.parameters())
        return [param for param in params if param.requires_grad]

    def _refresh_optimizer(self, force=False):
        """Rebuild optimizer/scheduler when the model changes its trainable phase.

        When the trainable parameter set changes (e.g. RecVAE alternates its
        encoder and decoder phases every few epochs), the optimizer and LR
        scheduler are rebuilt from scratch — Adam moment buffers and the LR
        schedule reset. This is intentional for RecVAE's alternating per-phase
        optimization; models with a fixed trainable set are unaffected because
        the param-id guard below early-returns.
        """
        params = self._get_optimizer_params()
        param_ids = tuple(id(param) for param in params)

        if not force and param_ids == self._optimizer_param_ids:
            return

        from ..training.factory import Components

        self._optimizer_param_ids = param_ids
        self.optimizer = Components.create_optimizer(self.model, self.config, params=params)
        self.lr_scheduler = Components.create_lr_scheduler(self.optimizer, self.config)

    def _should_eval(self, iteration: int, total_iterations: int) -> bool:
        """Determine whether to run evaluation at this iteration."""
        if not self.eval_enabled:
            return False
        if self.skip_eval_during_training:
            return iteration == total_iterations
        # Always evaluate the final epoch, even when eval_step does not divide
        # max_epochs (or eval_step > max_epochs), so best_valid_result is
        # available for the HPO objective. Match the federated trainer contract.
        return iteration % self.eval_step == 0 or iteration == total_iterations

    def fit(self, train_data, valid_data=None, test_data=None):
        """Unified training interface for centralized training."""
        self.training_start_time = time()
        return self._fit_centralized(train_data, valid_data, test_data)
    
    def _fit_centralized(self, train_data, valid_data, test_data):
        """Centralized training using unified training utilities"""
        if not self.requires_training:
            self.logger.info("Skipping centralized training because require_training=false")

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

            self._log_training_summary(self.best_epoch)
            return self.best_valid_score, self.best_valid_result, self.best_test_result

        for epoch_idx in range(self.start_epoch, self.max_epochs):
            # Pre-epoch processing
            self.model.pre_epoch_processing()
            self._refresh_optimizer()

            train_epoch(
                model=self.model,
                optimizer=self.optimizer,
                train_data=train_data,
                device=self.device,
                clip_grad_norm=self.clip_grad_norm,
                lr_scheduler=self.lr_scheduler,
                epoch_idx=epoch_idx,
                total_epochs=self.max_epochs,
                nan_abort_threshold=self.config["nan_abort_threshold"],
            )

            # Evaluation
            if self._should_eval(epoch_idx + 1, self.max_epochs):
                self._eval_and_update(valid_data, test_data, epoch_idx)

                if self._patience_exhausted():
                    self.logger.info(f"Early stopping at epoch {epoch_idx + 1}")
                    break

            self.model.post_epoch_processing()
            # Resume checkpoint at the epoch boundary (RNG/optimizer/best-tracking
            # captured AFTER post-epoch so a resumed run continues bit-identically).
            self._maybe_save_resume(epoch_idx + 1, train_data)

        self._log_training_summary(self._get_best_epoch())
        return self.best_valid_score, self.best_valid_result, self.best_test_result

    def _log_training_summary(self, best_epoch: int) -> None:
        """Log the end-of-training summary with key metrics."""
        total_time = time() - self.training_start_time
        TrainLogger.log_training_summary(
            total_time,
            self.best_valid_result,
            self.max_epochs,
            best_epoch,
            key_metrics=self._key_metrics(self.config),
            best_test_metrics=self.best_test_result,
        )

    def _patience_exhausted(self) -> bool:
        """Return whether validation patience should stop training."""
        return self.early_stopping and self.cur_step >= self.stopping_step

    @staticmethod
    def _key_metrics(config):
        """Build key metric names for training summary display."""
        config_metrics = config["metrics"]
        config_topk = config["topk"]
        return [f"{metric}@{k}" for metric in config_metrics[:3] for k in config_topk[:1]]

    @staticmethod
    def _display_metrics(result, key_metrics):
        """Extract display metrics from evaluation results."""
        display_metrics = {}
        if not result:
            return display_metrics
        for metric in key_metrics:
            lowered = metric.lower()
            if lowered in result:
                display_metrics[metric] = result[lowered]
            elif metric in result:
                display_metrics[metric] = result[metric]
        return display_metrics

    def _update_best(self, valid_result, test_result, epoch_idx):
        """Update best validation/test results and emit unified logging."""
        valid_score = extract_target_metric(valid_result, self.valid_metric)
        (
            self.best_valid_score,
            self.cur_step,
            update_flag,
        ) = early_stopping(
            valid_score,
            self.best_valid_score,
            self.cur_step,
            self.valid_metric_bigger,
        )

        if not update_flag:
            return False

        self.best_valid_result = valid_result
        if test_result:
            self.best_test_result = test_result
        self.best_epoch = epoch_idx + 1

        if hasattr(self.model, "state_dict"):
            self.best_model_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
            self.logger.info("Best model state saved at epoch %s", epoch_idx + 1)

        key_metrics = self._key_metrics(self.config)
        display_metrics = self._display_metrics(valid_result, key_metrics)
        TrainLogger.log_best_result(
            epoch_idx + 1,
            display_metrics,
            is_best=True,
            key_metrics=key_metrics,
        )
        return True

    def _eval_and_update(self, valid_data, test_data, epoch_idx):
        """Evaluate current state, log metrics, then update tracked best metrics."""
        self._evaluation_epoch = epoch_idx + 1
        valid_result = self.evaluate(valid_data, idx=0) if valid_data else {}
        if valid_result:
            TrainLogger.log_detailed_metrics(
                valid_result, "Valid",
                config_metrics=self.config["metrics"],
                config_topk=self.config["topk"],
            )

        test_result = {}
        if test_data:
            eval_test_during_training = self.config["eval_test_during_training"]
            eval_test_frequency = self.config["eval_test_frequency"]
            if eval_test_during_training and (epoch_idx + 1) % eval_test_frequency == 0:
                test_result = self.evaluate(test_data, is_test=True, idx=1)
                TrainLogger.log_detailed_metrics(
                    test_result, "Test",
                    config_metrics=self.config["metrics"],
                    config_topk=self.config["topk"],
                )

        is_best = self._update_best(valid_result, test_result, epoch_idx)
        eval_test_during_training = self.config["eval_test_during_training"]
        if is_best and eval_test_during_training and test_data and not test_result:
            self.best_test_result = self.evaluate(test_data, is_test=True, idx=1)
            TrainLogger.log_detailed_metrics(
                self.best_test_result, "Test (best epoch)",
                config_metrics=self.config["metrics"],
                config_topk=self.config["topk"],
            )
        self._evaluation_epoch = None
        return is_best

    def _get_best_epoch(self):
        """Get the epoch when best result was achieved."""
        return getattr(self, "best_epoch", 1)

    # ------------------------------------------------------------------
    # Training resume: full-state checkpoint (distinct from best_model.pth)
    # ------------------------------------------------------------------

    def _resume_checkpoint_path(self) -> str:
        """Path of the full-state resume checkpoint (distinct from best_model.pth)."""
        return os.path.join(
            self.config["checkpoint_dir"], self.config["resume_checkpoint_name"]
        )

    def _resume_state_dict(self, completed_epochs: int, train_data) -> dict:
        """Build the full-state resume payload.

        Subclasses override to add paradigm-specific state (federated client_models;
        sequential DataLoader generator) by calling ``super()`` then augmenting.
        ``completed_epochs`` is the number of finished epochs/rounds; resume starts
        the loop at this index.
        """
        return {
            "epoch": completed_epochs,
            "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "lr_scheduler_state_dict": (
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            ),
            "best_valid_score": self.best_valid_score,
            "best_valid_result": self.best_valid_result,
            "best_test_result": self.best_test_result,
            "best_epoch": self.best_epoch,
            "cur_step": self.cur_step,
            "best_model_state": self.best_model_state,
            # Model training-loop state that lives OUTSIDE state_dict (e.g. RecVAE's
            # alternating-phase counters); empty for most models.
            "model_resume": self.model.get_resume_state(),
            "rng": capture_rng_state(),
            # Dataloaders carry cumulative iteration-order state (the shuffled
            # working df / an explicit Generator) that the global RNG alone does
            # not pin; capture it so a resumed epoch sees the same batch order.
            "dataloader": (
                train_data.get_resume_state()
                if hasattr(train_data, "get_resume_state")
                else None
            ),
        }

    def _move_optimizer_state_to_device(self) -> None:
        """Move loaded optimizer state tensors onto the trainer device.

        ``optimizer.load_state_dict`` keeps state tensors on their saved device
        (CPU here, since the resume file is loaded with map_location='cpu'); the
        optimizer step would then mismatch GPU params. No-op on CPU.
        """
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _optimizer_state_matches(self, optimizer_state: dict) -> bool:
        """Whether a saved optimizer state's param groups match the current optimizer.

        Mismatch happens legitimately when resuming across a phase boundary (RecVAE:
        the checkpoint is written after post_epoch_processing advanced the phase, so
        the saved optimizer belongs to the JUST-TRAINED phase while the restored
        phase rebuilds a different param set). An uninterrupted run also discards
        the prior phase's optimizer at that boundary, so skipping the load matches it.
        """
        if self.optimizer is None:
            return False
        saved_groups = optimizer_state["param_groups"]
        current_groups = self.optimizer.param_groups
        if len(saved_groups) != len(current_groups):
            return False
        return all(
            len(saved["params"]) == len(current["params"])
            for saved, current in zip(saved_groups, current_groups)
        )

    def _restore_resume_state(self, checkpoint: dict, train_data) -> None:
        """Restore all fit state from a resume checkpoint (fail-fast on bad keys)."""
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # Restore model-level loop state (e.g. RecVAE alternating phase) so the
        # resumed run continues the cycle. The optimizer built at construction
        # already covers the construction-phase param set; load the saved optimizer
        # state into it WITHOUT rebuilding (rebuilding a fresh optimizer here, even
        # with identical state, perturbs subsequent float updates vs the
        # continuously-reused optimizer of an uninterrupted run).
        self.model.set_resume_state(checkpoint["model_resume"])
        optimizer_state = checkpoint["optimizer_state_dict"]
        if (
            optimizer_state is not None
            and self.optimizer is not None
            and self._optimizer_state_matches(optimizer_state)
        ):
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()
            if (
                checkpoint["lr_scheduler_state_dict"] is not None
                and self.lr_scheduler is not None
            ):
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        # When the saved optimizer does NOT match (resume lands on a phase boundary
        # where the saved optimizer belongs to a different phase), it is left
        # unloaded: the fit loop's _refresh_optimizer rebuilds a fresh optimizer for
        # the restored phase at the next epoch, exactly as an uninterrupted run does
        # when it crosses that boundary.
        self.best_valid_score = checkpoint["best_valid_score"]
        self.best_valid_result = checkpoint["best_valid_result"]
        self.best_test_result = checkpoint["best_test_result"]
        self.best_epoch = checkpoint["best_epoch"]
        self.cur_step = checkpoint["cur_step"]
        self.best_model_state = checkpoint["best_model_state"]
        restore_rng_state(checkpoint["rng"])
        if checkpoint["dataloader"] is not None and hasattr(train_data, "set_resume_state"):
            train_data.set_resume_state(checkpoint["dataloader"])
        self.start_epoch = checkpoint["epoch"]

    def save_training_state(self, completed_epochs: int, train_data) -> str:
        """Persist full fit state to the resume checkpoint and return its path."""
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        path = self._resume_checkpoint_path()
        torch.save(self._resume_state_dict(completed_epochs, train_data), path)
        self.logger.debug(
            "Resume checkpoint saved to %s (after epoch %s)", path, completed_epochs
        )
        return path

    def _maybe_save_resume(self, completed_epochs: int, train_data) -> None:
        """Write a resume checkpoint at the configured cadence (opt-in only)."""
        if not self.config["resume_training"]:
            return
        if completed_epochs % self.config["checkpoint_every_n_epochs"] == 0:
            self.save_training_state(completed_epochs, train_data)

    def load_training_state(self, train_data) -> int:
        """Restore fit state from the resume checkpoint if present; return start_epoch.

        Opt-in via config["resume_training"]. A missing checkpoint is the normal
        first-run state (returns 0, fresh start) — not an error. A present-but-bad
        checkpoint fails loudly inside _restore_resume_state.
        """
        path = self._resume_checkpoint_path()
        if not os.path.isfile(path):
            self.logger.info(
                "resume_training=true but no resume checkpoint at %s; starting fresh.",
                path,
            )
            return self.start_epoch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._restore_resume_state(checkpoint, train_data)
        self.logger.info(
            "Resumed training from %s at epoch %s/%s",
            path,
            self.start_epoch,
            self.max_epochs,
        )
        return self.start_epoch

    # ------------------------------------------------------------------
    # Checkpoint: unified save/load interface
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """Save the best model state to a checkpoint file.

        Args:
            checkpoint_dir: Directory to save the checkpoint.

        Returns:
            The path of the saved checkpoint file.
        """
        path = os.path.join(checkpoint_dir, "best_model.pth")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_state = (
            self.best_model_state
            if self.best_model_state is not None
            else {k: v.cpu() for k, v in self.model.state_dict().items()}
        )

        checkpoint = {
            "model_state_dict": model_state,
            "config": {
                "model": self.config["model"],
                "dataset": self.config["dataset"],
                "embedding_size": self.config["embedding_size"],
                "n_users": self.model.n_users,
                "n_items": self.model.n_items,
                "is_federated": self.config["is_federated"],
                "is_multimodal_model": self.config["is_multimodal_model"],
                "is_sequential": self.config["is_sequential"],
            },
            "best_valid_result": self.best_valid_result,
            "best_test_result": self.best_test_result,
            **self._checkpoint_payload(),
        }

        torch.save(checkpoint, path)
        self.logger.info("Checkpoint saved to %s", path)
        return path

    def _checkpoint_payload(self) -> dict:
        """Extra checkpoint entries for subclasses (overridden by the federated
        trainer to add per-client personal parameters). Empty by default."""
        return {}

    def evaluate_final_test(self, test_data, extra_state_pairs=None):
        """Run final test evaluation with the validation-best state loaded, then restore."""
        if test_data is None:
            return {}

        state_pairs = [] if extra_state_pairs is None else list(extra_state_pairs)

        original_state = None
        original_extra_states = {}
        if self.best_model_state is not None:
            self.logger.info(
                "Loading best validation model from epoch %s for final test evaluation...",
                self.best_epoch,
            )
            original_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

            for best_attr, current_attr in state_pairs:
                best_state = getattr(self, best_attr, None)
                if best_state is None:
                    continue
                original_extra_states[current_attr] = copy.deepcopy(
                    getattr(self, current_attr)
                )
                setattr(self, current_attr, copy.deepcopy(best_state))
        else:
            self.logger.info("Performing final test evaluation with current model...")

        test_result = self.evaluate(
            test_data,
            is_test=True,
            idx=1,
            write_export=True,
        )
        self.best_test_result = test_result

        if original_state is not None:
            self.model.load_state_dict(original_state)
        for current_attr, original_value in original_extra_states.items():
            setattr(self, current_attr, original_value)

        if test_result:
            self.logger.info("[Final Test] %s", dict2str(test_result))

        return test_result
    
    def update_final_test_result(self, final_test_result):
        """Update the final test result for summary logging"""
        if final_test_result:
            self.best_test_result = final_test_result
    
    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0, write_export=False):
        """Evaluate the model on the given data."""
        del is_test
        self.model.eval()
        return self._evaluate_centralized(
            eval_data,
            idx=idx,
            write_export=write_export,
        )
    
    def _evaluate_centralized(self, eval_data, idx=0, write_export=False):
        """Centralized evaluation: one full-batch forward pass per batch."""
        topk_batches = []
        topk_score_batches = []
        score_batches = []
        component_batches = {}
        topk = max(self.config["topk"])
        export_enabled = write_export and export_is_enabled(self.config)
        export_scores = export_enabled and export_includes_scores(self.config)

        for batch in eval_data:
            batch = _prepare_batch(batch, self.device)

            # Full-batch prediction → [batch_size, n_items]
            scores = self.model.full_sort_predict(batch)
            assert_finite_scores(scores, self.model)
            score_components = self._full_sort_score_components(batch)

            # Mask seen items (batch[1] = [2, n] of batch-local user idx / item id)
            # and rank with the shared stable tie-break convention.
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
                item_count=eval_data.dataset.item_num,
            )
            self.last_recommendation_export_paths = paths
            self.logger.info("Recommendation export saved: %s", paths)
        return result

    def _full_sort_score_components(self, batch):
        if not self.config["save_recommendation_score_components"]:
            return {}
        component_predictor = getattr(self.model, "full_sort_predict_components", None)
        if not callable(component_predictor):
            raise ValueError(
                "save_recommendation_score_components requires the model to implement "
                "full_sort_predict_components()."
            )
        return component_predictor(batch)

    def _append_score_components(self, component_batches, score_components):
        if not self.config["save_recommendation_score_components"]:
            return
        for component_name, component_scores in score_components.items():
            if component_name not in component_batches:
                component_batches[component_name] = []
            component_batches[component_name].append(component_scores.detach().cpu())

    def _save_recommendation_score_components(self, component_batches, eval_data, idx=0):
        """Save masked full-sort score components for model-specific diagnostics."""
        if not self.config["save_recommendation_score_components"]:
            return
        if not component_batches:
            return

        users = torch.as_tensor(eval_data.get_eval_users(), dtype=torch.long)
        payload = {
            "users": users,
            "positive_items": eval_data.get_eval_items(),
            "positive_lengths": eval_data.get_eval_len_list(),
            "item_pop_count": self._eval_item_pop_count(eval_data),
        }
        for component_name, component_scores in component_batches.items():
            scores = torch.cat(component_scores, dim=0).to(torch.float32)
            if len(users) != scores.shape[0]:
                raise ValueError(
                    "Saved recommendation score components require eval users and "
                    f"score rows to match, got {len(users)} users and "
                    f"{scores.shape[0]} rows for {component_name}."
                )
            payload[component_name] = scores

        max_k = max(self.config["topk"])
        artifact_tag = self._evaluation_artifact_tag(idx, max_k, "components")
        file_path = os.path.join(
            self.config["checkpoint_dir"],
            f"[{self.config['model']}]-[{self.config['dataset']}]-"
            f"[{artifact_tag}]-components.pt",
        )
        torch.save(payload, file_path)

    def _save_recommendation_scores(self, score_batches, eval_data, idx=0):
        """Save masked full-sort scores for post-hoc routing diagnostics."""
        if not self.config["save_recommendation_scores"]:
            return
        if not score_batches:
            return

        scores = torch.cat(score_batches, dim=0).to(torch.float32)
        users = torch.as_tensor(eval_data.get_eval_users(), dtype=torch.long)
        if len(users) != scores.shape[0]:
            raise ValueError(
                "Saved recommendation scores require eval users and score rows "
                f"to match, got {len(users)} users and {scores.shape[0]} rows."
            )

        max_k = max(self.config["topk"])
        artifact_tag = self._evaluation_artifact_tag(idx, max_k, "scores")
        file_path = os.path.join(
            self.config["checkpoint_dir"],
            f"[{self.config['model']}]-[{self.config['dataset']}]-"
            f"[{artifact_tag}]-scores.pt",
        )
        payload = {
            "users": users,
            "scores": scores,
            "positive_items": eval_data.get_eval_items(),
            "positive_lengths": eval_data.get_eval_len_list(),
            "item_pop_count": self._eval_item_pop_count(eval_data),
        }
        torch.save(payload, file_path)

    def _eval_item_pop_count(self, eval_data):
        """Compute training item counts for score-based bucket diagnostics."""
        counts = compute_item_pop_count(eval_data)
        if counts is None:
            return torch.empty(0, dtype=torch.float32)
        return torch.from_numpy(counts.astype(np.float32))

    def _evaluation_artifact_tag(self, idx, max_k, suffix):
        """Build a run-identifying tag for saved evaluation artifacts."""
        epoch_part = ""
        if self.config["save_recommendation_artifact_by_epoch"]:
            epoch = getattr(self, "_evaluation_epoch", None)
            if epoch is not None:
                epoch_part = f".epoch{epoch}"
        raw_tag = (
            f"{self.config['type']}.{self.config['comment']}."
            f"seed{self.config['seed']}{epoch_part}.idx{idx}.top{max_k}.{suffix}"
        )
        return artifact_token(raw_tag)
