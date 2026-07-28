"""Federated server-side aggregation.

Owns the per-round running-sum lifecycle (weight-average + server-gradient
delta) and the two-phase commit so a NaN-aborted round leaves the global model
and the per-client personal cache unchanged from round-start.

The aggregator owns only round-lifetime accumulator state. Personal updates are
STAGED here during ``accumulate`` and returned by ``finalize`` for the trainer to
commit; ``abort`` discards them. Half-commit is structurally unrepresentable:
the round loop is ``if nan: aggregator.abort() else: commit(aggregator.finalize())``.
"""

from __future__ import annotations

import torch

from .parameter_split import ParameterSplit


class FederatedAggregator:
    def __init__(self, model, split, device, config):
        self.model = model
        self.split = split
        self.device = device
        self.config = config
        self._running_weight_avg = None
        self._running_delta_sum = None
        self._running_row_weights = None
        self._server_param_snapshot = None
        self._online_total_weight = 0.0
        self._round_personal_updates = {}

    def begin_round(self, global_state) -> None:
        """Zero the GPU running sums for a new round and snapshot the delta
        reference (round-start global params for server-gradient params)."""
        server_grad_set = set(self.split.server_grad_param_names)
        # Federated aggregation requires at least one shared parameter.
        assert self.split.split_aware, (
            "federated aggregation requires a non-empty get_shared_parameters()"
        )
        self._weight_avg_names = [
            n for n in self.split.shared_param_names if n not in server_grad_set
        ]
        self._delta_names = list(self.split.server_grad_param_names)

        self._running_weight_avg = {
            n: torch.zeros_like(global_state[n], device=self.device)
            for n in self._weight_avg_names if n in global_state
        }
        self._running_delta_sum = {
            n: torch.zeros_like(global_state[n], device=self.device)
            for n in self._delta_names if n in global_state
        }
        # Per-row weight tracking for sparse embedding delta params.
        self._running_row_weights = {}
        for n in self._delta_names:
            if n in global_state and ParameterSplit.is_row_delta(self.model, n, global_state[n]):
                self._running_row_weights[n] = torch.zeros(
                    global_state[n].shape[0], device=self.device
                )
        # Delta reference: round-start global params for server-gradient params.
        self._server_param_snapshot = {
            n: global_state[n].to(self.device).clone()
            for n in self._delta_names if n in global_state
        }
        self._online_total_weight = 0.0
        self._round_personal_updates = {}

    def accumulate(self, user_id, named_params, named_buffers, staged_personal, n_k) -> None:
        """Add one client's contribution to the running sums and stage its
        personal params (committed only on finalize)."""
        if staged_personal is not None:
            self._round_personal_updates[user_id] = staged_personal

        for name in self._weight_avg_names:
            p = named_params.get(name)
            if p is None:
                p = named_buffers.get(name)
            if p is not None:
                self._running_weight_avg[name].add_(p.data, alpha=n_k)

        if self._delta_names:
            snapshot = self._server_param_snapshot
            for name in self._delta_names:
                p = named_params.get(name)
                if p is None:
                    p = named_buffers.get(name)
                if p is None or name not in snapshot:
                    continue
                delta = snapshot[name] - p.data
                self._running_delta_sum[name].add_(delta, alpha=n_k)
                # Sparse embedding deltas are normalized row-wise so an untouched
                # row is not diluted by clients that never updated it.
                if name in self._running_row_weights:
                    nonzero_mask = delta.abs().sum(dim=-1) > 0
                    self._running_row_weights[name].add_(nonzero_mask.float(), alpha=n_k)

        self._online_total_weight += n_k

    def finalize(self) -> dict:
        """Normalize the running sums, write the global model, and RETURN the
        staged personal updates for the trainer to commit. Then tear down."""
        if self._online_total_weight == 0:
            self._teardown()
            return {}

        if self._running_weight_avg:
            global_state = self.model.state_dict()
            for name, running_sum in self._running_weight_avg.items():
                if name in global_state:
                    global_state[name] = running_sum / self._online_total_weight
            self.model.load_state_dict(global_state, strict=False)

        if self._running_delta_sum:
            aggregated_deltas = {}
            for name, running_sum in self._running_delta_sum.items():
                if name in self._running_row_weights:
                    row_weights = self._running_row_weights[name].clamp(min=1e-12)
                    aggregated_deltas[name] = running_sum / row_weights.unsqueeze(-1)
                else:
                    aggregated_deltas[name] = running_sum / self._online_total_weight
            self._apply_server_grad_update(aggregated_deltas)

        personal = self._round_personal_updates
        self._teardown()
        return personal

    def abort(self) -> None:
        """Discard the round's accumulators + staged personal without committing."""
        self._teardown()

    def _teardown(self) -> None:
        self._running_weight_avg = None
        self._running_delta_sum = None
        self._running_row_weights = None
        self._round_personal_updates = None
        self._server_param_snapshot = None

    def _apply_server_grad_update(self, aggregated_deltas) -> None:
        """param -= server_learning_rate * delta. The delta already carries the
        client learning-rate effect (delta ≈ η_c·∇ under SGD)."""
        if not aggregated_deltas:
            return
        if "server_learning_rate" not in self.config:
            raise ValueError(
                "server_learning_rate must be explicitly set in model config "
                "for algorithms using server-gradient delta aggregation. "
                "No implicit default is provided."
            )
        server_lr = self.config["server_learning_rate"]
        global_state = self.model.state_dict()
        for param_name, delta in aggregated_deltas.items():
            if param_name in global_state:
                global_state[param_name] = (
                    global_state[param_name] - server_lr * delta.to(global_state[param_name].device)
                )
        self.model.load_state_dict(global_state, strict=False)
