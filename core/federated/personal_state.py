"""Federated personal-state eval choreography.

Owns the snapshot / merge / per-user-swap / restore dance that lets federated
eval run each client's personal parameters against the global model and then
leave that global model BYTE-IDENTICAL to how training produced it.

The manager operates on the trainer's ``client_models`` cache (passed into
``eval_session``); it holds only references to the model + parameter split. The
context manager guarantees restoration runs even on an exception.
"""

from __future__ import annotations


class PersonalStateManager:
    """Owns federated evaluation personal-state swaps."""

    def __init__(self, model, split):
        self.model = model
        self.split = split  # ParameterSplit: provides personal/row_personal name sets

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_row_personal(self, name) -> bool:
        return name in self.split.row_personal_param_names

    def personal_user_index(self, user_id):
        """Bounds-checked per-user row index (single home; used by training + eval)."""
        n_users = getattr(self.model, "n_users", None)
        if n_users is None:
            raise AttributeError(
                "Row-indexed personal state requires model.n_users to be defined."
            )
        user_idx = int(user_id)
        if user_idx < 0 or user_idx >= n_users:
            raise IndexError(
                f"Personal state user_id {user_id} is outside [0, {n_users})."
            )
        return user_idx

    @staticmethod
    def stored_personal_row(target, tensor, user_idx):
        if tuple(tensor.shape) == tuple(target.shape):
            return tensor[user_idx]
        return tensor

    # ------------------------------------------------------------------
    # Non-row personal params (swapped per-user during eval)
    # ------------------------------------------------------------------

    def classify_non_row(self, named_params, named_buffers) -> set:
        """Personal params that are NOT row-indexed (item tables, affine heads,
        fusion routers) — swapped wholesale per user during eval."""
        non_row = set()
        for pname in self.split.personal_param_names:
            param = named_params.get(pname)
            if param is None:
                param = named_buffers.get(pname)
            if param is not None and not self.is_row_personal(pname):
                non_row.add(pname)
        return non_row

    def get_clean_non_row(self, named_params, named_buffers, non_row_names) -> dict:
        """Snapshot non-row-indexed params from the current global model."""
        clean = {}
        for param_name in non_row_names:
            param = named_params.get(param_name)
            if param is None:
                param = named_buffers.get(param_name)
            if param is not None:
                clean[param_name] = param.data.clone()
        return clean

    def apply_non_row(self, client_models, user_id, named_params, named_buffers, non_row_names) -> None:
        """Swap non-row-indexed personal params for a specific user."""
        client_state = client_models.get(user_id)
        if not client_state:
            return
        for param_name in non_row_names:
            tensor = client_state.get(param_name)
            if tensor is None:
                continue
            param = named_params.get(param_name)
            if param is None:
                param = named_buffers.get(param_name)
            if param is not None:
                param.data.copy_(tensor.to(param.device))

    def load_non_row(self, non_row_state, named_params, named_buffers) -> None:
        """Overwrite non-row-indexed params in the global model from a snapshot."""
        for param_name, tensor in non_row_state.items():
            param = named_params.get(param_name)
            if param is None:
                param = named_buffers.get(param_name)
            if param is not None:
                param.data.copy_(tensor.to(param.device))

    # ------------------------------------------------------------------
    # Row-indexed personal params (merged into the global model once per eval)
    # ------------------------------------------------------------------

    def snapshot_row(self, client_models) -> dict:
        """Clone row-indexed personal params before eval merges per-client rows."""
        snapshot: dict = {}
        if not self.split.row_personal_param_names:
            return snapshot
        state = self.model.state_dict()
        for personal_state in client_models.values():
            for param_name in personal_state.keys():
                if param_name in snapshot or param_name not in state:
                    continue
                if self.is_row_personal(param_name):
                    snapshot[param_name] = state[param_name].detach().clone()
        return snapshot

    def load_row(self, snapshot) -> None:
        """Restore row-indexed personal params from a snapshot taken pre-eval."""
        if not snapshot:
            return
        state = self.model.state_dict()
        for param_name, tensor in snapshot.items():
            if param_name in state:
                state[param_name].copy_(tensor.to(state[param_name].device))
        self.model.load_state_dict(state, strict=False)

    def restore_row_into_global(self, client_models) -> None:
        """Populate the global model's row-indexed personal params from per-client
        caches (one row per client)."""
        state = self.model.state_dict()
        n_users = getattr(self.model, "n_users", None)
        if n_users is None:
            return

        for user_id, personal_state in client_models.items():
            for param_name, tensor in personal_state.items():
                if param_name not in state:
                    continue
                if self.is_row_personal(param_name):
                    param = state[param_name]
                    user_idx = self.personal_user_index(user_id)
                    row = self.stored_personal_row(param, tensor, user_idx)
                    state[param_name][user_idx] = row.to(param.device)

        self.model.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------
    # Eval session: snapshot -> merge row -> (per-user non-row swap) -> restore
    # ------------------------------------------------------------------

    def eval_session(self, client_models, split_aware):
        return _EvalSession(self, client_models, split_aware)


class _EvalSession:
    """Context manager for one federated eval pass. ``__enter__`` snapshots the
    clean global state and merges row-personal params in; ``apply_user`` swaps the
    per-user non-row params (restoring the clean baseline for cache-miss users);
    ``__exit__`` restores the global model exactly (non-row then row),
    and runs even on an exception."""

    def __init__(self, manager: PersonalStateManager, client_models, split_aware):
        self.mgr = manager
        self.client_models = client_models
        self.split_aware = split_aware
        self.named_params = dict(manager.model.named_parameters())
        self.named_buffers = dict(manager.model.named_buffers())
        self.non_row_names = set()
        self.has_non_row = False
        self.clean_non_row = {}
        self.clean_row = {}

    def __enter__(self):
        if self.split_aware:
            self.non_row_names = self.mgr.classify_non_row(self.named_params, self.named_buffers)
        self.has_non_row = len(self.non_row_names) > 0

        # Capture the post-aggregation non-row state BEFORE merging any personal params.
        self.clean_non_row = (
            self.mgr.get_clean_non_row(self.named_params, self.named_buffers, self.non_row_names)
            if self.has_non_row else {}
        )
        # Snapshot row-indexed personal params BEFORE the merge so the global model
        # can be restored after eval (the merge would otherwise persist into later rounds).
        self.clean_row = (
            self.mgr.snapshot_row(self.client_models)
            if (self.split_aware and self.client_models) else {}
        )
        # Merge row-indexed personal params (user embeddings) into the global model once.
        if self.split_aware and self.client_models:
            self.mgr.restore_row_into_global(self.client_models)
        return self

    def apply_user(self, user_id) -> None:
        """Swap per-user non-row personal params or restore the clean baseline."""
        if not (self.split_aware and self.has_non_row):
            return
        if user_id in self.client_models:
            self.mgr.apply_non_row(
                self.client_models, user_id, self.named_params, self.named_buffers, self.non_row_names
            )
        elif self.clean_non_row:
            self.mgr.load_non_row(self.clean_non_row, self.named_params, self.named_buffers)

    def __exit__(self, exc_type, exc, tb):
        if self.clean_non_row:
            self.mgr.load_non_row(self.clean_non_row, self.named_params, self.named_buffers)
        # Row restore is a whole-state load_state_dict, so it must run last.
        self.mgr.load_row(self.clean_row)
        return False
