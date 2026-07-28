"""Federated parameter-split resolution.

Static classification of model parameters into shared / personal / row-personal
/ server-gradient sets, plus the orphan and server-grad validations. Pure name
bookkeeping resolved once at trainer construction — no torch-state mutation.
"""

from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn


class ParameterSplit:
    """Resolve and hold the federated parameter classification for a model."""

    def __init__(self, model, config, logger):
        state_keys = set(model.state_dict().keys())
        self.shared_param_names, self.personal_param_names = self._resolve_split(
            model, state_keys, logger
        )
        self.split_aware = bool(self.shared_param_names)
        self.row_personal_param_names = self.resolve_row_personal(
            model, self.personal_param_names
        )
        self.server_grad_param_names = self._resolve_server_grad(
            model, state_keys, config, logger, self.shared_param_names, self.split_aware
        )
        self._validate_orphans(
            model, self.shared_param_names, self.personal_param_names, self.split_aware
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(model, accessor_name: str, state_keys: Iterable[str], logger) -> List[str]:
        accessor = getattr(model, accessor_name, None)
        if not callable(accessor):
            return []

        params = accessor()
        if not isinstance(params, dict):
            logger.warning(
                "%s must return a dict, got %s",
                accessor_name, type(params).__name__,
            )
            return []

        state_key_set = set(state_keys)
        return [name for name in params.keys() if name in state_key_set]

    @classmethod
    def _resolve_split(cls, model, state_keys, logger):
        shared_names = cls._extract(model, "get_shared_parameters", state_keys, logger)
        personal_names = cls._extract(model, "get_personal_parameters", state_keys, logger)

        overlap = set(shared_names) & set(personal_names)
        if overlap:
            logger.warning(
                "Overlapping federated parameter split detected: %s",
                sorted(overlap),
            )
            personal_names = [name for name in personal_names if name not in overlap]

        return shared_names, personal_names

    @staticmethod
    def resolve_row_personal(model, personal_param_names) -> set:
        """Resolve which personal params are indexed per-user (one row = one client).

        Row-personal params (e.g. user embeddings) are snapshotted/restored a
        single row at a time; all other personal params (item tables, affine
        heads, fusion routers) are kept whole per client. Classifying by
        ``tensor.shape[0] == n_users`` is ambiguous when ``n_items == n_users``
        (a ``[n_items, *]`` table is indistinguishable from a ``[n_users, *]``
        one by shape), so we identify user-indexed tables by the ``nn.Embedding``
        module whose ``num_embeddings == n_users``. A model may override the
        optional ``get_row_personal_parameter_names()`` to declare this
        explicitly; if the shapes are genuinely ambiguous and no declaration is
        provided, validation raises before per-client state is built.

        Static so the trainer can delegate to it on minimal model fixtures (the resolver is
        a pure function of model + the personal name list).
        """
        n_users = getattr(model, "n_users", None)
        if n_users is None or not personal_param_names:
            return set()
        personal = set(personal_param_names)

        declared = getattr(model, "get_row_personal_parameter_names", None)
        if callable(declared):
            names = declared()
            if names is not None:
                return set(names) & personal

        n_items = getattr(model, "n_items", None)
        if n_items is not None and n_items == n_users:
            raise ValueError(
                f"{type(model).__name__}: cannot disambiguate row-per-user "
                f"personal parameters when n_items == n_users (={n_users}); a "
                "[n_items, *] personal table is indistinguishable from a "
                "[n_users, *] one by shape. Implement "
                "get_row_personal_parameter_names() to declare which personal "
                "parameters are indexed per-user."
            )

        row_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings == n_users:
                weight_name = f"{module_name}.weight" if module_name else "weight"
                if weight_name in personal:
                    row_names.add(weight_name)
        return row_names

    @staticmethod
    def _resolve_server_grad(model, state_keys, config, logger, shared_names, split_aware):
        """Server-gradient (delta) aggregated param names + their validations."""
        sgp_accessor = getattr(model, "get_server_grad_param_names", None)
        if callable(sgp_accessor):
            names = list(sgp_accessor())
            missing = [n for n in names if n not in state_keys]
            if missing:
                # Mirror _validate_orphans: a typo/stale declared name must fail
                # loud — silently dropping it would fall back to FedAvg
                # weight-averaging instead of the intended server-LR delta rule.
                raise ValueError(
                    f"Model {config['model']}: get_server_grad_param_names() "
                    f"declares parameters missing from the model state_dict: "
                    f"{sorted(missing)}. Server-gradient declarations must use "
                    f"exact state_dict() keys."
                )
        else:
            names = []

        if names:
            if "server_learning_rate" not in config:
                raise ValueError(
                    f"Model {config['model']} declares server-gradient-aggregated "
                    f"params {names} but 'server_learning_rate' is not set in config. "
                    f"Each algorithm using delta aggregation must explicitly define "
                    f"its server_learning_rate."
                )
            logger.debug("Server-gradient-aggregated params (delta path): %s", names)

        if names and split_aware:
            server_grad_set = set(names)
            not_shared = server_grad_set - set(shared_names)
            if not_shared:
                raise ValueError(
                    f"server_grad_param_names {sorted(not_shared)} are not in "
                    f"shared_param_names. Every server-gradient-aggregated param "
                    f"must also be declared as shared."
                )
            weight_avg_shared = [n for n in shared_names if n not in server_grad_set]
            if weight_avg_shared:
                logger.info(
                    "Mixed aggregation: %d params delta-aggregated, %d params weight-averaged",
                    len(names), len(weight_avg_shared),
                )
        return names

    @staticmethod
    def _validate_orphans(model, shared_names, personal_names, split_aware):
        """Every parameter must be covered by shared ∪ personal.

        Parameters outside both sets cannot contribute to the federated update,
        so they indicate an invalid split declaration.
        """
        if not split_aware:
            return
        declared = set(shared_names) | set(personal_names)
        all_params = {name for name, _ in model.named_parameters()}
        orphans = all_params - declared
        if orphans:
            raise ValueError(
                "Federated parameter split is INCOMPLETE — the following parameters "
                "are neither shared nor personal and would be frozen at initialization: "
                f"{sorted(orphans)}. Fix get_shared_parameters()/get_personal_parameters() "
                "so they together cover every model parameter."
            )

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    @staticmethod
    def is_row_delta(model, name, tensor) -> bool:
        """Only 2-D embedding weights receive sparse per-row delta normalization."""
        module_name, _, param_name = name.rpartition(".")
        if param_name != "weight":
            return False
        module = model.get_submodule(module_name) if module_name else model
        return isinstance(module, nn.Embedding) and tensor.dim() == 2
