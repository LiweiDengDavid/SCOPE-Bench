# coding: utf-8
 

"""
RecommenderBase - Unified base class for all recommendation models
==================================================================

Provides a single base class for all paradigms (centralized, federated,
multimodal, sequential). Inheriting models only need to implement
``forward()`` and ``calculate_loss()``.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ..model_registry import infer_paradigm


class RecommenderBase(nn.Module, ABC):
    """Unified base class for all recommendation models.

    Provides device management, field mappings, and user/item counts.
    Multimodal features (v_feat, t_feat) are None by default; models that
    need them should call ``setup_multimodal_features()`` in their __init__.
    """

    # Whether calculate_loss consumes ALL K negative rows the dataloader emits
    # (interaction[2:]). Single-negative models read only interaction[2], so the
    # trainer fails fast when num_negatives>1 and this is False (see train_single).
    supports_multi_negatives = False

    def __init__(self, config, dataloader):
        super().__init__()

        self.config = config
        self.device = config["device"]

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID

        if hasattr(dataloader, 'dataset'):
            self.n_users = dataloader.dataset.get_user_num()
            self.n_items = dataloader.dataset.get_item_num()
        else:
            self.n_users = config["n_users"]
            self.n_items = config["n_items"]

        self.batch_size = config["train_batch_size"]
        self.embed_size = config["embedding_size"]
        self.dropout_rate = config["dropout_rate"]

        # Multimodal features: None by default, loaded explicitly by models that need them.
        self.v_feat = None
        self.t_feat = None

    def get_resume_state(self):
        """Model-level training state to checkpoint for resume.

        Override in models that carry training-loop state OUTSIDE state_dict (plain
        Python attributes, not params/buffers) — e.g. RecVAE's alternating-phase
        counters. The default has none. Restored via set_resume_state BEFORE the
        optimizer is rebuilt so phase-dependent param groups line up.
        """
        return {}

    def set_resume_state(self, state):
        """Restore model-level training state captured by get_resume_state (no-op by default)."""
        return None

    def setup_multimodal_features(self, config=None):
        """Load centralized multimodal features (visual and text embeddings)."""
        runtime_config = config or self.config
        if runtime_config["is_federated"]:
            self.v_feat = None
            self.t_feat = None
            return

        from ..data.features import setup_centralized_features

        setup_centralized_features(runtime_config, self)
    
    def to(self, device):
        """Move model and multimodal feature tensors to the given device."""
        super().to(device)
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            self.v_feat = self.v_feat.to(device)
        if hasattr(self, 't_feat') and self.t_feat is not None:
            self.t_feat = self.t_feat.to(device)
        self.device = device
        return self
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Core forward pass. Must be implemented by every model."""
        raise NotImplementedError

    @abstractmethod
    def calculate_loss(self, *args, **kwargs):
        """Compute training loss. Must be implemented by every model."""
        raise NotImplementedError

    def full_sort_predict(self, interaction, *args, **kwargs):
        """Return scores for all items for each user in the batch.

        Default implementation calls ``predict()`` once per user in a Python
        loop — O(batch_size) forward passes. Models should override this with
        a vectorized implementation for acceptable evaluation speed.

        Returns:
            Tensor of shape [batch_size, n_items].
        """
        users = interaction[0]
        if len(users.shape) > 0 and users.shape[0] > 1:
            all_scores = []
            for user_id in users:
                user_tensor = user_id.unsqueeze(0).expand(self.n_items)
                item_tensor = torch.arange(self.n_items).to(self.device)
                all_scores.append(self.predict(user_tensor, item_tensor))
            return torch.stack(all_scores)
        else:
            items = torch.arange(self.n_items).to(self.device)
            return self.predict(users.expand(self.n_items), items).unsqueeze(0)

    def predict(self, users, items, *args, **kwargs):
        """Score (user, item) pairs. Default delegates to forward()."""
        with torch.no_grad():
            return self.forward(users, items, *args, **kwargs)

    def pre_epoch_processing(self):
        """Optional hook called before each training epoch."""
        pass

    def post_epoch_processing(self):
        """Optional hook called after each training epoch."""
        pass

    def get_optimizer_params(self):
        """Return parameter groups for the optimizer.

        Override to use per-layer learning rates or weight-decay exclusions.
        """
        return self.parameters()

    def get_shared_parameters(self):
        """Return parameters to be aggregated across federated clients.

        Override in federated models. Keys must match ``state_dict()`` names.
        A federated model MUST declare a non-empty shared set: the empty default
        means "not split-aware", which the aggregator (begin_round) rejects with an
        assertion — plain all-parameter FedAvg is not supported by leaving this empty.
        """
        return {}

    def get_personal_parameters(self):
        """Return parameters that stay local to each federated client.

        Override in federated models.  Keys must match ``state_dict()`` names.
        Returns empty dict by default (no personal params).
        """
        return {}

    def get_server_grad_param_names(self):
        """Return names of shared params that use server-side delta aggregation.

        Per FedVLR Algorithm 1, D (item embeddings) and γ_j (fusion operator
        params) are aggregated via weighted-average of client deltas on the
        server, NOT via standard FedAvg weight averaging.

        Override in multimodal federated models. Non-MM models return [].
        """
        return []

    def get_row_personal_parameter_names(self):
        """Return names of personal params indexed per-user (one row = one client),
        to disambiguate when shape-based classification is ambiguous (n_items ==
        n_users). ``None`` (default) defers to ParameterSplit's nn.Embedding-based
        resolution; override to declare explicitly.
        """
        return None

    def get_regularization_loss(self):
        """Return an additional regularization term to add to the loss."""
        return 0.0

    def __str__(self):
        """Human-readable summary: module tree + trainable parameter count."""
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])

        lines = []
        paradigm = infer_paradigm(
            {
                "is_federated": self.config["is_federated"],
                "is_multimodal_model": self.config["is_multimodal_model"],
                "is_sequential": self.config["is_sequential"],
            }
        )
        lines.append(f"{self.__class__.__name__} (paradigm={paradigm})")
        lines.append("-" * 60)

        for name, module in self.named_children():
            module_str = str(module)
            if '\n' in module_str:
                module_lines = module_str.split('\n')
                lines.append(f"({name}): {module_lines[0]}")
                for line in module_lines[1:]:
                    lines.append(f"  {line}")
            else:
                lines.append(f"({name}): {module_str}")

        lines.append("-" * 60)
        lines.append(f"Trainable parameters: {params:,}")

        return "\n".join(lines)
