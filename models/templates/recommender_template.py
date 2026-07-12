# coding: utf-8
"""
MyModel - One-line description
==============================

Longer description of what the model does and its key innovation.

Paper: Author et al. "Paper Title." in VENUE YEAR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class MyModel(RecommenderBase):
    """Brief class description.

    Reference: Author et al. VENUE YEAR.
    """

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        # --- hyper-parameters (read from config) ---
        self.embedding_size = config["embedding_size"]

        # --- layers ---
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self._init_weights()

        # Uncomment for multimodal models:
        # self.setup_multimodal_features()  # loads self.v_feat, self.t_feat

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    # ------------------------------------------------------------------
    # Required: forward pass
    # ------------------------------------------------------------------

    def forward(self, users, items):
        """Compute user and item representations."""
        user_e = self.user_embedding(users)
        item_e = self.item_embedding(items)
        return user_e, item_e

    # ------------------------------------------------------------------
    # Required: training loss
    # ------------------------------------------------------------------

    def calculate_loss(self, interaction):
        """Compute training loss from a batch.

        Args:
            interaction: Tensor of shape [3, batch_size] — (user, pos_item, neg_item).
                         For federated models the format may differ; check the
                         dataloader documentation.

        Returns:
            Scalar loss tensor.
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    # ------------------------------------------------------------------
    # Required for evaluation: full-sort prediction
    # ------------------------------------------------------------------

    def full_sort_predict(self, interaction):
        """Return scores over ALL items for each user in the batch.

        This must be vectorized — never loop over users here.

        Args:
            interaction: Tensor [3+, batch_size] or dict-batch.

        Returns:
            Tensor of shape [batch_size, n_items].
        """
        if isinstance(interaction, dict):
            user = interaction[self.USER_ID]
        else:
            user = interaction[0]

        user_e = self.user_embedding(user)            # [B, D]
        all_item_e = self.item_embedding.weight       # [N, D]
        scores = torch.matmul(user_e, all_item_e.t()) # [B, N]
        return scores
