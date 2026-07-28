# coding: utf-8
"""
BPR (Bayesian Personalized Ranking) - Ported from RecBole
==========================================================

Ported directly from RecBole, keeping the algorithm fully consistent,
serving as the collaborative filtering baseline for NexusRec.

Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback."
    in UAI 2009.

RecBole Reference Implementation:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/bpr.py
"""

import torch
import torch.nn as nn
from core.base import RecommenderBase


class BPR(RecommenderBase):
    """BPR baseline model - ported directly from RecBole.

    Fully preserves the original RecBole algorithm logic:
    - User and item embedding layers
    - Pairwise ranking loss (BPR Loss)
    - Inner-product prediction

    Serves as the collaborative filtering baseline in the NexusRec framework for performance comparison.
    """

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        # Directly ported core components from RecBole
        self.embedding_size = config['embedding_size']

        # User and item embedding layers - identical to RecBole
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Xavier normal initialization, same as RecBole
        self._init_weights()

    def _init_weights(self):
        """Weight initialization - consistent with RecBole."""
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def get_user_embedding(self, user):
        """Get user embedding - RecBole original interface."""
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        """Get item embedding - RecBole original interface."""
        return self.item_embedding(item)

    def forward(self, user, item):
        """Forward pass - implemented exactly as in the original RecBole."""
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """Calculate BPR loss - preserves original RecBole logic.

        Args:
            interaction: interaction data tensor [3, batch_size] containing user_id, pos_item, neg_item

        Returns:
            BPR loss value
        """
        # In the NexusRec framework, interaction is a tensor; use integer indexing to access fields
        user = interaction[0]      # user IDs
        pos_item = interaction[1]  # positive item IDs
        neg_item = interaction[2]  # negative item IDs

        # Retrieve embeddings - same computation flow as RecBole
        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)

        # Compute scores - inner-product prediction
        pos_scores = torch.mul(user_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(user_e, neg_e).sum(dim=1)

        # BPR loss - exactly following the RecBole formula
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        return loss

    def predict(self, interaction):
        """Predict scores - RecBole original interface."""
        # Handle interaction format in the NexusRec framework
        if isinstance(interaction, dict):
            # Handle dict-based interaction batches.
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
        else:
            # Tensor format in production
            user = interaction[0]
            item = interaction[1]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        """Full-sort prediction - consistent with RecBole."""
        # Handle interaction format in the NexusRec framework
        if isinstance(interaction, dict):
            # Handle dict-based interaction batches.
            user = interaction[self.USER_ID]
        else:
            # Tensor format in production
            user = interaction[0]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return scores
