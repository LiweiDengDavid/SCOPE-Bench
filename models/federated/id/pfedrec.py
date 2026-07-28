# coding: utf-8
"""
PFedRec: Personalized Federated Recommendation

Original paper: Tan, C. et al. (2023). Personalized Federated Learning towards
Communication Efficiency, Robustness and Fairness in Recommendation Systems.

Key design: Dual personalization — shared item embeddings (global) + personal
prediction head (per-client). The prediction head is implemented as user-indexed
Embeddings so that per-user weights are restorable by the federated eval protocol
in FederatedTrainer._restore_personal_state().
"""

import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization


class PFedRec(RecommenderBase):
    """PFedRec recommendation model.

    Personal prediction head: each user has their own weight vector and bias
    stored as row-indexed Embeddings (shape [n_users, embed_size] and
    [n_users, 1]). This allows FederatedTrainer to restore each user's
    personal head via _restore_personal_state() during evaluation.
    """

    def __init__(self, config, dataloader):
        super(PFedRec, self).__init__(config, dataloader)
        self.config['server_learning_rate'] = self.config['learning_rate'] * self.n_items

        self.embed_size = config['embedding_size']

        # Shared item representation aggregated by the server.
        self.item_embed = nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embed_size
        )

        # Personal prediction head: user-indexed weights and bias.
        # Each row corresponds to one user's personal head parameters.
        # Shape: [n_users, embed_size] and [n_users, 1].
        # Row indexing lets the federated evaluator restore one user's head at a time.
        self.user_predictor_weight = nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embed_size
        )
        self.user_predictor_bias = nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=1
        )

        self.logistic = nn.Sigmoid()
        self.apply(xavier_normal_initialization)

    def get_server_grad_param_names(self):
        """item_embed uses delta aggregation (server-gradient path)."""
        return ['item_embed.weight']

    def get_shared_parameters(self):
        """Shared global item parameters -- aggregated across all clients."""
        return {
            "item_embed.weight": self.item_embed.weight,
        }

    def get_personal_parameters(self):
        """Client-specific prediction head (row-indexed by user_id)."""
        return {
            "user_predictor_weight.weight": self.user_predictor_weight.weight,
            "user_predictor_bias.weight": self.user_predictor_bias.weight,
        }

    def forward(self, users, items):
        """Compute interaction scores for a batch of (user, item) pairs.

        Args:
            users: LongTensor [batch] -- user IDs
            items: LongTensor [batch] -- item IDs

        Returns:
            Tensor [batch] -- predicted interaction probability
        """
        item_embed = self.item_embed(items)              # [batch, embed_size]
        w = self.user_predictor_weight(users)            # [batch, embed_size]
        b = self.user_predictor_bias(users)              # [batch, 1]
        # Element-wise dot product + personal bias
        pred = (item_embed * w).sum(dim=-1, keepdim=True) + b   # [batch, 1]
        return self.logistic(pred).squeeze(-1)           # [batch]

    def calculate_loss(self, interaction):
        """BCE loss over positive and negative samples."""
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]

        all_items = torch.cat([pos_items, neg_items])           # [2*batch]
        all_users = torch.cat([users, users])                   # [2*batch]
        labels = torch.cat([
            torch.ones(pos_items.size(0), device=self.device),
            torch.zeros(neg_items.size(0), device=self.device),
        ])

        pred = self.forward(all_users, all_items)               # [2*batch]
        # Loss: BCE, matching the reference implementation for this model.
        return nn.BCELoss()(pred, labels)

    def full_sort_predict(self, interaction, *args, **kwargs):
        """Full-ranking prediction for all items.

        Returns scores shaped [batch_size, n_items]. Each user is scored with
        their own personal prediction head restored by
        FederatedTrainer._restore_personal_state().
        """
        users = interaction[0]                                  # [batch]
        items = torch.arange(self.n_items, device=self.device) # [n_items]

        item_embed = self.item_embed(items)                     # [n_items, embed_size]
        w = self.user_predictor_weight(users)                   # [batch, embed_size]
        b = self.user_predictor_bias(users)                     # [batch, 1]

        # [batch, n_items] = [batch, embed_size] @ [embed_size, n_items]
        scores = torch.matmul(w, item_embed.T) + b              # [batch, n_items]
        return self.logistic(scores)
