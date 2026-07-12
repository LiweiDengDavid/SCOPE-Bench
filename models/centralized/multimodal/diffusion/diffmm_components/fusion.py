# coding: utf-8
"""
DiffMM User Feature Aggregation Module
======================================

Aggregates item features over each user's interaction history to produce user
feature embeddings used by DiffMM's inline modality fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserFeatureAggregator(nn.Module):
    """User feature aggregator.

    Aggregates item features based on the user's interaction history to generate user feature embeddings.
    """

    def __init__(self,
                 embedding_size: int,
                 aggregation_type: str = "mean",
                 use_attention: bool = False):
        """Initialize the user feature aggregator.

        Args:
            embedding_size: embedding dimension
            aggregation_type: aggregation type ("mean", "max", "attention", "weighted")
            use_attention: whether to use attention mechanism
        """
        super(UserFeatureAggregator, self).__init__()

        self.embedding_size = embedding_size
        self.aggregation_type = aggregation_type
        self.use_attention = use_attention

        if self.aggregation_type == "attention" or self.use_attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(embedding_size, embedding_size // 2),
                nn.ReLU(),
                nn.Linear(embedding_size // 2, 1)
            )

        if self.aggregation_type == "weighted":
            self.weight_layer = nn.Sequential(
                nn.Linear(embedding_size, 1),
                nn.Sigmoid()
            )

    def forward(self,
                item_features: torch.Tensor,
                user_item_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            item_features: item feature embeddings [n_items, embedding_size]
            user_item_matrix: user-item interaction matrix [n_users, n_items]

        Returns:
            torch.Tensor: user feature embeddings [n_users, embedding_size]
        """
        if self.aggregation_type == "mean":
            return self._mean_aggregation(item_features, user_item_matrix)
        elif self.aggregation_type == "max":
            return self._max_aggregation(item_features, user_item_matrix)
        elif self.aggregation_type == "attention":
            return self._attention_aggregation(item_features, user_item_matrix)
        elif self.aggregation_type == "weighted":
            return self._weighted_aggregation(item_features, user_item_matrix)
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

    def _mean_aggregation(self, item_features, user_item_matrix):
        """Mean aggregation."""
        # Compute number of interactions per user
        interaction_counts = user_item_matrix.sum(dim=1, keepdim=True)
        interaction_counts = torch.clamp(interaction_counts, min=1)  # avoid division by zero

        # Weighted sum
        user_features = torch.mm(user_item_matrix, item_features)

        # Normalize
        user_features = user_features / interaction_counts

        return user_features

    def _max_aggregation(self, item_features, user_item_matrix):
        """Max aggregation."""
        n_users, n_items = user_item_matrix.shape
        user_features = torch.zeros(n_users, self.embedding_size, device=item_features.device)

        for user_id in range(n_users):
            interacted_items = user_item_matrix[user_id].nonzero().squeeze(-1)
            if len(interacted_items) > 0:
                user_item_features = item_features[interacted_items]
                user_features[user_id] = user_item_features.max(dim=0)[0]

        return user_features

    def _attention_aggregation(self, item_features, user_item_matrix):
        """Attention aggregation."""
        n_users, n_items = user_item_matrix.shape
        user_features = torch.zeros(n_users, self.embedding_size, device=item_features.device)

        for user_id in range(n_users):
            interacted_items = user_item_matrix[user_id].nonzero().squeeze(-1)
            if len(interacted_items) > 0:
                user_item_features = item_features[interacted_items]  # [n_interacted, embedding_size]

                # Compute attention weights
                attention_scores = self.attention_layer(user_item_features)  # [n_interacted, 1]
                attention_weights = F.softmax(attention_scores, dim=0)

                # Weighted aggregation
                user_features[user_id] = (attention_weights * user_item_features).sum(dim=0)

        return user_features

    def _weighted_aggregation(self, item_features, user_item_matrix):
        """Weighted aggregation."""
        # Compute weights
        item_weights = self.weight_layer(item_features).squeeze(-1)  # [n_items]

        # Apply weights to interaction matrix
        weighted_matrix = user_item_matrix * item_weights.unsqueeze(0)

        # Weighted sum
        user_features = torch.mm(weighted_matrix, item_features)

        # Normalize
        weight_sums = weighted_matrix.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        user_features = user_features / weight_sums

        return user_features
