# coding: utf-8
 
"""
FedAvg for Recommendation
==========================

Federated Averaging applied to collaborative filtering.

Architecture:
- Shared (aggregated globally):  item_commonality embedding
- Personal (stays per-client):   user_embedding + affine_output

Each client trains on its own user's data. The item embedding is
averaged across clients each round (standard FedAvg). The user
embedding and output layer are personal and never leave the client.
"""

import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization


class FedAvg(RecommenderBase):
    """Federated Averaging recommendation model.

    Shared parameters (FedAvg-aggregated): item_commonality.
    Personal parameters (not aggregated):  user_embedding, affine_output.
    """

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        self.user_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.item_commonality = nn.Embedding(self.n_items, self.embed_size)
        self.affine_output = nn.Linear(self.embed_size * 2, 1)
        self.logistic = nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score (user, item) pairs."""
        user_emb = self.user_embedding(users)
        item_emb = self.item_commonality(items)
        pred = self.affine_output(torch.cat([user_emb, item_emb], dim=-1))
        return self.logistic(pred).squeeze(-1)

    def calculate_loss(self, batch):
        """Binary cross-entropy loss on positive/negative item pairs."""
        users, pos_items, neg_items = batch[0], batch[1], batch[2]
        items = torch.cat([pos_items, neg_items])
        users_exp = torch.cat([users, users])
        labels = torch.cat([
            torch.ones(pos_items.size(0), device=self.device),
            torch.zeros(neg_items.size(0), device=self.device),
        ])
        # BCE matches the probability-valued output returned by forward().
        return nn.functional.binary_cross_entropy(self.forward(users_exp, items), labels)

    def full_sort_predict(self, interaction, *args, **kwargs):
        """Return scores for all items for each user in the batch."""
        users = interaction[0]
        items = torch.arange(self.n_items, device=self.device)

        if users.dim() > 0 and users.shape[0] > 1:
            all_scores = []
            for user in users:
                u = user.unsqueeze(0).expand(self.n_items)
                all_scores.append(self.forward(u, items))
            return torch.stack(all_scores)

        u = users.expand(self.n_items) if users.dim() > 0 else users.unsqueeze(0).expand(self.n_items)
        return self.forward(u, items).unsqueeze(0)

    def get_shared_parameters(self):
        """Item embedding is aggregated by the server each round."""
        return {"item_commonality.weight": self.item_commonality.weight}

    def get_personal_parameters(self):
        """User embedding and output head stay local to each client."""
        return {
            "user_embedding.weight": self.user_embedding.weight,
            "affine_output.weight": self.affine_output.weight,
            "affine_output.bias": self.affine_output.bias,
        }
