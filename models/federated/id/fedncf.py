# coding: utf-8
"""
FedNCF: Federated Neural Collaborative Filtering
"""

import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization


class FedNCF(RecommenderBase):
    """Federated NCF with server-side item towers and client-side user heads."""

    def __init__(self, config, dataloader):
        super(FedNCF, self).__init__(config, dataloader)

        self.embedding_size = config['embedding_size']
        self.latent_dim_mf = self.embedding_size
        self.latent_dim_mlp = self.embedding_size

        # MLP path embedding layers
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim_mlp)

        # MF path embedding layers
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim_mf)

        # MLP layer configuration
        layers = [2 * self.latent_dim_mlp, self.latent_dim_mlp, self.latent_dim_mlp // 2, self.latent_dim_mlp // 4]

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1] + self.latent_dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # Apply initialization
        self.apply(xavier_normal_initialization)

    def get_shared_parameters(self):
        """Shared parameters that should participate in global aggregation.

        The server aggregates item-side embeddings and interaction layers. User
        embeddings and the prediction head remain client-specific so evaluation
        can restore each client's personal state without exporting user factors.
        """
        shared = {
            "embedding_item_mlp.weight": self.embedding_item_mlp.weight,
            "embedding_item_mf.weight": self.embedding_item_mf.weight,
        }
        for idx, layer in enumerate(self.fc_layers):
            shared[f"fc_layers.{idx}.weight"] = layer.weight
            shared[f"fc_layers.{idx}.bias"] = layer.bias
        return shared

    def get_personal_parameters(self):
        """Personal user-side parameters kept on each client (user embeddings + prediction head)."""
        return {
            "embedding_user_mlp.weight": self.embedding_user_mlp.weight,
            "embedding_user_mf.weight": self.embedding_user_mf.weight,
            "affine_output.weight": self.affine_output.weight,
            "affine_output.bias": self.affine_output.bias,
        }

    def forward(self, user_indices, item_indices):
        """Forward pass - unified interface."""
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # MLP path
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        # MF path
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # Pass through MLP layers
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # Fuse MLP and MF paths
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def calculate_loss(self, interaction):
        """Calculate loss - unified interface."""
        user, poss, negs = interaction[0], interaction[1], interaction[2]
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32, device=self.device)
        ratings[:poss.size(0)] = 1
        users = torch.full_like(items, user[0], dtype=torch.long, device=self.device)

        pred = self.forward(users, items)
        return nn.BCELoss()(pred.view(-1), ratings)

    def full_sort_predict(self, interaction, *args, **kwargs):
        """Full-sort prediction - unified interface."""
        # ID-based model does not use multimodal features; ignore extra arguments
        user = interaction[0]
        if user.dim() == 0:
            user = user.unsqueeze(0)
        users = user.expand(self.n_items)
        items = torch.arange(self.n_items, device=self.device)
        scores = self.forward(users, items)
        return scores.view(1, -1)
