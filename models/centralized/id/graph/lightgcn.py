# coding: utf-8
"""
LightGCN (Light Graph Convolution Network) - ported from RecBole
=================================================================

Ported directly from RecBole, preserving graph structure and convolution logic
exactly, serving as the graph learning baseline in NexusRec.

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation."
    in SIGIR 2020.

RecBole Reference Implementation:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/lightgcn.py
"""

import numpy as np
import torch
import torch.nn as nn
from core.base import EmbLoss, RecommenderBase
from core.utils import build_norm_adj_matrix


class LightGCN(RecommenderBase):
    """LightGCN baseline model - ported directly from RecBole.

    Fully preserves the RecBole graph structure and algorithm logic:
    - User-item interaction graph construction
    - Adjacency matrix normalization (Laplacian normalization)
    - Multi-layer graph convolution propagation
    - Linear aggregation of final embeddings

    Serves as the graph learning baseline in NexusRec for validating
    graph neural network processing capability.
    """
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        
        # Build interaction matrix from dataloader - consistent with RecBole data handling
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        # Core parameters - identical to RecBole defaults
        self.latent_dim = config['embedding_size']
        self.n_layers = config['num_layers']
        self.embedding_weight_decay = float(config['embedding_weight_decay'])
        self.require_pow = config['require_pow']

        # Embedding layers - same initialization as RecBole
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # Cache variables for full_sort acceleration - RecBole original design
        self.restore_user_e = None
        self.restore_item_e = None

        # Loss function
        self.reg_loss_fn = EmbLoss()

        # Build the normalized adjacency matrix - core graph structure
        self.norm_adj_matrix = build_norm_adj_matrix(
            self.interaction_matrix, self.n_users, self.n_items, self.device)

        # Weight initialization - consistent with RecBole
        self._init_weights()
    def _init_weights(self):
        """Weight initialization - consistent with RecBole."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    def get_ego_embeddings(self):
        """Retrieve initial ego embeddings - RecBole original interface."""
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def forward(self):
        """Forward pass - multi-layer graph convolution - follows RecBole original implementation."""
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        # Multi-layer graph convolution propagation
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Linear combination of all layer embeddings - core LightGCN design
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings
    
    def calculate_loss(self, interaction):
        """Calculate loss - BPR loss + regularization loss."""
        # Clear the embedding cache
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # In NexusRec, interaction is a tensor; access elements by integer index
        user = interaction[0]      # user IDs
        pos_item = interaction[1]  # positive item IDs
        neg_item = interaction[2]  # negative item IDs

        # Forward pass to obtain embeddings
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # BPR loss computation
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        # Regularization loss - using the original (ego) embeddings
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss_fn(
            u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.embedding_weight_decay * reg_loss
        return loss
    
    def predict(self, interaction):
        """Predict scores for a given interaction."""
        # Handle interaction format in NexusRec
        if isinstance(interaction, dict):
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
        else:
            user = interaction[0]
            item = interaction[1]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
    
    def full_sort_predict(self, interaction):
        """Full-sort prediction - uses cached embeddings for acceleration."""
        # Handle interaction format in NexusRec
        if isinstance(interaction, dict):
            user = interaction[self.USER_ID]
        else:
            user = interaction[0]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Clear the full-graph embedding cache on every weight reload.

        ``restore_user_e``/``restore_item_e`` are plain attributes (not in
        ``state_dict``). Clearing them here forces a recompute against loaded
        weights while preserving the within-eval cache.
        """
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self.restore_user_e = None
        self.restore_item_e = None
        return result
