# coding: utf-8
"""
SGL (Self-supervised Graph Learning) - ported from RecBole baseline
====================================================================

Self-supervised graph learning based on LightGCN, improving recommendation
performance through graph augmentation and contrastive learning.

Reference:
    Jiancan Wu et al. "Self-supervised Graph Learning for Recommendation." in SIGIR 2021.

RecBole Reference Implementation:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/sgl.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import BPRLoss, EmbLoss, RecommenderBase, xavier_normal_initialization
from core.utils import build_norm_adj_matrix


class SGL(RecommenderBase):
    """SGL baseline model - Self-supervised Graph Learning

    Based on the LightGCN architecture, augmented with self-supervised
    learning components:
    - Graph augmentation strategies (Node Dropout, Edge Dropout, Random Walk)
    - Contrastive learning loss function
    - Joint optimization of main task (BPR) + auxiliary task (SSL)
    """
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        
        # Get interaction matrix from dataloader
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        # Core parameters - aligned with RecBole
        self.latent_dim = config['embedding_size']
        self.n_layers = config['num_layers']
        self.embedding_weight_decay = float(config['embedding_weight_decay'])

        # SGL-specific parameters
        self.ssl_weight = config['ssl_weight']
        self.temperature = config['temperature']
        # RecBole SGL computes the L2 reg as ||x||_p^p (require_pow); config-driven
        # like LightGCN so the regularization mode stays in YAML, not hardcoded.
        self.require_pow = config['require_pow']
        self.aug_type = config['aug_type']
        self.drop_ratio = config['drop_ratio']

        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # Build normalized adjacency matrix
        self.norm_adj_matrix = build_norm_adj_matrix(
            self.interaction_matrix, self.n_users, self.n_items, self.device)

        # Cache variables for full_sort acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # Loss functions
        self.loss_fn = BPRLoss()
        self.reg_loss_fn = EmbLoss()

        # Weight initialization
        self.apply(xavier_normal_initialization)
    def _sparse_renormalize(self, sparse_graph):
        """Apply Laplacian re-normalization D^{-0.5} A D^{-0.5} to a sparse graph"""
        sparse_graph = sparse_graph.coalesce()
        indices = sparse_graph.indices()
        size = sparse_graph.size()
        n_nodes = size[0]

        # Compute per-node degree (number of outgoing edges)
        row_indices = indices[0]
        degree = torch.zeros(n_nodes, device=self.device)
        degree.scatter_add_(0, row_indices, torch.ones(row_indices.shape[0], device=self.device))

        # Compute D^{-0.5}, avoiding division by zero
        degree_inv_sqrt = torch.pow(degree + 1e-7, -0.5)

        # Get degree normalization values for source and destination nodes
        src_norm = degree_inv_sqrt[indices[0]]
        dst_norm = degree_inv_sqrt[indices[1]]

        # New edge weight = D^{-0.5}[src] * 1 * D^{-0.5}[dst]
        new_values = src_norm * dst_norm

        return torch.sparse_coo_tensor(indices, new_values, size).to(self.device)

    def graph_augmentation(self, graph, aug_type, drop_ratio):
        """Graph augmentation strategy"""
        if aug_type == 'ND':  # Node Dropout
            return self.node_dropout(graph, drop_ratio)
        elif aug_type == 'ED':  # Edge Dropout
            return self.edge_dropout(graph, drop_ratio)
        elif aug_type == 'RW':  # Random Walk
            return self.random_walk(graph, drop_ratio)
        else:
            return graph
    
    def node_dropout(self, graph, drop_ratio):
        """Node dropout - remove nodes and re-normalize"""
        graph = graph.coalesce()
        indices = graph.indices()
        size = graph.size()

        # Randomly select nodes to keep
        n_nodes = size[0]
        keep_mask = torch.rand(n_nodes, device=self.device) > drop_ratio

        # Only retain edges whose both endpoints are kept
        src_nodes = indices[0]
        dst_nodes = indices[1]
        edge_mask = keep_mask[src_nodes] & keep_mask[dst_nodes]

        new_indices = indices[:, edge_mask]
        # Create unnormalized adjacency matrix (edge weights = 1)
        new_values = torch.ones(new_indices.shape[1], device=self.device)

        dropped_graph = torch.sparse_coo_tensor(new_indices, new_values, size)
        # Re-apply Laplacian normalization
        return self._sparse_renormalize(dropped_graph)
    
    def edge_dropout(self, graph, drop_ratio):
        """Edge dropout - remove edges symmetrically and re-normalize.

        The adjacency is symmetric (A + A.T), so each interaction appears as two
        directed edges (u->i and i->u). Tie each edge to its transpose via a
        canonical (min,max) pair id and apply ONE keep decision per pair, so the
        dropped graph stays symmetric (matches RecBole, which drops interaction
        pairs before symmetrizing). _sparse_renormalize assumes symmetry.
        """
        graph = graph.coalesce()
        indices = graph.indices()
        size = graph.size()
        n_nodes = size[0]

        lo = torch.minimum(indices[0], indices[1])
        hi = torch.maximum(indices[0], indices[1])
        pair_id = lo * n_nodes + hi
        unique_pairs, inverse = torch.unique(pair_id, return_inverse=True)
        pair_keep = torch.rand(unique_pairs.shape[0], device=self.device) > drop_ratio
        keep_mask = pair_keep[inverse]

        new_indices = indices[:, keep_mask]
        # Create unnormalized adjacency matrix (edge weights = 1)
        new_values = torch.ones(new_indices.shape[1], device=self.device)

        dropped_graph = torch.sparse_coo_tensor(new_indices, new_values, size)
        # Re-apply Laplacian normalization
        return self._sparse_renormalize(dropped_graph)
    
    def random_walk(self, graph, drop_ratio):
        """Random walk augmentation - not implemented."""
        raise NotImplementedError(
            "Random Walk augmentation is not implemented in NexusRec. "
            "Use aug_type='ED' (edge dropout) or 'ND' (node dropout) instead."
        )
    
    def get_ego_embeddings(self):
        """Get initial (ego) embeddings"""
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def forward(self, graph=None):
        """Forward pass - graph convolution"""
        if graph is None:
            graph = self.norm_adj_matrix

        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        # Multi-layer graph convolution
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(graph, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Linear combination of all-layer embeddings
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings
    
    def ssl_loss(self, user_emb1, user_emb2, item_emb1, item_emb2,
                 user_all_emb2, item_all_emb2):
        """Self-supervised contrastive loss - uses all users/items as negatives (consistent with RecBole)"""
        # Normalize batch embeddings
        user_emb1 = F.normalize(user_emb1, dim=1)
        user_emb2 = F.normalize(user_emb2, dim=1)
        item_emb1 = F.normalize(item_emb1, dim=1)
        item_emb2 = F.normalize(item_emb2, dim=1)

        # Normalize all embeddings for negative computation
        all_user2_norm = F.normalize(user_all_emb2, dim=1)
        all_item2_norm = F.normalize(item_all_emb2, dim=1)

        # User contrastive loss - using all users as negatives
        pos_score_user = torch.sum(user_emb1 * user_emb2, dim=1)
        ttl_score_user = torch.matmul(user_emb1, all_user2_norm.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.temperature)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.temperature), dim=1)
        ssl_loss_user = -torch.mean(torch.log(pos_score_user / ttl_score_user))

        # Item contrastive loss - using all items as negatives
        pos_score_item = torch.sum(item_emb1 * item_emb2, dim=1)
        ttl_score_item = torch.matmul(item_emb1, all_item2_norm.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.temperature)
        ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.temperature), dim=1)
        ssl_loss_item = -torch.mean(torch.log(pos_score_item / ttl_score_item))

        return ssl_loss_user + ssl_loss_item
    
    def calculate_loss(self, interaction):
        """Calculate total loss = BPR loss + self-supervised loss"""
        # Clear cache
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        # Embeddings from original graph
        user_all_embeddings, item_all_embeddings = self.forward()

        # Compute BPR loss
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        # Route through BPRLoss so the supervised pairwise term uses the shared
        # numerical guard.
        mf_loss = self.loss_fn(pos_scores, neg_scores)

        # Regularization loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss_fn(
            u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow
        )

        # Self-supervised loss - uses all users/items as negatives (consistent with RecBole)
        ssl_loss = 0
        if self.ssl_weight > 0:
            # Create two augmented graphs
            aug_graph1 = self.graph_augmentation(self.norm_adj_matrix, self.aug_type, self.drop_ratio)
            aug_graph2 = self.graph_augmentation(self.norm_adj_matrix, self.aug_type, self.drop_ratio)

            # Get embeddings from augmented graphs
            user_all_emb1, item_all_emb1 = self.forward(aug_graph1)
            user_all_emb2, item_all_emb2 = self.forward(aug_graph2)

            # Extract batch sample embeddings
            user_emb1 = user_all_emb1[user]
            user_emb2 = user_all_emb2[user]
            item_emb1 = item_all_emb1[pos_item]
            item_emb2 = item_all_emb2[pos_item]

            # Compute self-supervised loss (using full embeddings as negatives)
            ssl_loss = self.ssl_loss(user_emb1, user_emb2, item_emb1, item_emb2,
                                     user_all_emb2, item_all_emb2)

        return mf_loss + self.embedding_weight_decay * reg_loss + self.ssl_weight * ssl_loss
    
    def predict(self, interaction):
        """Predict scores"""
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
        """Full-sort prediction"""
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
        ``state_dict``); loading the validation-best weights for final-test eval
        would otherwise score the loaded model on last-epoch embeddings whenever
        ``best_epoch != last_epoch``. Forces a recompute against loaded weights.
        """
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self.restore_user_e = None
        self.restore_item_e = None
        return result
