# coding: utf-8
"""
FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.utils import build_norm_adj_matrix


def mm_adj_cache_filename(knn_k, mm_image_weight, vision_feature_file, text_feature_file):
    """Cache key for the blended mm graph: every input that determines it.

    The weight is encoded via str(float) — a lossless float repr — so nearby
    values (0.10/0.15/0.19) never alias to the same cache file; the feature file
    names tie the cache to the embeddings it was built from. Mirrors DRAGON.
    """
    return 'mm_adj_freedomdsp_{}_{}_{}_{}.pt'.format(
        knn_k, mm_image_weight, vision_feature_file, text_feature_file
    )


class FREEDOM(RecommenderBase):
    """FREEDOM model with fixed item-item graphs and denoised user-item graphs."""

    def __init__(self, config, dataloader):
        super(FREEDOM, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # Base parameters
        self.embedding_dim = config['embedding_size']
        self.feature_embedding_size = config['feature_embedding_size']
        # calculate_loss mixes CF-space user embeddings with projected modality
        # features in a single BPR term, so the two spaces must be equal-dim.
        if self.embedding_dim != self.feature_embedding_size:
            raise ValueError(
                "FREEDOM requires embedding_size == feature_embedding_size "
                f"(got {self.embedding_dim} vs {self.feature_embedding_size})."
            )
        self.knn_k = config['knn_k']
        self.num_layers = config['num_mm_layers']
        self.n_ui_layers = config['num_ui_layers']
        # Modality-loss coefficient, decoupled from optimizer L2.
        self.reg_weight = float(config['reg_weight'])
        self.mm_image_weight = config['mm_image_weight']
        self.dropout_rate = config['dropout_rate']
        self.n_nodes = self.n_users + self.n_items

        # Build interaction matrix and adjacency matrix
        self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = build_norm_adj_matrix(
            self.interaction_matrix, self.n_users, self.n_items, self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Multimodal feature processing - set up automatically
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(
            dataset_path,
            mm_adj_cache_filename(
                self.knn_k,
                self.mm_image_weight,
                config['features']['vision_feature_file'],
                config['features']['text_feature_file'],
            ),
        )

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feature_embedding_size)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feature_embedding_size)

        # Build multimodal adjacency matrix
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location="cpu", weights_only=False)

        if self.mm_adj is None:
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        if self.mm_adj is not None:
            self.mm_adj = self.mm_adj.to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        """Build the KNN adjacency matrix."""
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        """Compute the normalized Laplacian matrix."""
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size, dtype=torch.float32)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size, dtype=torch.float32)
    def pre_epoch_processing(self):
        """Pre-epoch setup."""
        if self.dropout_rate <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout_rate))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj.shape, dtype=torch.float32).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        """Normalize the adjacency matrix."""
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size, dtype=torch.float32)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        """Retrieve edge indices and normalized edge values."""
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).to(dtype=torch.long)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj=None):
        """Forward pass - unified interface."""
        if adj is None:
            adj = self.masked_adj if self.masked_adj is not None else self.norm_adj
            
        h = self.item_id_embedding.weight
        for i in range(self.num_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        """Compute the BPR loss."""
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        """Calculate loss - unified interface."""
        users, pos_items = interaction[0], interaction[1]
        neg_items = interaction[2] if len(interaction) > 2 else None

        # Generate negative samples if not provided
        if neg_items is None:
            # Generate random negative samples
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
            
        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

    def full_sort_predict(self, interaction):
        """Full-sort prediction - unified interface."""
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
