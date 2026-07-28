# coding: utf-8
r"""
PGL
################################################
Reference:
    Yu et al. "Mind Individual Information! Principal Graph Learning for
    Multimedia Recommendation." AAAI 2025 (Oral).
    Official repo: https://github.com/demonph10/PGL

Faithful port of the official ``src/models/pgl.py``. PGL uses a FREEDOM/
LATTICE-style backbone (NOT MGCN): per-modality user/item embeddings concatenated
along the feature dim, a single frozen item-item modality graph ``mm_adj``, and a
LightGCN-style U-I propagation over a *principal subgraph* of the interaction
graph. The principal subgraph is selected by ``mode``:

  * ``'global'``: a low-rank SVD "principal subgraph" of the normalized U-I
    adjacency, built ONCE (repo ``global_subgraph_extraction``);
  * ``'local'``: a degree-sensitive edge-pruned subgraph refreshed each epoch,
    keeping ~30% of edges (repo ``pre_epoch_processing``). This is the shipped
    best config (``mode: ['local']``).

Training loss = ``mf_loss + reg_weight * cl_loss``, where ``cl_loss`` is an
InfoNCE over dropout self-views at temperature 0.2 and ``reg_weight`` is that CL
coefficient (repo ``calculate_loss``). There is NO embedding-L2 term.
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


class PGL(RecommenderBase):
    def __init__(self, config, dataloader):
        super(PGL, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.mode = config['mode']
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['num_layers']          # item-item (mm) propagation layers
        self.n_ui_layers = config['num_ui_layers']    # user-item propagation layers
        self.reg_weight = float(config['reg_weight'])  # CL loss coefficient (repo reg_weight)
        self.mm_image_weight = float(config['mm_image_weight'])
        self.prune_ratio = float(config['prune_ratio'])
        # InfoNCE temperature and the global principal-subgraph top/bottom
        # singular-value fraction (kept in YAML per the no-literals rule).
        self.temperature = float(config['temperature'])
        self.principal_ratio = float(config['principal_ratio'])

        self.n_nodes = self.n_users + self.n_items
        self.sub_graph, self.mm_adj = None, None

        # load dataset info
        self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)

        # Per-modality user embeddings, concatenated along the feature dim.
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset']) if 'data_path' in config else None
        mm_adj_file = (
            os.path.join(dataset_path, 'mm_adj_pgl_{}_{}.pt'.format(self.knn_k, int(10 * self.mm_image_weight)))
            if dataset_path else None
        )

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Frozen item-item modality graph (KNN over raw modality features).
        if mm_adj_file is not None and os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location="cpu", weights_only=False).to(self.device)
        else:
            image_adj, text_adj = None, None
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            if mm_adj_file is not None:
                torch.save(self.mm_adj.cpu(), mm_adj_file)
            self.mm_adj = self.mm_adj.to(self.device)

        # Principal subgraph selection (mutually-exclusive modes).
        if self.mode == 'global':
            # Built ONCE from the normalized U-I adjacency (repo builds in get_norm_adj_mat).
            self.global_u, self.global_v = self.build_global_subgraph()
            self.sub_graph = None  # global mode propagates via global_propagate, not a sparse adj
        else:
            # Local mode: refreshed each epoch by pre_epoch_processing.
            self.sub_graph = self.build_local_subgraph()

        self.dropoutf = nn.Dropout(self.dropout_rate)

    # ------------------------------------------------------------------ graphs

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def get_knn_adj_mat(self, mm_embeddings):
        """KNN item-item graph over modality features (repo PGL.get_knn_adj_mat)."""
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True) + 1e-12)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_k = min(self.knn_k, sim.size(1))
        _, knn_ind = torch.topk(sim, knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(mm_embeddings.device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]).float(), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        """Symmetric normalized U-I bipartite adjacency (repo PGL.get_norm_adj_mat).

        Assembles the same A = D^-0.5 (bipartite adjacency) D^-0.5 as the repo,
        building the sparse adjacency via COO blocks (the repo's private
        ``dok_matrix._update`` was removed in newer scipy).
        """
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        row = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        col = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones(len(row), dtype=np.float32)
        A = sp.coo_matrix((data, (row, col)), shape=(self.n_nodes, self.n_nodes), dtype=np.float32)
        diag = np.array((A > 0).sum(axis=1)).flatten() + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = sp.coo_matrix(D * A * D)
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def _normalize_adj_m(self, indices, adj_size):
        """Symmetric degree normalization of a bipartite (user x item) edge set."""
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]).float(), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        return rows_inv_sqrt * cols_inv_sqrt

    def get_edge_info(self):
        """Degree-normalized U-I edge list (repo PGL.get_edge_info)."""
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def build_global_subgraph(self):
        """GLOBAL-aware principal subgraph via top/bottom singular-value product.

        Ported from repo PGL.global_subgraph_extraction, kept as low-rank factors
        so the n_nodes x n_nodes product (infeasible at ~26k x 26k) is never
        materialized. torch.svd_lowrank consumes the SPARSE adjacency directly.
        Propagation applies P @ X = U_scaled @ (V^T @ X). NOTE: the "bottom"
        components here are the tail of the top-q approximation, not the true
        spectral tail (memory-safe stand-in for sparsesvd's full-spectrum bottom).
        """
        q = max(min(self.embedding_dim, self.n_nodes - 1), 1)
        u, s, v = torch.svd_lowrank(self.norm_adj, q=q)
        num_top_bottom = max(int(self.principal_ratio * q), 1)
        product_s = s[:num_top_bottom] * s[-num_top_bottom:]  # element-wise top x bottom
        u_scaled = u[:, :num_top_bottom] * product_s.unsqueeze(0)
        v_top = v[:, :num_top_bottom]
        return u_scaled.contiguous().to(self.device), v_top.contiguous().to(self.device)

    def global_propagate(self, x):
        """P @ x = U_scaled @ (V_top^T @ x); O(n_nodes * r * d), never O(n_nodes^2)."""
        return self.global_u @ (self.global_v.transpose(0, 1) @ x)

    def build_local_subgraph(self):
        """LOCAL-aware principal subgraph via degree-sensitive edge pruning.

        Ported from repo PGL.pre_epoch_processing: keep a fraction of edges
        (repo keeps ~30% via degree_len = 0.3*|E|) sampled by normalized degree
        weight WITH replacement, renormalize, symmetrize into a full-graph adj.
        The keep fraction is (1 - prune_ratio); with the shipped prune_ratio=0.7
        this reproduces the repo's 0.3 keep.
        """
        num_edges = self.edge_values.size(0)
        keep_len = int(num_edges * (1.0 - self.prune_ratio))
        keep_len = min(max(keep_len, 1), num_edges)
        degree_idx = torch.multinomial(self.edge_values, keep_len)  # with replacement (repo)
        keep_indices = self.edge_indices[:, degree_idx].clone()
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), dim=1)
        return torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj.shape).coalesce().to(self.device)

    def pre_epoch_processing(self):
        # Local mode refreshes its degree-pruned subgraph each epoch (repo).
        if self.mode != 'global':
            self.sub_graph = self.build_local_subgraph()

    # ------------------------------------------------------------------ model

    def forward(self, adj):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        image_feats, text_feats = F.normalize(image_feats), F.normalize(text_feats)
        user_embeds = torch.cat([self.user_image.weight, self.user_text.weight], dim=1)
        item_embeds = torch.cat([image_feats, text_feats], dim=1)

        # Item-item propagation over the frozen modality graph.
        h = item_embeds
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # User-Item LightGCN propagation over the selected principal subgraph.
        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]
        cur = ego_embeddings
        for i in range(self.n_ui_layers):
            if self.mode == 'global' and adj is None:
                cur = self.global_propagate(cur)
            else:
                cur = torch.sparse.mm(adj, cur)
            all_embeddings += [cur]
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        return -torch.mean(maxi)

    def InfoNCE(self, view1, view2, temperature):
        """PGL contrastive term (repo PGL.InfoNCE)."""
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def _train_adj(self):
        """Adjacency the U-I view propagates over during training (per mode)."""
        return None if self.mode == 'global' else self.sub_graph

    def calculate_loss(self, interaction):
        if len(interaction) == 3:
            users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        elif len(interaction) == 2:
            users, pos_items = interaction[0], interaction[1]
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)
        else:
            raise ValueError(f"Unsupported interaction format with {len(interaction)} elements")

        ua_embeddings, ia_embeddings = self.forward(self._train_adj())

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        cl_loss = (self.InfoNCE(self.dropoutf(u_g_embeddings), self.dropoutf(u_g_embeddings), self.temperature)
                   + self.InfoNCE(self.dropoutf(pos_i_g_embeddings), self.dropoutf(pos_i_g_embeddings), self.temperature)) / 2
        return batch_mf_loss + self.reg_weight * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        # Prediction propagates over the full normalized U-I adjacency (repo).
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
