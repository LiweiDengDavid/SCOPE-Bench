# coding: utf-8
r"""
BeFA
################################################
Reference:
    Qile Fan, Penghang Yu, Zhiyi Tan, Bing-Kun Bao, Guanming Lu (2025).
    "BeFA: A General Behavior-driven Feature Adapter for Multimedia
    Recommendation." AAAI 2025.
    https://github.com/fqldom/BeFA  (LATTICE_BeFA.py)

BeFA is a general behavior-driven feature adapter. Pre-trained encoders extract
content features (v_feat/t_feat) that suffer information drift/omission; BeFA
reconstructs those content features (a per-modality MLP trained end-to-end under
the behavioral BPR signal) so that they better reflect user preferences. The
adapter is plug-and-play and runs BEFORE the backbone consumes the modality
features.

This baseline reproduces BeFA as a standalone model wrapping a configurable
backbone. The official repo demonstrates BeFA on a LATTICE backbone (a user-item
LightGCN view fused with per-modality item-item KNN-graph views, trained with
BPR + L2), so ``backbone="lattice"`` is the default and only supported value.

Design note (adapter gate): the official code always replaces the projected
modality feature with ``BeFA(projected)``. We express the reconstruction as a
residual gated by ``adapter_weight``:

    effective_feature = projected + adapter_weight * (BeFA(projected) - projected)

At ``adapter_weight=1.0`` (default) this is exactly the reference behaviour
(feature fully reconstructed); at ``adapter_weight=0.0`` the adapter is disabled
and the backbone sees the raw projected features, so the adapter's contribution
is controllable and testable.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase
from core.utils import build_knn_normalized_graph, build_sim

SUPPORTED_BACKBONES = ("lattice",)


class BeFA(RecommenderBase):
    def __init__(self, config, dataloader):
        super(BeFA, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        self.backbone = config["backbone"]
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"BeFA: unsupported backbone {self.backbone!r}; "
                f"supported: {SUPPORTED_BACKBONES}"
            )

        self.sparse = True
        self.embedding_dim = config["embedding_size"]
        self.n_ui_layers = config["num_ui_layers"]
        self.n_layers = config["num_layers"]
        self.knn_k = config["knn_k"]
        self.layer_times = config["layer_times"]
        self.adapter_weight = float(config["adapter_weight"])
        # In-model embedding-L2 coefficient, decoupled from optimizer weight_decay.
        self.reg_weight = float(config["reg_weight"])

        # load dataset info
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # User-item normalized bipartite adjacency (+ user->item block R).
        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # Per-modality item-item KNN graphs built in-memory from the pretrained
        # content features (no on-disk caching, so this runs on any dataset).
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_normalized_graph(
                image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type="sym"
            )
            self.image_original_adj = image_adj.to(self.device)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_normalized_graph(
                text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type="sym"
            )
            self.text_original_adj = text_adj.to(self.device)

        # Raw-dim -> embedding projections.
        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # BeFA behavior-driven feature adapters (per modality). A 3-linear MLP
        # with hidden width embedding_dim * layer_times that reconstructs the
        # projected content feature; the final Sigmoid keeps it in [0, 1].
        if self.v_feat is not None:
            self.BeFA_v = self._build_adapter()
        if self.t_feat is not None:
            self.BeFA_t = self._build_adapter()

    def _build_adapter(self):
        hidden = self.embedding_dim * self.layer_times
        return nn.Sequential(
            nn.Linear(self.embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden, self.embedding_dim),
            nn.Sigmoid(),
        )

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            # Add a small epsilon value to avoid division by zero.
            rowsum = rowsum + 1e-7
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[: self.n_users, self.n_users :]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def _apply_adapter(self, projected, adapter):
        """Residual behavior-driven reconstruction gated by adapter_weight."""
        reconstructed = adapter(projected)
        return projected + self.adapter_weight * (reconstructed - projected)

    def forward(self, adj):
        # Project raw modality dims into the embedding space, then reconstruct
        # via the behavior-driven adapter BEFORE the backbone consumes them.
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = self._apply_adapter(image_feats, self.BeFA_v)
            image_item_embeds = torch.multiply(self.item_id_embedding.weight, image_feats)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = self._apply_adapter(text_feats, self.BeFA_t)
            text_item_embeds = torch.multiply(self.item_id_embedding.weight, text_feats)

        # User-Item View (LightGCN).
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View (per-modality KNN graph propagation).
        side_views = []
        if self.v_feat is not None:
            for _ in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
            image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
            side_views.append(torch.cat([image_user_embeds, image_item_embeds], dim=0))
        if self.t_feat is not None:
            for _ in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
            text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
            side_views.append(torch.cat([text_user_embeds, text_item_embeds], dim=0))

        side_embeds = torch.stack(side_views, dim=0).mean(dim=0)
        all_embeds = content_embeds + side_embeds

        return torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (
            1.0 / 2 * (users ** 2).sum()
            + 1.0 / 2 * (pos_items ** 2).sum()
            + 1.0 / 2 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        emb_loss = self.reg_weight * regularizer
        return mf_loss, emb_loss

    def calculate_loss(self, interaction):
        # Handle different interaction formats (mirror MGCN): sample negatives
        # internally when only (users, pos_items) are provided.
        if len(interaction) == 3:
            users = interaction[0]
            pos_items = interaction[1]
            neg_items = interaction[2]
        elif len(interaction) == 2:
            users = interaction[0]
            pos_items = interaction[1]
            neg_items = torch.randint(
                0, self.n_items, pos_items.shape, device=pos_items.device
            )
        else:
            raise ValueError(
                f"Unsupported interaction format with {len(interaction)} elements"
            )

        ua_embeddings, ia_embeddings = self.forward(self.norm_adj)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )
        return batch_mf_loss + batch_emb_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
