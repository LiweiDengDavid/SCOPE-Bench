# coding: utf-8
"""BM3: Bootstrap Latent Representations for Multi-modal Recommendation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from core.base import EmbLoss, RecommenderBase
from core.utils import build_norm_adj_matrix


class BM3(RecommenderBase):
    """BM3 recommendation model."""

    def __init__(self, config, dataloader):
        super(BM3, self).__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # Base hyperparameters (aligned with MMRec official BM3)
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["embedding_size"]
        self.n_layers = config["num_layers"]
        # Manual embedding-L2 coefficient, decoupled from optimizer weight_decay.
        self.reg_weight = float(config["reg_weight"])
        self.ssl_weight = config["ssl_weight"]
        self.dropout = config["dropout_rate"]
        self.n_nodes = self.n_users + self.n_items

        # Build normalised adjacency matrix
        self.norm_adj = build_norm_adj_matrix(
            dataloader.inter_matrix(form="coo").astype(np.float32),
            self.n_users, self.n_items, self.device)

        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Unified predictor matching MMRec official BM3 structure
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()
        nn.init.xavier_normal_(self.predictor.weight)

        # Multimodal feature layers (auto-configured)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=False
            )
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=False
            )
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)
    def forward(self, *args, **kwargs):
        """Graph-convolution forward pass."""
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        """BM3 bootstrap contrastive loss."""
        # online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (
            (loss_ui + loss_iu).mean()
            + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori)
            + self.ssl_weight * (loss_t + loss_v + loss_tv + loss_vt)
        )

    def full_sort_predict(self, interaction):
        """Full-ranking prediction via inner product."""
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui
