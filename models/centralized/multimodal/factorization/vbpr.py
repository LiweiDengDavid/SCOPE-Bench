# coding: utf-8
 
r"""
VBPR
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback
Ruining He, Julian McAuley. AAAI'16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import BPRLoss, EmbLoss, RecommenderBase, xavier_normal_initialization


class VBPR(RecommenderBase):
    """VBPR baseline with explicit ID and multimodal item components."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)

        # Manual embedding-L2 coefficient, decoupled from optimizer weight_decay.
        self.reg_weight = float(config["reg_weight"])

        self.user_id_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.user_modal_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embed_size)

        # Cache concatenated item features once (static tensors, never change)
        self._item_features = self._build_item_features()
        feature_dim = self._item_features.shape[1] if self._item_features is not None else 0
        self.item_linear = (
            nn.Linear(feature_dim, self.embed_size) if feature_dim > 0 else None
        )

        self.loss_fn = BPRLoss()
        self.reg_loss_fn = EmbLoss()
        self.apply(xavier_normal_initialization)

    def _align_features(self, features):
        """Align feature rows to n_items by padding or trimming."""
        if features is None:
            return None
        features = features.float()
        n = features.shape[0]
        if n < self.n_items:
            padding = torch.zeros(
                self.n_items - n,
                features.shape[1],
                dtype=features.dtype,
                device=features.device,
            )
            return torch.cat([features, padding], dim=0)
        if n > self.n_items:
            return features[:self.n_items]
        return features

    def _build_item_features(self):
        """Concatenate and cache multimodal features once at init."""
        parts = []
        if self.t_feat is not None:
            parts.append(self._align_features(self.t_feat))
        if self.v_feat is not None:
            parts.append(self._align_features(self.v_feat))
        if parts:
            return torch.cat(parts, dim=-1)
        return None

    def _get_item_embeddings(self, dropout=0.0):
        item_id_e = F.dropout(
            self.item_id_embedding.weight, dropout, training=self.training
        )

        if self._item_features is None or self.item_linear is None:
            item_modal_e = torch.zeros(
                self.n_items,
                self.embed_size,
                device=self.device,
                dtype=item_id_e.dtype,
            )
        else:
            item_modal_e = self.item_linear(self._item_features.to(item_id_e.device))
            item_modal_e = F.dropout(
                item_modal_e, dropout, training=self.training
            )

        return torch.cat([item_id_e, item_modal_e], dim=-1)

    def forward(self, dropout=0.0):
        user_id_e = F.dropout(
            self.user_id_embedding.weight, dropout, training=self.training
        )
        user_modal_e = F.dropout(
            self.user_modal_embedding.weight, dropout, training=self.training
        )
        user_e = torch.cat([user_id_e, user_modal_e], dim=-1)
        item_e = self._get_item_embeddings(dropout=dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user]
        pos_e = item_embeddings[pos_item]
        neg_e = item_embeddings[neg_item]

        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)

        mf_loss = self.loss_fn(pos_scores, neg_scores)
        reg_loss = self.reg_loss_fn(user_e, pos_e, neg_e)
        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction, *args, **kwargs):
        user = interaction[0]
        if user.dim() == 0:
            user = user.unsqueeze(0)

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user]
        scores = torch.matmul(user_e, item_embeddings.transpose(0, 1))
        return scores
