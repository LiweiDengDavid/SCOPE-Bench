# coding: utf-8
"""
MMPFedRec: multimodal PFedRec with FedVLR-style item fusion.

The model keeps PFedRec's dual-personalized scoring contract:

    score(u, i) = <w_u, item_i> + b_u

The item representation is shared and aggregated on the server. The predictor
head ``w_u, b_u`` is row-indexed by user and remains client-specific.
Multimodal features enter only through the shared item representation, using
the same FusionLayer contract as the other FedVLR-family hosts.
"""

import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization
from core.utils import modal_ablation, resolve_multimodal_ablation
from .components.modules import FusionLayer


class MMPFedRec(RecommenderBase):
    """PFedRec backbone with shared multimodal item fusion."""

    supports_multi_negatives = True

    def __init__(self, config, dataloader):
        super(MMPFedRec, self).__init__(config, dataloader)
        self.config["server_learning_rate"] = (
            self.config["learning_rate"] * self.n_items
        )
        self.setup_multimodal_features(config)

        self.feature_dim = config["features"]["text_dim"]
        self.latent_size = config["feature_embedding_size"]

        self.item_commonality = nn.Embedding(self.n_items, self.latent_size)
        self.user_predictor_weight = nn.Embedding(self.n_users, self.latent_size)
        self.user_predictor_bias = nn.Embedding(self.n_users, 1)

        self.fusion = FusionLayer(
            self.feature_dim,
            fusion_module=config["fusion_method"],
            latent_dim=self.latent_size,
            project_id=False,
            dropout=config["dropout_rate"],
            visual_dim=config["features"]["visual_dim"],
        )

        self.apply(xavier_normal_initialization)

    def get_shared_parameters(self):
        """Shared item-side parameters aggregated on the server."""
        shared = {
            "item_commonality.weight": self.item_commonality.weight,
        }
        for name, param in self.fusion.named_parameters():
            if "router" not in name:
                shared[f"fusion.{name}"] = param
        return shared

    def get_personal_parameters(self):
        """Client-specific PFedRec predictor head and FedVLR router."""
        personal = {
            "user_predictor_weight.weight": self.user_predictor_weight.weight,
            "user_predictor_bias.weight": self.user_predictor_bias.weight,
        }
        for name, param in self.fusion.named_parameters():
            if "router" in name:
                personal[f"fusion.{name}"] = param
        return personal

    def get_row_personal_parameter_names(self):
        """Personal parameters whose first dimension indexes users."""
        return {
            "user_predictor_weight.weight",
            "user_predictor_bias.weight",
        }

    def get_server_grad_param_names(self):
        """Parameters updated through the federated delta-aggregation path."""
        names = ["item_commonality.weight"]
        for name, _ in self.fusion.named_parameters():
            if "router" not in name:
                names.append(f"fusion.{name}")
        return names

    def _item_features(self, item_indices, txt_embed=None, vision_embed=None):
        item_embed = self.item_commonality(item_indices)
        if txt_embed is None:
            txt_embed = self.t_feat if self.t_feat is not None else torch.zeros(
                self.n_items,
                self.feature_dim,
                device=self.device,
            )
        if vision_embed is None:
            vision_embed = self.v_feat if self.v_feat is not None else torch.zeros(
                self.n_items,
                self.feature_dim,
                device=self.device,
            )

        text_feature = txt_embed[item_indices].detach()
        visual_feature = vision_embed[item_indices].detach()
        item_embed, text_feature, visual_feature = modal_ablation(
            item_embed,
            text_feature,
            visual_feature,
            **resolve_multimodal_ablation(self.config),
        )
        return self.fusion(item_embed, text_feature, visual_feature)

    def forward(self, user_indices, item_indices, txt_embed=None, vision_embed=None):
        item_embed = self._item_features(item_indices, txt_embed, vision_embed)
        weight = self.user_predictor_weight(user_indices)
        bias = self.user_predictor_bias(user_indices)
        logits = (item_embed * weight).sum(dim=-1) + bias.squeeze(-1)
        return logits.view(-1, 1)

    def calculate_loss(self, interaction):
        users, pos_items = interaction[0], interaction[1]
        negative_rows = list(interaction[2:])
        if not negative_rows:
            raise ValueError("MMPFedRec.calculate_loss requires at least one negative row.")

        items = torch.cat([pos_items] + negative_rows)
        labels = torch.zeros(items.size(0), dtype=torch.float32, device=self.device)
        labels[:pos_items.size(0)] = 1
        repeated_users = users.repeat(len(negative_rows) + 1)

        logits = self.forward(repeated_users, items, self.t_feat, self.v_feat)
        return nn.BCEWithLogitsLoss()(logits.view(-1), labels)

    def full_sort_predict(self, interaction, **kwargs):
        users = interaction[0]
        if isinstance(users, torch.Tensor) and users.dim() == 0:
            users = users.unsqueeze(0)
        items = torch.arange(self.n_items, device=self.device)
        item_table = self._item_features(items, self.t_feat, self.v_feat)
        weight = self.user_predictor_weight(users)
        bias = self.user_predictor_bias(users)
        scores = torch.matmul(weight, item_table.T) + bias
        return torch.sigmoid(scores)
