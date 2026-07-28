# coding: utf-8
"""
MMFCF: Multimodal Federated Collaborative Filtering

This is the multimodal extension of FCF (Ammad-Ud-Din et al. 2019) used as a
baseline in the FedVLR paper. It extends plain FCF (equal-weighted FedAvg of
matrix factorization) with multimodal feature fusion:
  - FusionLayer fuses item_commonality, text, and visual features
  - A personal user_embedding personalizes predictions per client
  - A MoE-capable router kept personal (not aggregated)

The vanilla FCF baseline (no multimodal, no personal embedding) is implemented
in models/fcf.py. MMFCF is not directly comparable to the FCF column in the
FedVLR results table.
"""

import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization
from .components.modules import FusionLayer
from core.utils import modal_ablation, resolve_multimodal_ablation


class MMFCF(RecommenderBase):
    """Multimodal FCF with shared item fusion and a personal user embedding."""

    def __init__(self, config, dataloader):
        super(MMFCF, self).__init__(config, dataloader)
        self.config['server_learning_rate'] = self.config['learning_rate'] * self.n_items
        self.setup_multimodal_features(config)

        # embed_size: pre-extracted multimodal feature dim (FusionLayer in_dim for txt/vis)
        # latent_size: collaborative latent dim (ID embeddings, fusion output, affine input)
        self.embed_size = config["features"]["text_dim"]
        self.latent_size = config["feature_embedding_size"]

        # Item commonality embedding in latent_size: already in latent space, no id_affine needed
        self.item_commonality = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_size
        )

        # User embedding (personal): personalizes predictions via element-wise product
        # with the fused item representation before the final affine head.
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_size
        )

        # project_id=False because item_commonality is already in latent_size.
        # visual_dim declares the visual feature contract for asymmetric encoders.
        self.fusion = FusionLayer(
            self.embed_size,
            fusion_module=config["fusion_method"],
            latent_dim=self.latent_size,
            project_id=False,
            dropout=config["dropout_rate"],
            visual_dim=config["features"]["visual_dim"],
        )

        # Output layer
        self.affine_output = torch.nn.Linear(
            in_features=self.latent_size, out_features=1
        )

        # Apply parameter initialization
        self.apply(xavier_normal_initialization)

    def set_item_commonality(self, item_commonality):
        """Set the item commonality feature embedding layer."""
        self.item_commonality.load_state_dict(item_commonality.state_dict())

    def get_shared_parameters(self):
        """Shared global parameters aggregated across clients."""
        shared = {
            'item_commonality.weight': self.item_commonality.weight,
        }
        for name, param in self.fusion.named_parameters():
            if 'router' not in name:
                shared[f'fusion.{name}'] = param
        return shared

    def get_personal_parameters(self):
        """Client-specific parameters kept off the server average."""
        personal = {
            'user_embedding.weight': self.user_embedding.weight,
            'affine_output.weight': self.affine_output.weight,
            'affine_output.bias': self.affine_output.bias,
        }
        for name, param in self.fusion.named_parameters():
            if 'router' in name:
                personal[f'fusion.{name}'] = param
        return personal

    def get_server_grad_param_names(self):
        """D (item_commonality) + γ_j (fusion non-router) use delta aggregation."""
        names = ['item_commonality.weight']
        for name, _ in self.fusion.named_parameters():
            if 'router' not in name:
                names.append(f'fusion.{name}')
        return names

    def forward(self, user_indices, item_indices, txt_embed=None, vision_embed=None):
        """Forward pass - unified interface."""
        item_commonality = self.item_commonality(item_indices)

        # Handle multimodal features
        if txt_embed is None:
            txt_embed = self.t_feat if self.t_feat is not None else torch.zeros(
                self.n_items, self.embed_size, device=self.device)
        if vision_embed is None:
            vision_embed = self.v_feat if self.v_feat is not None else torch.zeros(
                self.n_items, self.embed_size, device=self.device)

        # Detach features to prevent them from entering the computation graph
        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()

        # Perform multimodal ablation
        item_commonality, txt, vision = modal_ablation(
            item_commonality,
            txt,
            vision,
            **resolve_multimodal_ablation(self.config),
        )

        # Multimodal feature fusion → latent_size dim
        fused = self.fusion(item_commonality, txt, vision)

        # Personalize: element-wise product with user embedding
        user_embed = self.user_embedding(user_indices)
        out = fused * user_embed

        # Produce prediction
        pred = self.affine_output(out)

        return pred  # return raw logits directly

    def calculate_loss(self, interaction):
        """Calculate loss - unified interface."""
        user, poss, negs = interaction[0], interaction[1], interaction[2]
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32, device=self.device)
        ratings[:poss.size(0)] = 1
        users = torch.cat([user, user])

        pred = self.forward(users, items, self.t_feat, self.v_feat)
        return nn.BCEWithLogitsLoss()(pred.view(-1), ratings)

    def full_sort_predict(self, interaction, **kwargs):
        """Full-sort prediction - unified interface."""
        if isinstance(interaction, list):
            user = interaction[0]
            if isinstance(user, torch.Tensor):
                user = user[0]
        else:
            user = interaction[0]

        if isinstance(user, torch.Tensor) and user.dim() == 0:
            user = user.unsqueeze(0)

        items = torch.arange(self.n_items, device=self.device)
        user_id = user.item() if isinstance(user, torch.Tensor) else int(user)
        users = torch.full((self.n_items,), user_id, dtype=torch.long, device=self.device)
        logits = self.forward(users, items, self.t_feat, self.v_feat)

        # Apply sigmoid at inference time to obtain probabilities
        return torch.sigmoid(logits).view(1, -1)
