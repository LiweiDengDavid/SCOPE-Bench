# coding: utf-8
"""
MMFedNCF: Multimodal Federated Neural Collaborative Filtering
"""

import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization
from .components.modules import FusionLayer
from core.utils import modal_ablation, resolve_multimodal_ablation


class MMFedNCF(RecommenderBase):
    """Multimodal FedNCF with shared item fusion and personal user heads."""

    def __init__(self, config, dataloader):
        super(MMFedNCF, self).__init__(config, dataloader)
        self.config['server_learning_rate'] = self.config['learning_rate'] * self.n_items
        self.setup_multimodal_features(config)

        # embedding_size here is the pre-extracted multimodal feature dim (FusionLayer
        # in_dim for txt/vis) — source it from the declared modality dims, not the
        # ID-embedding knob, so feature projection remains tied to feature metadata.
        # latent_size: collaborative latent dim (ID embeddings, fusion output, affine input)
        self.embedding_size = config['features']['text_dim']
        self.latent_size = config['feature_embedding_size']

        self.latent_dim_mf = self.latent_size
        self.latent_dim_mlp = self.latent_size

        # item_commonality in latent_size: already in latent space, no id_affine needed
        self.item_commonality = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_size
        )

        # User ID embeddings in latent_size (64): combined with the 64-dim fusion
        # output. Item-side GMF/MLP tables are intentionally absent: BLFM replacement
        # fusion uses the fused item representation in place of an item embedding.
        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_mf
        )

        # project_id=False because item_commonality is already in latent_size.
        # visual_dim declares the visual feature contract for asymmetric encoders.
        self.fusion = FusionLayer(
            self.embedding_size,
            fusion_module=config["fusion_method"],
            latent_dim=self.latent_size,
            project_id=False,
            dropout=config["dropout_rate"],
            visual_dim=config["features"]["visual_dim"],
        )

        # MLP layer configuration
        layers = [
            2 * self.latent_dim_mlp,
            self.latent_dim_mlp,
            self.latent_dim_mlp // 2,
            self.latent_dim_mlp // 4,
        ]

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=layers[-1] + self.latent_dim_mf, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

        # Apply parameter initialization
        self.apply(xavier_normal_initialization)

    def get_shared_parameters(self):
        """Shared multimodal item-side parameters aggregated on the server."""
        shared = {
            "item_commonality.weight": self.item_commonality.weight,
        }
        for name, param in self.fusion.named_parameters():
            if 'router' not in name:
                shared[f"fusion.{name}"] = param
        # fc_layers (MLP interaction layers) are shared on the server.
        # Rationale: MLP weights transform [user_emb || item_emb] concatenations;
        # placing them on the server allows cross-client generalization of the MLP
        # transformation and reduces per-client storage. The FedVLR paper does not
        # specify placement for MLP layers; this follows the FedNCF (Perifanis 2022)
        # convention of sharing item-side and transformation weights.
        for idx, layer in enumerate(self.fc_layers):
            shared[f"fc_layers.{idx}.weight"] = layer.weight
            shared[f"fc_layers.{idx}.bias"] = layer.bias
        return shared

    def get_personal_parameters(self):
        """Personal user-side embeddings and routing preferences kept on each client."""
        personal = {
            "embedding_user_mlp.weight": self.embedding_user_mlp.weight,
            "embedding_user_mf.weight": self.embedding_user_mf.weight,
            "affine_output.weight": self.affine_output.weight,
            "affine_output.bias": self.affine_output.bias,
        }
        for name, param in self.fusion.named_parameters():
            if 'router' in name:
                personal[f"fusion.{name}"] = param
        return personal

    def get_server_grad_param_names(self):
        """D (item embeddings) + γ_j (fusion non-router) use delta aggregation.

        fc_layers remain weight-averaged (paper does not specify their placement
        in the BLFM framework; keeping them weight-averaged is a safe default).
        """
        names = [
            'item_commonality.weight',
        ]
        for name, _ in self.fusion.named_parameters():
            if 'router' not in name:
                names.append(f'fusion.{name}')
        return names

    def forward(self, user_indices, item_indices, txt_embed=None, vision_embed=None):
        """Forward pass - unified interface."""
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)

        # Handle multimodal features
        if txt_embed is None:
            txt_embed = self.t_feat if self.t_feat is not None else torch.zeros(
                self.n_items, self.embedding_size, device=self.device)
        if vision_embed is None:
            vision_embed = self.v_feat if self.v_feat is not None else torch.zeros(
                self.n_items, self.embedding_size, device=self.device)

        # Detach features to prevent them from entering the computation graph
        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()
        item_commonality = self.item_commonality(item_indices)

        # Perform multimodal ablation
        item_commonality, txt, vision = modal_ablation(
            item_commonality,
            txt,
            vision,
            **resolve_multimodal_ablation(self.config),
        )

        # Multimodal feature fusion
        out = self.fusion(item_commonality, txt, vision)

        # BLFM replacement fusion — fused representation replaces item embedding
        mlp_vector = torch.cat([user_embedding_mlp, out], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, out)

        # MLP path
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # Concatenate MLP output and GMF output
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)

        return logits

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
        txt_embed = self.t_feat
        vis_embed = self.v_feat

        # Check input type and handle appropriately
        if isinstance(interaction, list):
            user = interaction[0]
            if isinstance(user, torch.Tensor):
                user = user[0]  # extract single user ID
        else:
            user = interaction[0]

        if isinstance(user, torch.Tensor) and user.dim() == 0:
            user = user.unsqueeze(0)

        # Retrieve user embeddings
        user_mlp = self.embedding_user_mlp(user)
        user_mf = self.embedding_user_mf(user)

        # Process and fuse features
        item_feats, txt_feats, vis_feats = self._process_features(txt_embed, vis_embed)
        fused_item = self.fusion(item_feats, txt_feats, vis_feats)

        # Expand user vector to match the number of items
        user_mlp = user_mlp.expand(self.n_items, -1)
        user_mf = user_mf.expand(self.n_items, -1)

        # Replacement fusion (consistent with forward())
        mlp_vector = torch.cat([user_mlp, fused_item], dim=-1)
        mf_vector = torch.mul(user_mf, fused_item)

        # MLP path forward pass
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # Concatenate MLP and MF results
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        # Compute final scores
        logits = self.affine_output(vector)
        scores = self.logistic(logits)

        return scores.view(1, -1)

    def _process_features(self, txt_feat=None, vis_feat=None):
        """Process feature data and apply modality ablation operations."""
        # Retrieve item commonality embeddings (always uses the full weight matrix)
        item_commonality = self.item_commonality.weight

        # Ensure features exist; create zero tensors as fallback (embedding_size for txt/vis)
        if txt_feat is None:
            txt_feat = torch.zeros(self.n_items, self.embedding_size, device=self.device)

        if vis_feat is None:
            vis_feat = torch.zeros(self.n_items, self.embedding_size, device=self.device)

        # Apply modality ablation
        item_commonality, txt_feat, vis_feat = modal_ablation(
            item_commonality,
            txt_feat,
            vis_feat,
            **resolve_multimodal_ablation(self.config),
        )

        return item_commonality, txt_feat, vis_feat
