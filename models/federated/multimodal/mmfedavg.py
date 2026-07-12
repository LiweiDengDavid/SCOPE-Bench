# coding: utf-8
 
"""MMFedAvg: multimodal federated averaging with personalized fusion."""

import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization
from .components.modules import FusionLayer


class MMFedAvg(RecommenderBase):
    """Multimodal Federated Averaging recommendation model."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        # embed_size feeds FusionLayer and feature fallbacks as the pre-extracted
        # modality dimension, independent of the collaborative ID embedding size.
        self.embed_size = config["features"]["text_dim"]
        self.config['server_learning_rate'] = self.config['learning_rate'] * self.n_items
        self.setup_multimodal_features(config)

        # Item commonality embedding in latent_dim: already in latent space, no id_affine needed
        latent_dim = config['feature_embedding_size']
        self.item_commonality = nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=latent_dim
        )

        # project_id=False because item_commonality is already in latent_dim.
        # visual_dim declares the visual feature contract for asymmetric encoders.
        self.fusion = FusionLayer(
            self.embed_size,
            fusion_module=config['fusion_method'],
            latent_dim=latent_dim,
            project_id=False,
            dropout=config['dropout_rate'],
            visual_dim=config["features"]["visual_dim"],
        )

        # Output layer
        output_dim = latent_dim
        self.affine_output = nn.Linear(output_dim, 1)

        # User embedding (personal): personalizes predictions via element-wise product
        # with the fused item representation before the final affine head.
        self.user_embedding = nn.Embedding(self.n_users, output_dim)
        self.logistic = nn.Sigmoid()

        # Parameter initialization
        self.apply(xavier_normal_initialization)
    
    def forward(self, user_indices, item_indices, txt_embed=None, vision_embed=None):
        """Forward pass.

        NOTE: This model applies sigmoid inside forward() and uses binary_cross_entropy
        in calculate_loss(). This differs from MMFedNCF/MMFedRAP/MMFCF which return raw
        logits and use BCEWithLogitsLoss. Do NOT change this without also updating
        calculate_loss() — the two must stay in sync.
        """
        # Retrieve ID embedding
        item_embed = self.item_commonality(item_indices)

        # Fall back to automatically managed features if none are provided
        if txt_embed is None:
            txt_embed = self.t_feat
        if vision_embed is None:
            vision_embed = self.v_feat

        # .detach() on pretrained features to prevent gradient flow
        txt_feat = txt_embed[item_indices].detach() if txt_embed is not None else torch.zeros(
            item_embed.shape[0], self.embed_size, device=self.device)
        vis_feat = vision_embed[item_indices].detach() if vision_embed is not None else torch.zeros(
            item_embed.shape[0], self.embed_size, device=self.device)

        # Fuse features
        fused = self.fusion(item_embed, txt_feat, vis_feat)

        # Personalize: element-wise product with user embedding
        user_emb = self.user_embedding(user_indices)
        out = fused * user_emb

        # Produce prediction
        pred = self.affine_output(out)
        rating = self.logistic(pred)

        return rating.squeeze(-1)
    
    def calculate_loss(self, batch):
        """Calculate loss."""
        users, pos_items, neg_items = batch[0], batch[1], batch[2]

        # Construct positive/negative sample batch
        items = torch.cat([pos_items, neg_items])
        labels = torch.cat([
            torch.ones(pos_items.size(0), device=self.device),
            torch.zeros(neg_items.size(0), device=self.device)
        ])
        # Duplicate users to match the concatenated items tensor
        user_indices = torch.cat([users, users])

        # Forward pass (uses automatically managed features)
        predictions = self.forward(user_indices, items)

        # BCE matches the probability-valued output returned by forward().
        loss = nn.functional.binary_cross_entropy(predictions, labels)

        return loss
    
    def full_sort_predict(self, interaction, **kwargs):
        """Full-sort prediction."""
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
        scores = self.forward(users, items, self.t_feat, self.v_feat)

        return scores.view(1, -1)
    
    def get_shared_parameters(self):
        """Get shared parameters to be aggregated in federated learning."""
        shared_params = {
            'item_commonality.weight': self.item_commonality.weight
        }
        for name, param in self.fusion.named_parameters():
            if 'router' not in name:
                shared_params[f'fusion.{name}'] = param
        return shared_params

    def get_personal_parameters(self):
        """Get personalized parameters that are excluded from federated aggregation."""
        personal = {
            'user_embedding.weight': self.user_embedding.weight,
            'affine_output.weight': self.affine_output.weight,
            'affine_output.bias': self.affine_output.bias
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
