# coding: utf-8
"""
MMFedRAP: Multimodal Federated Recommendation with Personalization

Uses the same FusionLayer pattern as MMFedNCF/MMFCF (router inside as personal),
with two MMFedRAP-specific differences:
- item_personality is added as a client-owned residual after shared fusion
- independence loss separates personality from commonality

Both item_commonality and item_personality are in latent_size (not embedding_size),
so FusionLayer uses project_id=False to skip the ID projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase, xavier_normal_initialization
from .components.modules import FusionLayer
from core.utils import modal_ablation, resolve_multimodal_ablation


class MMFedRAP(RecommenderBase):
    """MMFedRAP: multimodal FedRAP with FusionLayer (router as personal)."""

    def __init__(self, config, dataloader):
        super(MMFedRAP, self).__init__(config, dataloader)
        self.config['server_learning_rate'] = self.config['learning_rate'] * self.n_items
        self.setup_multimodal_features(config)

        # embedding_size here is the pre-extracted modality dimension consumed by
        # FusionLayer, independent of the collaborative ID embedding size.
        self.embedding_size = config['features']['text_dim']
        self.latent_size = config['feature_embedding_size']
        self.alpha = float(config['alpha'])
        self.beta = float(config['beta'])
        self.loss_warmup_rounds = float(config['loss_warmup_rounds'])
        self.head_stability_weight = float(config["head_stability_weight"])
        self.head_stability_head_quantile = float(config["head_stability_head_quantile"])

        # Both embeddings in latent_size so independence loss can compare them
        self.item_commonality = nn.Embedding(self.n_items, self.latent_size)
        self.item_personality = nn.Embedding(self.n_items, self.latent_size)

        # FusionLayer keeps router parameters personal and non-router projections shared.
        # project_id=False because commonality is already in latent_size.
        # visual_dim declares the visual feature contract for asymmetric encoders.
        self.fusion = FusionLayer(
            self.embedding_size,
            fusion_module=config["fusion_method"],
            latent_dim=self.latent_size,
            project_id=False,
            dropout=config["dropout_rate"],
            visual_dim=config["features"]["visual_dim"],
        )

        # Output layer
        self.affine_output = nn.Linear(self.latent_size, 1)

        # Dynamic loss weighting (epoch propagated by FederatedTrainer)
        self.current_epoch = 0
        self.register_buffer(
            "head_stability_item_mask",
            self._build_head_stability_item_mask(dataloader),
        )

        self.apply(xavier_normal_initialization)

    def _build_head_stability_item_mask(self, dataloader):
        """Mark head items from training counts for optional stability loss."""
        mask = torch.zeros(self.n_items, dtype=torch.bool)
        if self.head_stability_weight <= 0:
            return mask

        df = dataloader.dataset.df
        iid_field = dataloader.dataset.iid_field
        counts = torch.zeros(self.n_items, dtype=torch.float32)
        item_counts = df[iid_field].value_counts()
        for item_id, count in item_counts.items():
            if 0 <= item_id < self.n_items:
                counts[int(item_id)] = float(count)
        observed = counts[counts > 0]
        if observed.numel() == 0:
            raise ValueError("head_stability_weight requires non-empty item counts.")
        head_cut = torch.quantile(observed, self.head_stability_head_quantile)
        return counts >= head_cut

    def get_shared_parameters(self):
        """Shared global parameters aggregated across clients."""
        shared = {
            "item_commonality.weight": self.item_commonality.weight,
        }
        for name, param in self.fusion.named_parameters():
            if 'router' not in name:
                shared[f"fusion.{name}"] = param
        return shared

    def get_personal_parameters(self):
        """Personal client-specific parameters."""
        personal = {
            "item_personality.weight": self.item_personality.weight,
            "affine_output.weight": self.affine_output.weight,
            "affine_output.bias": self.affine_output.bias,
        }
        for name, param in self.fusion.named_parameters():
            if 'router' in name:
                personal[f"fusion.{name}"] = param
        return personal

    def get_server_grad_param_names(self):
        """D (item_commonality) + γ_j (fusion non-router) use delta aggregation."""
        names = ['item_commonality.weight']
        for name, _ in self.fusion.named_parameters():
            if 'router' not in name:
                names.append(f'fusion.{name}')
        return names

    def forward(self, item_indices, txt_embed=None, vision_embed=None):
        """Forward pass - unified interface."""
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        # Handle multimodal features
        if txt_embed is None:
            txt_embed = self.t_feat if self.t_feat is not None else torch.zeros(
                self.n_items, self.embedding_size, device=self.device)
        if vision_embed is None:
            vision_embed = self.v_feat if self.v_feat is not None else torch.zeros(
                self.n_items, self.embedding_size, device=self.device)

        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()

        # Multimodal ablation (on commonality only)
        processed_id, processed_txt, processed_vision = modal_ablation(
            item_commonality, txt, vision,
            **resolve_multimodal_ablation(self.config),
        )

        # Fuse commonality + multimodal features (router inside FusionLayer)
        fused = self.fusion(processed_id, processed_txt, processed_vision)

        # Apply client-owned personality as a residual after shared multimodal fusion.
        out = fused + item_personality

        logits = self.affine_output(out)

        return logits, item_personality, item_commonality

    def forward_no_modal(self, item_indices, txt_embed=None, vision_embed=None):
        """Forward pass using the ID-only preference signal."""
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        if txt_embed is None:
            txt_embed = self.t_feat if self.t_feat is not None else torch.zeros(
                self.n_items, self.embedding_size, device=self.device)
        if vision_embed is None:
            vision_embed = self.v_feat if self.v_feat is not None else torch.zeros(
                self.n_items, self.embedding_size, device=self.device)

        txt = txt_embed[item_indices].detach()
        vision = vision_embed[item_indices].detach()

        processed_id = item_commonality
        processed_txt = torch.zeros_like(txt)
        processed_vision = torch.zeros_like(vision)
        fused = self.fusion(processed_id, processed_txt, processed_vision)
        out = fused + item_personality
        logits = self.affine_output(out)
        return logits

    def calculate_loss(self, interaction):
        """Calculate loss - unified interface."""
        _, poss, negs = interaction[0], interaction[1], interaction[2]
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32, device=self.device)
        ratings[:poss.size(0)] = 1

        pred, item_personality, item_commonality = self.forward(items, self.t_feat, self.v_feat)

        # BCEWithLogitsLoss matches the raw logits returned by forward().
        bce_loss = F.binary_cross_entropy_with_logits(pred.view(-1), ratings)

        # Independence loss: personality and commonality should differ (mean-reduced)
        independency_loss = F.mse_loss(item_personality, item_commonality)

        # Regularization loss
        reg_loss = F.l1_loss(item_commonality, torch.zeros_like(item_commonality))

        # Dynamic weighting
        scale = math.tanh(self.current_epoch / self.loss_warmup_rounds)
        total_loss = bce_loss - self.alpha * scale * independency_loss + self.beta * scale * reg_loss

        if self.head_stability_weight > 0:
            head_mask = self.head_stability_item_mask[items]
            if torch.any(head_mask):
                no_modal_pred = self.forward_no_modal(items[head_mask], self.t_feat, self.v_feat)
                head_stability_loss = F.mse_loss(
                    pred.view(-1)[head_mask],
                    no_modal_pred.detach().view(-1),
                )
                total_loss = total_loss + self.head_stability_weight * scale * head_stability_loss

        return total_loss

    def full_sort_predict(self, interaction, **kwargs):
        """Full-sort prediction - unified interface."""
        items = torch.arange(self.n_items, device=self.device)
        logits, _, _ = self.forward(items, self.t_feat, self.v_feat)
        return torch.sigmoid(logits).view(1, -1)


from core.federated import FederatedTrainer


class MMFedRAPTrainer(FederatedTrainer):
    """MMFedRAP-specific trainer with per-round multiplicative LR decay (decay_rate,
    matching the FedRAP schedule). The schedule is model-specific and is
    persisted with the trainer resume state."""

    def _update_hyperparams(self, epoch_idx):
        decay_rate = self.config['decay_rate']
        self.config['learning_rate'] *= decay_rate
        self.config['server_learning_rate'] *= decay_rate

    def _resume_state_dict(self, completed_epochs: int, train_data) -> dict:
        # The LR schedule is updated in place each round; persist the current
        # decayed values so a resumed run continues the same optimization path.
        state = super()._resume_state_dict(completed_epochs, train_data)
        state["decayed_learning_rate"] = self.config["learning_rate"]
        state["decayed_server_learning_rate"] = self.config["server_learning_rate"]
        return state

    def _restore_resume_state(self, checkpoint: dict, train_data) -> None:
        super()._restore_resume_state(checkpoint, train_data)
        self.config["learning_rate"] = checkpoint["decayed_learning_rate"]
        self.config["server_learning_rate"] = checkpoint["decayed_server_learning_rate"]
