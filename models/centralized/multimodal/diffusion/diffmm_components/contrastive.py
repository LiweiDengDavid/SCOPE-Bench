# coding: utf-8
"""
Contrastive Learning Module - DiffMM SSL Component
===================================================

Implements self-supervised contrastive learning functionality:
- Multimodal data augmentation strategies
- InfoNCE contrastive loss function
- Positive/negative sample construction
- Hard negative mining

Based on SimCLR and multimodal contrastive learning theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import random


class MultiModalAugmentation(nn.Module):
    """Multimodal data augmenter.

    Provides multiple augmentation strategies to improve contrastive learning:
    - Feature Dropout: randomly drop partial feature dimensions
    - Gaussian Noise: add random noise
    - Feature Masking: randomly mask partial modalities
    - Feature Mixing: mix features from different samples
    """

    def __init__(self,
                 dropout_rate: float = 0.1,
                 noise_scale: float = 0.1,
                 masking_rate: float = 0.2,
                 mix_up_alpha: float = 0.2,
                 augment_dropout_prob: float = 0.7,
                 augment_noise_prob: float = 0.6,
                 augment_mask_prob: float = 0.4):
        """Initialize the multimodal augmenter.

        Args:
            dropout_rate: feature dropout ratio
            noise_scale: noise intensity
            masking_rate: feature masking ratio
            mix_up_alpha: feature mixing parameter
            augment_dropout_prob: probability of applying dropout augmentation in create_augmented_views
            augment_noise_prob: probability of applying noise augmentation in create_augmented_views
            augment_mask_prob: probability of applying masking augmentation in create_augmented_views
        """
        super(MultiModalAugmentation, self).__init__()

        self.dropout_rate = dropout_rate
        self.noise_scale = noise_scale
        self.masking_rate = masking_rate
        self.mix_up_alpha = mix_up_alpha
        self.augment_dropout_prob = augment_dropout_prob
        self.augment_noise_prob = augment_noise_prob
        self.augment_mask_prob = augment_mask_prob

        # Learnable augmentation parameters
        self.noise_proj = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.noise_proj.weight, 1.0)

    def feature_dropout(self, x: torch.Tensor, rate: Optional[float] = None) -> torch.Tensor:
        """Feature dropout augmentation."""
        if not self.training:
            return x

        rate = rate or self.dropout_rate
        mask = torch.rand_like(x) > rate
        return x * mask / (1 - rate)

    def gaussian_noise(self, x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
        """Gaussian noise augmentation."""
        if not self.training:
            return x

        scale = scale or self.noise_scale
        noise = torch.randn_like(x) * scale
        # Use learnable noise intensity
        adaptive_scale = self.noise_proj(torch.tensor([[scale]], device=x.device))
        return x + noise * adaptive_scale.squeeze()

    def feature_masking(self, x: torch.Tensor, rate: Optional[float] = None) -> torch.Tensor:
        """Feature masking augmentation."""
        if not self.training:
            return x

        rate = rate or self.masking_rate
        batch_size, dim = x.shape
        mask_size = int(dim * rate)

        masked_x = x.clone()
        for i in range(batch_size):
            mask_indices = torch.randperm(dim)[:mask_size]
            masked_x[i, mask_indices] = 0

        return masked_x

    def create_augmented_views(self, x: torch.Tensor, num_views: int = 2) -> List[torch.Tensor]:
        """Create multiple augmented views."""
        views = []

        for _ in range(num_views):
            # Combine multiple augmentation strategies
            aug_x = x

            # Randomly apply augmentations
            if random.random() < self.augment_dropout_prob:
                aug_x = self.feature_dropout(aug_x)

            if random.random() < self.augment_noise_prob:
                aug_x = self.gaussian_noise(aug_x)

            if random.random() < self.augment_mask_prob:
                aug_x = self.feature_masking(aug_x)

            views.append(aug_x)

        return views


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss function.

    Implements the standard InfoNCE loss, supporting:
    - Multiple similarity computation methods
    - Temperature parameter tuning
    - Hard negative mining
    - In-batch negative sampling
    """

    def __init__(self,
                 temperature: float = 0.1,
                 similarity_type: str = "cosine",
                 negative_mining: bool = True,
                 mining_ratio: float = 0.5):
        """Initialize the InfoNCE loss.

        Args:
            temperature: temperature parameter controlling softmax sharpness
            similarity_type: similarity computation type ("cosine", "dot", "l2")
            negative_mining: whether to enable hard negative mining
            mining_ratio: ratio of hard negatives to select
        """
        super(InfoNCELoss, self).__init__()

        self.temperature = temperature
        self.similarity_type = similarity_type
        self.negative_mining = negative_mining
        self.mining_ratio = mining_ratio

    def compute_similarity(self, anchor: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix."""
        if self.similarity_type == "cosine":
            # Cosine similarity
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            targets_norm = F.normalize(targets, p=2, dim=1)
            similarity = torch.mm(anchor_norm, targets_norm.t())

        elif self.similarity_type == "dot":
            # Dot product similarity
            similarity = torch.mm(anchor, targets.t())

        elif self.similarity_type == "l2":
            # L2 distance (converted to similarity)
            anchor_expanded = anchor.unsqueeze(1)   # [batch, 1, dim]
            targets_expanded = targets.unsqueeze(0)  # [1, batch, dim]
            l2_distance = torch.norm(anchor_expanded - targets_expanded, p=2, dim=2)
            similarity = -l2_distance  # smaller distance means higher similarity

        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")

        return similarity / self.temperature

    def hard_negative_mining(self, similarity: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """Hard negative mining."""
        # Obtain negative sample similarities
        negative_mask = ~positive_mask
        negative_similarity = similarity * negative_mask.float() + positive_mask.float() * (-1e9)

        # Select hard negatives with the highest similarity
        batch_size = similarity.size(0)
        num_negatives = negative_mask.sum(dim=1)
        num_hard_negatives = (num_negatives.float() * self.mining_ratio).long()

        hard_negative_mask = torch.zeros_like(negative_mask)

        for i in range(batch_size):
            if num_hard_negatives[i] > 0:
                _, top_indices = torch.topk(negative_similarity[i], num_hard_negatives[i])
                hard_negative_mask[i, top_indices] = True

        return hard_negative_mask

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negatives: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass to compute the InfoNCE loss.

        Args:
            anchor: anchor embeddings [batch_size, dim]
            positive: positive sample embeddings [batch_size, dim]
            negatives: negative sample embeddings [batch_size, num_negatives, dim] or None (use in-batch negatives)

        Returns:
            Loss dictionary containing total loss and statistics
        """
        batch_size = anchor.size(0)

        if negatives is None:
            # In-batch negative sampling: use other samples in the batch as negatives
            all_samples = torch.cat([positive, anchor], dim=0)  # [2*batch_size, dim]

            # Compute similarity matrix
            similarity = self.compute_similarity(anchor, all_samples)  # [batch_size, 2*batch_size]

            # Exclude each anchor's own embedding (column batch_size+i) from acting
            # as an in-batch negative — its self-similarity would dominate the
            # InfoNCE denominator and contaminate the contrastive signal.
            self_idx = torch.arange(batch_size, device=anchor.device)
            similarity[self_idx, batch_size + self_idx] = -1e9

            # Create positive sample mask (first batch_size entries are positives)
            positive_mask = torch.zeros(batch_size, 2 * batch_size, dtype=torch.bool, device=anchor.device)
            positive_mask[range(batch_size), range(batch_size)] = True  # corresponding positive samples

        else:
            # Use provided negative samples
            num_negatives = negatives.size(1)
            all_samples = torch.cat([positive.unsqueeze(1), negatives], dim=1)  # [batch_size, 1+num_negatives, dim]
            all_samples = all_samples.view(-1, all_samples.size(-1))  # [batch_size*(1+num_negatives), dim]

            # Compute similarity
            similarity = self.compute_similarity(anchor, all_samples)  # [batch_size, batch_size*(1+num_negatives)]
            similarity = similarity.view(batch_size, batch_size, 1 + num_negatives)

            # Keep only diagonal similarities (each anchor corresponds to its own pos/neg samples)
            similarity = similarity[range(batch_size), range(batch_size), :]  # [batch_size, 1+num_negatives]

            # Positive sample mask
            positive_mask = torch.zeros(batch_size, 1 + num_negatives, dtype=torch.bool, device=anchor.device)
            positive_mask[:, 0] = True  # first entry is the positive sample

        # Hard negative mining
        if self.negative_mining:
            negative_mask = self.hard_negative_mining(similarity, positive_mask)
            # Combine positive samples and hard negatives
            selected_mask = positive_mask | negative_mask
            # Recompute similarity (only for selected samples)
            masked_similarity = similarity * selected_mask.float() + (~selected_mask).float() * (-1e9)
        else:
            masked_similarity = similarity

        # Compute InfoNCE loss
        if negatives is None:
            # In-batch negative sampling case
            labels = torch.arange(batch_size, device=anchor.device)
        else:
            # Provided negative samples case
            labels = torch.zeros(batch_size, device=anchor.device, dtype=torch.long)

        loss = F.cross_entropy(masked_similarity, labels)

        # Compute accuracy and other statistics
        with torch.no_grad():
            _, predicted = torch.max(masked_similarity, 1)
            accuracy = (predicted == labels).float().mean()

            # Positive sample similarity statistics
            pos_similarity = similarity[positive_mask].mean()

            # Negative sample similarity statistics
            neg_mask = ~positive_mask
            if neg_mask.any():
                neg_similarity = similarity[neg_mask].mean()
            else:
                neg_similarity = torch.tensor(0.0, device=anchor.device)

        return {
            'contrastive_loss': loss,
            'contrastive_accuracy': accuracy,
            'positive_similarity': pos_similarity,
            'negative_similarity': neg_similarity,
            'temperature': self.temperature
        }


class ContrastiveLearningModule(nn.Module):
    """Complete contrastive learning module.

    Integrates data augmentation and contrastive loss, providing a unified interface.
    """

    def __init__(self,
                 embedding_dim: int,
                 temperature: float = 0.1,
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 contrastive_method: int = 1):
        """Initialize the contrastive learning module.

        Args:
            embedding_dim: embedding dimension
            temperature: contrastive learning temperature parameter
            augmentation_config: data augmentation configuration
            contrastive_method: contrastive method type (1: user contrast, 2: item contrast, 3: cross-modal contrast)
        """
        super(ContrastiveLearningModule, self).__init__()

        self.embedding_dim = embedding_dim
        self.contrastive_method = contrastive_method

        # Data augmenter
        aug_config = augmentation_config or {}
        self.augmentation = MultiModalAugmentation(
            dropout_rate=aug_config['dropout_rate'],
            noise_scale=aug_config['noise_scale'],
            masking_rate=aug_config['masking_rate'],
            mix_up_alpha=aug_config['mix_up_alpha'],
            augment_dropout_prob=aug_config['augment_dropout_prob'],
            augment_noise_prob=aug_config['augment_noise_prob'],
            augment_mask_prob=aug_config['augment_mask_prob'],
        )

        # Contrastive loss function
        self.infonce_loss = InfoNCELoss(
            temperature=temperature,
            similarity_type="cosine",
            negative_mining=True,
            mining_ratio=aug_config['mining_ratio'],
        )

        # Projection head (optional)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2)
        )

    def create_contrastive_pairs(self, user_embeddings: torch.Tensor,
                                item_embeddings: torch.Tensor,
                                users: torch.Tensor,
                                items: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create positive/negative sample pairs for contrastive learning."""
        # Retrieve user and item embeddings
        batch_user_emb = user_embeddings[users]  # [batch_size, embedding_dim]
        batch_item_emb = item_embeddings[items]  # [batch_size, embedding_dim]

        contrastive_pairs = {}

        if self.contrastive_method == 1:
            # User contrast: two augmented views of the same user
            user_view1 = self.augmentation.create_augmented_views(batch_user_emb, num_views=1)[0]
            user_view2 = self.augmentation.create_augmented_views(batch_user_emb, num_views=1)[0]

            contrastive_pairs['anchor'] = self.projection_head(user_view1)
            contrastive_pairs['positive'] = self.projection_head(user_view2)

        elif self.contrastive_method == 2:
            # Item contrast: two augmented views of the same item
            item_view1 = self.augmentation.create_augmented_views(batch_item_emb, num_views=1)[0]
            item_view2 = self.augmentation.create_augmented_views(batch_item_emb, num_views=1)[0]

            contrastive_pairs['anchor'] = self.projection_head(item_view1)
            contrastive_pairs['positive'] = self.projection_head(item_view2)

        elif self.contrastive_method == 3:
            # Cross-modal contrast: user-item interaction contrast
            interaction_emb = batch_user_emb + batch_item_emb  # simple interaction representation

            view1 = self.augmentation.create_augmented_views(interaction_emb, num_views=1)[0]
            view2 = self.augmentation.create_augmented_views(interaction_emb, num_views=1)[0]

            contrastive_pairs['anchor'] = self.projection_head(view1)
            contrastive_pairs['positive'] = self.projection_head(view2)

        return contrastive_pairs

    def forward(self, user_embeddings: torch.Tensor,
                item_embeddings: torch.Tensor,
                users: torch.Tensor,
                items: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to compute the contrastive learning loss."""

        # Create contrastive sample pairs
        contrastive_pairs = self.create_contrastive_pairs(
            user_embeddings, item_embeddings, users, items
        )

        # Compute InfoNCE loss
        loss_dict = self.infonce_loss(
            anchor=contrastive_pairs['anchor'],
            positive=contrastive_pairs['positive']
        )

        return loss_dict


# Convenience function
def create_contrastive_learning_module(config: Dict[str, Any],
                                     embedding_dim: int) -> ContrastiveLearningModule:
    """Create a ContrastiveLearningModule instance.

    Args:
        config: configuration dictionary
        embedding_dim: embedding dimension

    Returns:
        ContrastiveLearningModule instance
    """
    contrastive_config = config['contrastive_learning']
    aug_config = dict(contrastive_config['augmentation'])

    return ContrastiveLearningModule(
        embedding_dim=embedding_dim,
        temperature=contrastive_config['temperature'],
        augmentation_config=aug_config,
        contrastive_method=contrastive_config['contrastive_method']
    )
