# coding: utf-8
"""DiffMM multimodal diffusion recommendation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.base import RecommenderBase
from core.utils import build_norm_adj_matrix
from .diffmm_components.fusion import (
    UserFeatureAggregator,
)
from .diffmm_components.diffusion import (
    create_diffusion_process
)
from .diffmm_components.contrastive import (
    create_contrastive_learning_module
)


class DiffMM(RecommenderBase):
    """DiffMM multimodal diffusion recommendation model.

    Core features:
    - Graph-diffusion-based user-item interaction modeling
    - Multimodal feature fusion (text, visual, audio)
    - Self-supervised contrastive learning
    - Adaptive modal weight learning
    """

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.setup_multimodal_features(config)
        self.dataloader = dataloader
        cfg = config
        loss_cfg = cfg['loss_weights']
        mm_cfg = cfg['multimodal']
        weight_cfg = mm_cfg['modal_fusion_weights']

        self.embedding_size = cfg['embedding_size']
        self.n_layers = cfg['num_layers']
        self.main_loss_weight = loss_cfg['main_loss']
        self.ssl_loss_weight = loss_cfg['ssl_loss']
        self.diffusion_loss_weight = loss_cfg['diffusion_loss']
        self.enable_diffusion = cfg['diffusion']['enable_diffusion']
        self.use_text = mm_cfg['use_text']
        self.use_image = mm_cfg['use_image']
        self.use_audio = mm_cfg['use_audio']
        self.learnable_weights = weight_cfg['learnable']
        self.init_equal_weights = weight_cfg['init_equal']
        self.normalize_weights = weight_cfg['normalization']
        self.interaction_matrix = None

        self._init_embeddings()
        self._init_graph_components()
        self._init_multimodal_components()
        self._init_fusion_weights()
        self._init_diffusion_components()
        self._init_ssl_components()
        self.norm_adj_matrix = build_norm_adj_matrix(
            dataloader.inter_matrix(form="coo").astype(np.float32),
            self.n_users,
            self.n_items,
            self.device,
        )

    def _init_embeddings(self):
        """Initialize base embedding layers."""
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _init_graph_components(self):
        """Initialize graph convolution components."""
        self.n_gcn_layers = self.n_layers
        self.dropout = nn.Dropout(self.dropout_rate)

    def _build_feature_transform(self, feature_dim):
        """Project one modality into the collaborative space."""
        return nn.Sequential(
            nn.Linear(feature_dim, self.embedding_size),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(self.embedding_size),
        )

    def _init_multimodal_components(self):
        """Initialize multimodal components."""
        if self.t_feat is not None and self.use_text:
            self.text_transform = self._build_feature_transform(self.t_feat.shape[1])

        if self.v_feat is not None and self.use_image:
            self.visual_transform = self._build_feature_transform(self.v_feat.shape[1])

        if hasattr(self, 'a_feat') and self.a_feat is not None and self.use_audio:
            self.audio_transform = self._build_feature_transform(self.a_feat.shape[1])

        user_agg_cfg = self.config['multimodal']['user_aggregation']
        self.user_aggregator = UserFeatureAggregator(
            embedding_size=self.embedding_size,
            aggregation_type=user_agg_cfg['type'],
            use_attention=user_agg_cfg['use_attention'],
        )

    def _count_modalities(self):
        """Count active modalities including the ID embedding."""
        count = 1
        if self.t_feat is not None and self.use_text:
            count += 1
        if self.v_feat is not None and self.use_image:
            count += 1
        if hasattr(self, 'a_feat') and self.a_feat is not None and self.use_audio:
            count += 1
        return count

    def _init_fusion_weights(self):
        """Initialize the global modal-fusion weight vector.

        Fusion uses one shared (static) learnable weight vector so training
        and evaluation fuse modalities identically.
        """
        if self.learnable_weights:
            n_modalities = self._count_modalities()
            if self.init_equal_weights:
                self.modal_fusion_weights = nn.Parameter(torch.full((n_modalities,), 1.0 / n_modalities))
            else:
                self.modal_fusion_weights = nn.Parameter(torch.randn(n_modalities) * 0.1)

    def _init_diffusion_components(self):
        """Initialize diffusion model components."""
        if self.enable_diffusion:
            self.diffusion_process = create_diffusion_process(
                config=self.config,
                input_dim=self.embedding_size,
                condition_dim=self.embedding_size * 2,
            )
        else:
            self.diffusion_process = None

    def _init_ssl_components(self):
        """Initialize SSL contrastive learning components."""
        if self.ssl_loss_weight > 0:
            self.contrastive_learning = create_contrastive_learning_module(
                config=self.config,
                embedding_dim=self.embedding_size,
            )
        else:
            self.contrastive_learning = None

    def _get_cached_interaction_matrix(self, device):
        """Cache the dense user-item interaction matrix on demand."""
        if self.interaction_matrix is None:
            matrix = self.dataloader.inter_matrix(form='coo').tocsr()
            self.interaction_matrix = torch.FloatTensor(matrix.toarray())
        return self.interaction_matrix.to(device)

    def _collect_modal_embeddings(self):
        """Collect per-modality user/item embeddings before fusion."""
        modal_embs = [(self.user_embedding.weight, self.item_embedding.weight)]

        if self.t_feat is not None and self.use_text and hasattr(self, 'text_transform'):
            item_emb = self.text_transform(self.t_feat)
            modal_embs.append((self._compute_user_feature_embedding(item_emb), item_emb))

        if self.v_feat is not None and self.use_image and hasattr(self, 'visual_transform'):
            item_emb = self.visual_transform(self.v_feat)
            modal_embs.append((self._compute_user_feature_embedding(item_emb), item_emb))

        if hasattr(self, 'a_feat') and self.a_feat is not None and self.use_audio and hasattr(self, 'audio_transform'):
            item_emb = self.audio_transform(self.a_feat)
            modal_embs.append((self._compute_user_feature_embedding(item_emb), item_emb))

        return modal_embs

    def _fuse_with_static_weights(self, modal_embs):
        """Fuse modalities with one global weight vector."""
        weights = self.modal_fusion_weights
        if self.normalize_weights:
            weights = F.softmax(weights, dim=0)
        user_emb = sum(weight * user for weight, (user, _) in zip(weights, modal_embs))
        item_emb = sum(weight * item for weight, (_, item) in zip(weights, modal_embs))
        return user_emb, item_emb

    def _get_multimodal_embeddings(self):
        """Get fused multimodal user/item embeddings.

        Uses one static learnable weight vector when learnable_weights is on,
        otherwise a plain modality average — the same path for train and eval.
        """
        modal_embs = self._collect_modal_embeddings()
        if self.learnable_weights and len(modal_embs) > 1:
            return self._fuse_with_static_weights(modal_embs)

        user_emb = sum(user for user, _ in modal_embs) / len(modal_embs)
        item_emb = sum(item for _, item in modal_embs) / len(modal_embs)
        return user_emb, item_emb

    def _compute_user_feature_embedding(self, item_features):
        """Compute user feature embeddings from the user's interaction history.

        Args:
            item_features: item feature embeddings [n_items, embedding_size]

        Returns:
            torch.Tensor: user feature embeddings [n_users, embedding_size]
        """
        user_item = self._get_cached_interaction_matrix(item_features.device)
        return self.user_aggregator(item_features, user_item)

    def forward(self, training=True):
        """Forward pass.

        Args:
            training: whether the model is in training mode

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output embeddings for users and items
        """
        user_embeddings, item_embeddings = self._get_multimodal_embeddings()
        if training:
            user_embeddings = self.dropout(user_embeddings)
            item_embeddings = self.dropout(item_embeddings)

        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        layer_embeddings = [all_embeddings]
        for _ in range(self.n_gcn_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            layer_embeddings.append(all_embeddings)

        final_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
        return final_embeddings[:self.n_users], final_embeddings[self.n_users:]

    def calculate_loss(self, interaction):
        """Calculate the loss function.

        Args:
            interaction: interaction data, supports 2-element or 3-element format

        Returns:
            torch.Tensor: total loss
        """
        if len(interaction) == 3:
            users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        elif len(interaction) == 2:
            users, pos_items = interaction[0], interaction[1]
            neg_items = torch.randint(0, self.n_items, pos_items.shape, device=pos_items.device)
        else:
            raise ValueError(f"Unexpected interaction format with {len(interaction)} elements")

        # Train on graph-propagated embeddings, matching the representation used
        # by full_sort_predict. forward(training=True) does fuse -> dropout -> GCN -> layer-mean.
        user_embeddings, item_embeddings = self.forward(training=True)
        total_loss = self.main_loss_weight * self._calculate_bpr_loss(
            user_embeddings,
            item_embeddings,
            users,
            pos_items,
            neg_items,
        )
        if self.ssl_loss_weight > 0 and self.contrastive_learning is not None:
            contrastive_out = self.contrastive_learning(
                user_embeddings, item_embeddings, users, pos_items
            )
            total_loss += self.ssl_loss_weight * contrastive_out['contrastive_loss']

        if self.enable_diffusion and self.diffusion_loss_weight > 0 and self.diffusion_process is not None:
            total_loss += self.diffusion_loss_weight * self._calculate_diffusion_loss(
                user_embeddings, item_embeddings, users, pos_items
            )

        return total_loss

    def _calculate_bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        """Calculate BPR loss.

        Args:
            user_emb: user embeddings [n_users, embedding_size]
            item_emb: item embeddings [n_items, embedding_size]
            users: user indices [batch_size]
            pos_items: positive item indices [batch_size]
            neg_items: negative item indices [batch_size]

        Returns:
            torch.Tensor: BPR loss
        """
        user_embed = user_emb[users]    # [batch_size, embedding_size]
        pos_embed = item_emb[pos_items]  # [batch_size, embedding_size]
        neg_embed = item_emb[neg_items]  # [batch_size, embedding_size]
        pos_scores = (user_embed * pos_embed).sum(dim=1)  # [batch_size]
        neg_scores = (user_embed * neg_embed).sum(dim=1)  # [batch_size]
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def _calculate_diffusion_loss(self, user_emb, item_emb, users, pos_items):
        """Calculate diffusion loss.

        Args:
            user_emb: user embeddings [n_users, embedding_size]
            item_emb: item embeddings [n_items, embedding_size]
            users: user indices [batch_size]
            pos_items: positive item indices [batch_size]

        Returns:
            torch.Tensor: diffusion loss
        """
        batch_user_emb = user_emb[users]      # [batch_size, embedding_size]
        batch_item_emb = item_emb[pos_items]  # [batch_size, embedding_size]
        target_embeddings = batch_user_emb * batch_item_emb  # [batch_size, embedding_size]
        condition = torch.cat([batch_user_emb, batch_item_emb], dim=-1)  # [batch_size, 2*embedding_size]
        diffusion_out = self.diffusion_process.training_losses(
            x_start=target_embeddings,
            condition=condition
        )
        return diffusion_out['diffusion_loss']

    def full_sort_predict(self, interaction):
        """Full-sort prediction.

        Args:
            interaction: interaction data containing users

        Returns:
            torch.Tensor: prediction scores [batch_size, n_items]
        """
        users = interaction[0]

        user_embeddings, item_embeddings = self.forward(training=False)
        user_emb = user_embeddings[users]  # [batch_size, embedding_size]
        return torch.mm(user_emb, item_embeddings.transpose(0, 1))
