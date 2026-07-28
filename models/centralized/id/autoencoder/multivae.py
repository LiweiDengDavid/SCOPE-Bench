# coding: utf-8
"""
MultiVAE (Multinomial Variational Autoencoder) - ported from RecBole baseline
==============================================================================

Variational autoencoder for collaborative filtering, modeling user-item
interactions with a multinomial likelihood function.

Reference:
    Dawen Liang et al. "Variational Autoencoders for Collaborative Filtering." in WWW 2018.

RecBole Reference Implementation:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/multivae.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import RecommenderBase, xavier_normal_initialization


class MultiVAE(RecommenderBase):
    """MultiVAE baseline model - Multinomial Variational Autoencoder

    Models user implicit feedback with a variational autoencoder architecture:
    - Encoder: encodes the user interaction vector into a latent representation
    - Reparameterization trick: samples from the latent distribution
    - Decoder: reconstructs the user's item preference distribution
    - Variational lower bound: reconstruction loss + KL divergence regularization
    """
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        
        # Core parameters
        self.latent_dim = config['latent_dim']
        self.dropout_prob = config['dropout_rate']

        # Accept either a scalar first-layer size or an explicit hidden-size list.
        mlp_hidden_size_param = config['mlp_hidden_size']
        n_layers = config['num_layers']

        if isinstance(mlp_hidden_size_param, list):
            # Old format: use the list directly
            self.mlp_hidden_size = mlp_hidden_size_param
        else:
            # New format: generate list from first-layer dimension and number of layers
            first_dim = mlp_hidden_size_param
            self.mlp_hidden_size = [first_dim // (2**i) for i in range(n_layers)]
            # Ensure all dimensions are positive
            self.mlp_hidden_size = [max(dim, 1) for dim in self.mlp_hidden_size]

        self.anneal_cap = config['anneal_cap']
        self.total_anneal_steps = config['total_anneal_steps']

        # Current training step count (used for KL annealing)
        self.update_count = 0

        # Encoder
        encoder_layers = []
        encoder_size = [self.n_items] + self.mlp_hidden_size + [self.latent_dim]

        for i in range(len(encoder_size) - 1):
            encoder_layers.append(nn.Linear(encoder_size[i], encoder_size[i + 1]))
            if i < len(encoder_size) - 2:  # No activation on the last layer
                encoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        decoder_size = [self.latent_dim // 2] + self.mlp_hidden_size[::-1] + [self.n_items]

        for i in range(len(decoder_size) - 1):
            decoder_layers.append(nn.Linear(decoder_size[i], decoder_size[i + 1]))
            if i < len(decoder_size) - 2:  # No activation on the last layer
                decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # Weight initialization
        self.apply(xavier_normal_initialization)

        # Build user-item interaction matrix (registered as the _history_matrix
        # buffer inside; read via get_user_history). The return value is the same
        # buffer tensor, so no separate attribute is kept (matches RecVAE).
        self._build_history_item_matrix(dataloader)

    def _build_history_item_matrix(self, dataloader):
        """Build user history interaction matrix [n_users, n_items]"""
        inter_matrix = dataloader.inter_matrix(form='csr').astype(np.float32)
        dense_matrix = torch.FloatTensor(inter_matrix.toarray())
        self.register_buffer('_history_matrix', dense_matrix)
        return dense_matrix

    def get_user_history(self, user_ids):
        """Get history interaction vectors for the specified users"""
        if user_ids.dim() == 0:
            user_ids = user_ids.unsqueeze(0)
        return self._history_matrix[user_ids]  # [batch_size, n_items]

    def reparameterize(self, mu, logvar):
        """Reparameterization trick.

        Samples from N(mu, var) while keeping gradients flowing via the
        reparameterization trick.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            # Small-std (0.01) reparameterization noise, matching the RecBole MultiVAE
            # baseline this ports and the sibling RecVAE — keeps the sampled latent close
            # to mu (unit-variance randn would inject ~100x the train-time stochasticity).
            eps = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, rating_matrix):
        """Forward pass - VAE architecture

        Args:
            rating_matrix: User-item rating matrix [batch_size, n_items]

        Returns:
            Reconstructed rating matrix
        """
        # Normalize input
        normalized_rating_matrix = F.normalize(rating_matrix)
        normalized_rating_matrix = self.dropout(normalized_rating_matrix)

        # Encoder - obtain mean and log-variance
        h = self.encoder(normalized_rating_matrix)
        mu = h[:, :self.latent_dim // 2]
        logvar = h[:, self.latent_dim // 2:]

        # Reparameterization sampling
        z = self.reparameterize(mu, logvar)

        # Decoder - reconstruction
        recon_rating_matrix = self.decoder(z)

        return recon_rating_matrix, mu, logvar
    
    def calculate_loss(self, interaction):
        """Calculate variational lower bound loss"""
        user = interaction[0]

        # Get the user's true history interaction vector
        rating_matrix = self.get_user_history(user)  # [batch_size, n_items]

        # VAE forward pass
        recon_rating_matrix, mu, logvar = self.forward(rating_matrix)

        # Reconstruction loss - multinomial likelihood
        # Using log softmax + negative log likelihood
        log_softmax_var = F.log_softmax(recon_rating_matrix, dim=1)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * rating_matrix, dim=1))

        # KL divergence loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Update step counter BEFORE computing the anneal coefficient (matches
        # RecBole multivae.py, where `self.update += 1` precedes the anneal min(),
        # so the first training batch uses a non-zero anneal step).
        self.update_count += 1

        # KL annealing
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        # Total loss
        loss = neg_ll + anneal * kl_loss

        return loss

    def get_resume_state(self):
        """Persist the KL-annealing step counter (a plain attr, not in state_dict)
        so a resumed run continues the anneal schedule instead of restarting at 0."""
        return {"update_count": self.update_count}

    def set_resume_state(self, state):
        self.update_count = state["update_count"]

    def predict(self, interaction):
        """Predict scores"""
        if isinstance(interaction, dict):
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
        else:
            user = interaction[0]
            item = interaction[1]

        # Handle scalar user
        if user.dim() == 0:
            user = user.unsqueeze(0)

        # Get the user's true history interaction vector
        rating_matrix = self.get_user_history(user)  # [batch_size, n_items]

        # Forward pass
        with torch.no_grad():
            recon_rating_matrix, _, _ = self.forward(rating_matrix)

        # Get prediction scores for the specified items
        scores = recon_rating_matrix.gather(1, item.unsqueeze(1)).squeeze(1)
        return scores
    
    def full_sort_predict(self, interaction):
        """Full-sort prediction"""
        if isinstance(interaction, dict):
            user = interaction[self.USER_ID]
        else:
            user = interaction[0]

        # Handle user dimension
        if user.dim() == 0:
            user = user.unsqueeze(0)

        # Get the user's true history interaction vector
        rating_matrix = self.get_user_history(user)  # [batch_size, n_items]

        # Forward pass
        with torch.no_grad():
            recon_rating_matrix, _, _ = self.forward(rating_matrix)

        return recon_rating_matrix
