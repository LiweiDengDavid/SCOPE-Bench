# coding: utf-8
"""
RecVAE aligned with the RecBole reference implementation.
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase, xavier_normal_initialization


def swish(x):
    """Swish activation used by the reference encoder."""
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    """Elementwise log-density of a diagonal Gaussian."""
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class Encoder(nn.Module):
    """Residual encoder matching the RecBole RecVAE structure."""

    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_prob):
        x = F.normalize(x)
        x = F.dropout(x, dropout_prob, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class CompositePrior(nn.Module):
    """Composite prior used by RecVAE."""

    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights):
        super().__init__()
        self.mixture_weights = mixture_weights

        self.register_buffer("mu_prior", torch.zeros(1, latent_dim))
        self.register_buffer("logvar_prior", torch.zeros(1, latent_dim))
        self.register_buffer("logvar_uniform_prior", torch.full((1, latent_dim), 10.0))

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        self.encoder_old.eval()

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        standard_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        posterior_prior = log_norm_pdf(z, post_mu, post_logvar)
        uniform_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [standard_prior, posterior_prior, uniform_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        density_per_gaussian = torch.stack(gaussians, dim=-1)
        return torch.logsumexp(density_per_gaussian, dim=-1)


class RecVAE(RecommenderBase):
    """RecVAE with reference-aligned loss and alternating phase hooks."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        self.hidden_dimension = int(config["hidden_dim"])
        self.latent_dimension = int(config["latent_dim"])
        self.dropout_prob = float(config["dropout_rate"])
        self.beta = float(config["beta"])
        self.gamma = float(config["gamma"])
        self.encoder_epochs = int(config["encoder_epochs"])
        self.decoder_epochs = int(config["decoder_epochs"])
        self.mixture_weights = config["mixture_weights"]

        self.encoder = Encoder(self.hidden_dimension, self.latent_dimension, self.n_items)
        self.prior = CompositePrior(
            self.hidden_dimension,
            self.latent_dimension,
            self.n_items,
            self.mixture_weights,
        )
        self.decoder = nn.Linear(self.latent_dimension, self.n_items)

        self.training_phase = "encoder"
        self.phase_epoch_count = 0

        self._build_history_item_matrix(dataloader)
        self.apply(xavier_normal_initialization)
        self.update_prior()

    def _build_history_item_matrix(self, dataloader):
        """Cache the user-history matrix as a model buffer."""
        inter_matrix = dataloader.inter_matrix(form="csr").astype(np.float32)
        dense_matrix = torch.FloatTensor(inter_matrix.toarray())
        self.register_buffer("_history_matrix", dense_matrix)

    def get_user_history(self, user_ids):
        """Return dense interaction vectors for the requested users."""
        if user_ids.dim() == 0:
            user_ids = user_ids.unsqueeze(0)
        return self._history_matrix[user_ids]

    def _extract_user_ids(self, interaction):
        """Support both dict and tuple/list interaction formats."""
        if isinstance(interaction, dict):
            return interaction[self.USER_ID]
        return interaction[0]

    def get_optimizer_params(self):
        """Alternate between encoder and decoder optimization phases.

        The two phases return disjoint parameter sets, so TrainerBase rebuilds
        the optimizer/scheduler on each phase switch (see _refresh_optimizer);
        per-phase optimizer state is reset by design, matching the RecVAE
        reference's alternating-optimizer training.
        """
        if self.training_phase == "encoder":
            return self.encoder.parameters()
        return self.decoder.parameters()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        return mu

    def forward(self, rating_matrix, dropout_prob=0.0):
        mu, logvar = self.encoder(rating_matrix, dropout_prob=dropout_prob)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        return x_pred, mu, logvar, z

    def calculate_loss(self, interaction):
        """Negative ELBO with composite-prior KL term."""
        user_id = self._extract_user_ids(interaction)
        rating_matrix = self.get_user_history(user_id)

        dropout_prob = self.dropout_prob if self.training_phase == "encoder" else 0.0
        x_pred, mu, logvar, z = self.forward(rating_matrix, dropout_prob)

        if self.gamma:
            kl_weight = self.gamma * rating_matrix.sum(dim=-1)
        else:
            kl_weight = torch.full(
                (rating_matrix.size(0),),
                self.beta,
                device=rating_matrix.device,
            )

        marginal_log_likelihood = (
            F.log_softmax(x_pred, dim=-1) * rating_matrix
        ).sum(dim=-1).mean()
        kl_divergence = (
            (log_norm_pdf(z, mu, logvar) - self.prior(rating_matrix, z))
            .sum(dim=-1)
            .mul(kl_weight)
            .mean()
        )
        return -(marginal_log_likelihood - kl_divergence)

    def predict(self, interaction):
        user_id = self._extract_user_ids(interaction)
        rating_matrix = self.get_user_history(user_id)
        scores, _, _, _ = self.forward(rating_matrix, self.dropout_prob)

        if isinstance(interaction, dict) and self.ITEM_ID in interaction:
            item_id = interaction[self.ITEM_ID]
            return scores[torch.arange(len(item_id), device=scores.device), item_id]
        if not isinstance(interaction, dict) and len(interaction) > 1:
            item_id = interaction[1]
            return scores[torch.arange(len(item_id), device=scores.device), item_id]
        return scores

    def full_sort_predict(self, interaction):
        user_id = self._extract_user_ids(interaction)
        rating_matrix = self.get_user_history(user_id)
        scores, _, _, _ = self.forward(rating_matrix, self.dropout_prob)
        return scores

    def update_prior(self):
        """Refresh the frozen posterior prior snapshot."""
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
        self.prior.encoder_old.eval()

    def get_resume_state(self):
        """Persist the alternating-phase counters (plain attrs, absent from
        state_dict) so a resumed run continues the encoder/decoder cycle instead
        of replaying it from 'encoder'/0."""
        return {
            "training_phase": self.training_phase,
            "phase_epoch_count": self.phase_epoch_count,
        }

    def set_resume_state(self, state):
        self.training_phase = state["training_phase"]
        self.phase_epoch_count = state["phase_epoch_count"]

    def post_epoch_processing(self):
        """Advance alternating training phases after each epoch."""
        self.phase_epoch_count += 1

        if self.training_phase == "encoder":
            if self.phase_epoch_count >= self.encoder_epochs:
                self.update_prior()
                self.training_phase = "decoder"
                self.phase_epoch_count = 0
        elif self.phase_epoch_count >= self.decoder_epochs:
            self.training_phase = "encoder"
            self.phase_epoch_count = 0
