# coding: utf-8
"""
FedRAP: Federated Recommendation with Personalization
"""

import math
import torch
import torch.nn as nn

from core.base import RecommenderBase, xavier_normal_initialization


class FedRAP(RecommenderBase):
    """FedRAP model with shared item commonality and client-side personality."""

    def __init__(self, config, dataloader):
        super(FedRAP, self).__init__(config, dataloader)
        self.config['server_learning_rate'] = self.config['learning_rate'] * self.n_items

        self.embedding_size = config['embedding_size']
        self.alpha = float(config['alpha'])
        self.beta = float(config['beta'])
        # Rounds over which the independency/reg loss weight warms up (tanh schedule).
        self.loss_warmup_rounds = float(config['loss_warmup_rounds'])

        # Client-owned item personality embeddings.
        self.item_personality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)

        # Server-owned item commonality embeddings.
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)

        # Output layer
        self.affine_output = torch.nn.Linear(in_features=self.embedding_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # FederatedTrainer updates this round counter before local optimization.
        self.current_epoch = 0

        # Initialize parameters
        self.apply(xavier_normal_initialization)

    def get_server_grad_param_names(self):
        """item_commonality uses delta aggregation (server-gradient path)."""
        return ['item_commonality.weight']

    def get_shared_parameters(self):
        """Shared global parameters."""
        return {
            "item_commonality.weight": self.item_commonality.weight,
        }

    def get_personal_parameters(self):
        """Personal client-specific parameters."""
        return {
            "item_personality.weight": self.item_personality.weight,
            "affine_output.weight": self.affine_output.weight,
            "affine_output.bias": self.affine_output.bias,
        }

    def forward(self, item_indices):
        """Forward pass - unified interface."""
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        pred = self.affine_output(item_personality + item_commonality)
        rating = self.logistic(pred)

        return rating, item_personality, item_commonality

    def calculate_loss(self, interaction):
        """Calculate loss - unified interface."""
        _, poss, negs = interaction[0], interaction[1], interaction[2]
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32, device=self.device)
        ratings[:poss.size(0)] = 1

        pred, item_personality, item_commonality = self.forward(items)

        # Loss: BCE, matching the reference implementation for this model.
        bce_loss = nn.BCELoss()(pred.view(-1), ratings)

        # Independence loss: mean-reduced MSE for batch-size invariance
        independency_loss = nn.functional.mse_loss(item_personality, item_commonality)

        # Regularization loss
        dummy_target = torch.zeros_like(item_commonality)
        reg_loss = nn.L1Loss()(item_commonality, dummy_target)

        scale = math.tanh(self.current_epoch / self.loss_warmup_rounds)
        total_loss = bce_loss - self.alpha * scale * independency_loss + self.beta * scale * reg_loss

        return total_loss

    def full_sort_predict(self, interaction, *args, **kwargs):
        """Full-sort prediction - unified interface."""
        # Ignore extra arguments since FedRAP is an ID-based model and does not use multimodal features
        items = torch.arange(self.n_items, device=self.device)
        scores, _, _ = self.forward(items)
        return scores.view(1, -1)


from core.federated import FederatedTrainer


class FedRAPTrainer(FederatedTrainer):
    """FedRAP-specific trainer with per-round multiplicative LR decay (decay_rate,
    per the FedRAP reference). The schedule is model-specific and is persisted
    with the trainer resume state."""

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
