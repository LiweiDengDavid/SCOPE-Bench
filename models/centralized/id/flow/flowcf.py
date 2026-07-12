# coding: utf-8

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import RecommenderBase


def mean_flat(tensor):
    """Mean over non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_embedding_pi(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class FlowModel(nn.Module):
    """MLP flow model conditioned on timestep embeddings."""

    def __init__(
        self,
        dims: list[int],
        time_emb_size: int,
        time_type="cat",
        act_func="tanh",
        norm=False,
        init_dropout=0,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.dims = dims.copy()
        self.time_type = time_type
        self.time_emb_dim = time_emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            self.dims[0] += self.time_emb_dim
        else:
            raise ValueError(f"Unimplemented timestep embedding type {self.time_type}")

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i < len(self.dims) - 2:
                if act_func == "tanh":
                    self.layers.append(nn.Tanh())
                elif act_func == "relu":
                    self.layers.append(nn.ReLU())
                elif act_func == "sigmoid":
                    self.layers.append(nn.Sigmoid())
                self.layers.append(nn.Dropout(dropout_rate))
        self.init_dropout = nn.Dropout(init_dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.init_dropout(x)
        h = torch.cat([x, emb], dim=-1)
        for layer in self.layers:
            h = layer(h)
        return h


class FlowCF(RecommenderBase):
    """Flow matching for collaborative filtering."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.dataloader = dataloader
        self.n_steps = config['n_steps']
        self.s_steps = config['s_steps']
        self.time_steps = torch.linspace(0, 1, self.n_steps + 1)
        self.time_emb_size = config['time_embedding_size']
        dims = [self.n_items] + config['dims_mlp'] + [self.n_items]
        self.flow_model = FlowModel(
            dims=dims,
            time_emb_size=self.time_emb_size,
            init_dropout=config['init_dropout'],
            dropout_rate=config['dropout_rate'],
        )
        self.item_frequencies = self._compute_item_frequencies()
        self._build_history_items()

    def _compute_item_frequencies(self):
        item_counts = torch.zeros(self.n_items, device=self.device)
        train_data = self.dataloader.dataset.df
        item_field = self.dataloader.dataset.iid_field
        for _, row in train_data.iterrows():
            item_counts[row[item_field]] += 1
        return item_counts / self.n_users

    def _build_history_items(self):
        self.history_item_id = {}
        train_data = self.dataloader.dataset.df
        user_field = self.dataloader.dataset.uid_field
        item_field = self.dataloader.dataset.iid_field
        for user_id, group in train_data.groupby(user_field):
            self.history_item_id[user_id] = torch.tensor(group[item_field].tolist(), device=self.device)

    @staticmethod
    def _to_user_list(users):
        if isinstance(users, (list, tuple)):
            return list(users)
        return users.cpu().tolist()

    def get_rating_matrix(self, users):
        user_list = self._to_user_list(users)
        batch_size = len(user_list)
        rating_matrix = torch.zeros(batch_size, self.n_items, device=self.device)
        for i, user_id in enumerate(user_list):
            if user_id in self.history_item_id:
                item_ids = self.history_item_id[user_id]
                valid_item_ids = item_ids[item_ids < self.n_items]
                if valid_item_ids.numel() > 0:
                    rating_matrix[i, valid_item_ids] = 1.0
        return rating_matrix

    def flow_forward(self, x, t):
        return self.flow_model(x, t)

    def _sample_time_steps(self, batch_size, device):
        step_ids = torch.randint(0, self.n_steps, (batch_size,), device=device)
        return self.time_steps.to(device)[step_ids]

    def _sample_prior(self, batch_size, device):
        return torch.bernoulli(self.item_frequencies.expand(batch_size, -1)).to(device)

    @staticmethod
    def _interpolate(prior_sample, target_matrix, time_steps):
        random_mask = torch.rand_like(target_matrix, dtype=torch.float32) <= time_steps.unsqueeze(-1)
        return torch.where(random_mask, target_matrix, prior_sample)
    
    def calculate_loss(self, batch):
        users = batch[0].long()
        target_matrix = self.get_rating_matrix(users)
        time_steps = self._sample_time_steps(target_matrix.size(0), target_matrix.device)
        prior_sample = self._sample_prior(target_matrix.size(0), target_matrix.device)
        current_state = self._interpolate(prior_sample, target_matrix, time_steps)
        model_output = self.flow_forward(current_state, time_steps)
        return mean_flat((target_matrix - model_output) ** 2).mean()

    @staticmethod
    def _flow_velocity(current_state, next_state, time_steps):
        return (next_state - current_state) / (1 - time_steps.unsqueeze(-1) + 1e-8)

    @staticmethod
    def _step_state(current_state, velocity, time_steps, next_time_steps):
        delta_t = next_time_steps.unsqueeze(-1) - time_steps.unsqueeze(-1)
        pos_probs = current_state + velocity * delta_t
        neg_probs = 1 - pos_probs
        return torch.stack([neg_probs, pos_probs], dim=-1).argmax(dim=-1)

    def full_sort_predict(self, interaction, *args, **kwargs):
        users = interaction[0]
        observed = self.get_rating_matrix(users)
        current_state = observed.clone()
        next_state = current_state

        for i_t in range(self.n_steps - self.s_steps, self.n_steps):
            time_steps = self.time_steps[i_t].repeat(current_state.shape[0]).to(observed.device)
            next_state = self.flow_forward(current_state, time_steps)
            if i_t == self.n_steps - 1:
                break
            next_time_steps = self.time_steps[i_t + 1].repeat(current_state.shape[0]).to(observed.device)
            velocity = self._flow_velocity(current_state, next_state, time_steps)
            current_state = self._step_state(current_state, velocity, time_steps, next_time_steps)
            current_state = torch.logical_or(observed.bool(), current_state.bool()).float()

        return next_state

    def predict(self, users, items):
        full_scores = self.full_sort_predict([users])
        return full_scores.gather(1, items.unsqueeze(1)).squeeze(1)

    def forward(self, users, items):
        return self.predict(users, items)
