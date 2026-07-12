"""
CAM_AE: Contextual Attention Model - Autoencoder
Original source: https://github.com/jackfrost168/CF_Diff
"""

import torch as th
import torch.nn as nn

from .diffusion_utils import timestep_embedding


class CAM_AE(nn.Module):
    def __init__(
        self,
        in_dims,
        emb_size=10,
        norm=True,
        dropout=0.5,
        d_model=650,
        n_heads=1,
        n_layers=1,
        cross_fusion_weight=0.5,
    ):
        super(CAM_AE, self).__init__()

        self.in_dims = in_dims
        self.emb_size = emb_size
        self.norm = norm
        self.d_model = d_model
        self.cross_fusion_weight = cross_fusion_weight

        emb_size = d_model

        self.time_embed = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size),
        )

        self.encoder = nn.Sequential(
            nn.Linear(in_dims, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model)
        )

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Second hop embedding
        self.second_hop_embedding = nn.Linear(in_dims, d_model)

        # Final projection layers
        self.proj = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, in_dims)

        self.dropout = nn.Dropout(dropout)

        if self.norm:
            self.ln = nn.LayerNorm(in_dims)

    def forward(self, x, timesteps, x_sec_hop=None, return_dict=False):
        """
        Apply the model to an input batch with cross-attention on multi-hop features.

        Architecture follows CF-Diff paper:
        - Main stream: encodes current denoising state
        - Multi-hop stream: encodes high-order connectivity
        - Cross-attention: Q from multi-hop, K/V from main stream

        :param x: an [N x C x ...] Tensor of inputs (current noisy state).
        :param timesteps: a 1-D batch of timesteps.
        :param x_sec_hop: second hop features (if available)
        :return: an [N x C x ...] Tensor of outputs.
        """
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.emb_size))

        # Encode main stream (current denoising state)
        h = self.encoder(x)

        # Add time embedding to main stream
        h = h + emb

        # Apply self-attention layers on main stream
        for attention, norm in zip(self.attention_layers, self.norms):
            h_input = h.unsqueeze(1)  # Add sequence dimension for attention
            h_att, _ = attention(h_input, h_input, h_input)
            h = norm(h + h_att.squeeze(1))  # Residual connection
            h = self.dropout(h)

        # Cross-attention with second hop features (if provided)
        # Paper: Query from multi-hop neighbors, Key/Value from main stream
        if x_sec_hop is not None:
            # Encode second hop features separately
            h_sec_hop = self.second_hop_embedding(x_sec_hop)

            # Cross-attention: Q from second_hop, K/V from main stream
            # This conditions the denoising on high-order connectivity
            h_main = h.unsqueeze(1)  # [B, 1, d_model]
            h_query = h_sec_hop.unsqueeze(1)  # [B, 1, d_model]

            # Use first attention layer for cross-attention (reuse architecture)
            h_cross, _ = self.attention_layers[0](
                query=h_query,
                key=h_main,
                value=h_main
            )

            # Fuse cross-attention output with main stream
            # Weighted combination as in paper
            alpha = self.cross_fusion_weight
            h = alpha * h + (1 - alpha) * h_cross.squeeze(1)

        # Final projection
        h = th.tanh(self.proj(h))
        h = self.proj2(h)

        if self.norm:
            h = self.ln(h)

        return h