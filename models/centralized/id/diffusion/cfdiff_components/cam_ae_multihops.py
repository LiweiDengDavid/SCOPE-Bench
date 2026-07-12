"""
CAM_AE_multihops: Multi-hop Contextual Attention Model - Autoencoder
Original source: https://github.com/jackfrost168/CF_Diff
"""

import torch as th
import torch.nn as nn

from .diffusion_utils import timestep_embedding


class CAM_AE_multihops(nn.Module):
    def __init__(
        self,
        in_dims,
        emb_size=10,
        norm=True,
        dropout=0.5,
        d_model=650,
        n_heads=1,
        n_layers=1,
        hop_fusion_weights=(0.5, 0.3, 0.2),
    ):
        super(CAM_AE_multihops, self).__init__()

        self.in_dims = in_dims
        self.emb_size = emb_size
        self.norm = norm
        self.d_model = d_model
        self.hop_fusion_weights = hop_fusion_weights

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

        # Multi-hop embeddings
        self.second_hop_embedding = nn.Linear(in_dims, d_model)
        self.third_hop_embedding = nn.Linear(in_dims, d_model)

        # Additional attention for second hop
        self.second_hop_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.second_hop_norm = nn.LayerNorm(d_model)

        # Additional attention for third hop
        self.third_hop_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.third_hop_norm = nn.LayerNorm(d_model)

        # Final projection layers
        self.proj = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, in_dims)

        self.dropout = nn.Dropout(dropout)

        if self.norm:
            self.ln = nn.LayerNorm(in_dims)

    def forward(self, x, timesteps, x_sec_hop=None, return_dict=False):
        """
        Apply the model to an input batch with multi-hop cross-attention.

        Architecture follows CF-Diff paper for multi-hop connectivity:
        - Main stream: encodes current denoising state
        - Multi-hop streams: encode 2-hop and 3-hop neighborhoods
        - Cross-attention: Q^(h) from each hop, K/V from main stream
        - Aggregation: z̄_t = Σ_h α_h f_h(Attention_h)

        :param x: an [N x C x ...] Tensor of inputs (current noisy state).
        :param timesteps: a 1-D batch of timesteps.
        :param x_sec_hop: multi-hop features concatenated [second_hop | third_hop]
        :return: an [N x C x ...] Tensor of outputs.
        """
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.emb_size))

        # Encode main stream (1-hop: current denoising state)
        h_main = self.encoder(x)

        # Add time embedding to main stream
        h_main = h_main + emb

        # Apply self-attention layers on main stream
        for attention, norm in zip(self.attention_layers, self.norms):
            h_input = h_main.unsqueeze(1)
            h_att, _ = attention(h_input, h_input, h_input)
            h_main = norm(h_main + h_att.squeeze(1))
            h_main = self.dropout(h_main)

        # Cross-attention with multi-hop features (if provided)
        # Paper formula: z̄_t = Σ_h α_h f_h(Attention_h(Q^(h), K_t, V_t))
        if x_sec_hop is not None:
            # Split multi-hop features [second_hop | third_hop]
            second_hop_feat = x_sec_hop[:, 0:self.in_dims]
            third_hop_feat = x_sec_hop[:, self.in_dims:2*self.in_dims]

            # Encode each hop separately
            h_2hop = self.second_hop_embedding(second_hop_feat)
            h_3hop = self.third_hop_embedding(third_hop_feat)

            # Prepare main stream as Key/Value for cross-attention
            h_main_kv = h_main.unsqueeze(1)  # [B, 1, d_model]

            # Cross-attention for 2-hop: Q from 2-hop, K/V from main
            h_2hop_query = h_2hop.unsqueeze(1)  # [B, 1, d_model]
            h_2hop_cross, _ = self.second_hop_attention(
                query=h_2hop_query,
                key=h_main_kv,
                value=h_main_kv
            )
            h_2hop_cross = self.second_hop_norm(h_2hop + h_2hop_cross.squeeze(1))

            # Cross-attention for 3-hop: Q from 3-hop, K/V from main
            h_3hop_query = h_3hop.unsqueeze(1)  # [B, 1, d_model]
            h_3hop_cross, _ = self.third_hop_attention(
                query=h_3hop_query,
                key=h_main_kv,
                value=h_main_kv
            )
            h_3hop_cross = self.third_hop_norm(h_3hop + h_3hop_cross.squeeze(1))

            # Weighted aggregation following paper
            # α_h decreases with hop distance (paper uses various α values)
            alpha_1, alpha_2, alpha_3 = self.hop_fusion_weights  # 1/2/3-hop weights from config
            # Normalize weights
            total_alpha = alpha_1 + alpha_2 + alpha_3
            alpha_1, alpha_2, alpha_3 = alpha_1/total_alpha, alpha_2/total_alpha, alpha_3/total_alpha

            # Aggregate: z̄_t = α_1·h_main + α_2·h_2hop + α_3·h_3hop
            h = alpha_1 * h_main + alpha_2 * h_2hop_cross + alpha_3 * h_3hop_cross
        else:
            h = h_main

        # Final projection
        h = th.tanh(self.proj(h))
        h = self.proj2(h)

        if self.norm:
            h = self.ln(h)

        return h