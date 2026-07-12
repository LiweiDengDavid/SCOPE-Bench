import torch
import torch.nn as nn
from typing import Optional


class SumExpert(nn.Module):
    """Sum expert: simply adds three input embedding vectors together.

    This is the simplest fusion method, directly summing three feature
    vectors without introducing additional parameters. Suitable when
    feature vectors share the same semantic space and have similar importance.
    """

    def __init__(self):
        """Initialize the sum expert"""
        super(SumExpert, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Add three embedding vectors together

        Args:
            x: First embedding vector, shape [batch_size, embed_size]
            y: Second embedding vector, shape [batch_size, embed_size]
            z: Third embedding vector, shape [batch_size, embed_size]

        Returns:
            Fused embedding vector, shape [batch_size, embed_size]
        """
        # Verify that input dimensions match
        if not (x.shape == y.shape == z.shape):
            raise ValueError(f"Input shapes do not match: {x.shape}, {y.shape}, {z.shape}")

        # Directly sum the three input embedding vectors
        return x + y + z


class MLPExpert(nn.Module):
    """MLP expert: fuses three embeddings using a multi-layer perceptron.

    Concatenates the three feature vectors and applies a non-linear
    transformation via an MLP, enabling learning of more complex feature
    interactions.
    """

    def __init__(self, embed_size: int, hidden_size: Optional[int] = None, dropout: float = 0.1):
        """Initialize the MLP expert

        Args:
            embed_size: Dimension of the input embedding vectors
            hidden_size: Hidden layer dimension, defaults to embed_size * 2
            dropout: Dropout probability for regularization
        """
        super(MLPExpert, self).__init__()

        if hidden_size is None:
            hidden_size = embed_size * 2

        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.LayerNorm(embed_size)  # Layer normalization for improved stability
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Fuse three embedding vectors using MLP

        Args:
            x: First embedding vector, shape [batch_size, embed_size]
            y: Second embedding vector, shape [batch_size, embed_size]
            z: Third embedding vector, shape [batch_size, embed_size]

        Returns:
            Fused embedding vector, shape [batch_size, embed_size]
        """
        # Concatenate the three embedding vectors and feed into the MLP
        concat_features = torch.cat([x, y, z], dim=-1)
        fused_features = self.mlp(concat_features)

        return fused_features


class MultiHeadAttentionExpert(nn.Module):
    """Multi-head attention expert: self-attention across the three modalities.

    The modalities form a length-3 sequence (one token each), so the softmax
    runs over three keys and the attention weights genuinely depend on the
    query/key projections. The previous formulation self-attended over a
    length-1 sequence (softmax over a single key is identically 1), which made
    the module provably linear with dead q/k parameters.
    """

    def __init__(self, embed_size: int, num_heads: int = 4, dropout: float = 0.1):
        """Initialize the multi-head attention expert

        Args:
            embed_size: Dimension of the input embedding vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttentionExpert, self).__init__()

        # Ensure embed_size is divisible by num_heads
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        # Multi-head self-attention across the modality tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape [batch_size, seq_len, embed_dim]
        )

        # Fusion layer (CrossAttentionExpert pattern): concat the attended
        # modality tokens and map back to the embedding dimension.
        self.fusion = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),
            nn.LayerNorm(embed_size)
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Fuse three embedding vectors using multi-head self-attention

        Args:
            x: First embedding vector, shape [batch_size, embed_size]
            y: Second embedding vector, shape [batch_size, embed_size]
            z: Third embedding vector, shape [batch_size, embed_size]

        Returns:
            Fused embedding vector, shape [batch_size, embed_size]
        """
        # Stack modalities as a length-3 sequence
        sequence = torch.stack([x, y, z], dim=1)  # [batch_size, 3, embed_size]

        # Self-attention across the modality tokens
        attn_output, _ = self.attn(sequence, sequence, sequence)  # [batch_size, 3, embed_size]

        # Concatenate attended tokens and fuse
        concat_features = attn_output.reshape(attn_output.shape[0], -1)  # [batch_size, embed_size*3]
        output = self.fusion(concat_features)  # [batch_size, embed_size]

        return output


class GateExpert(nn.Module):
    """Gate expert: fuses three embedding vectors using a gating mechanism.

    Each modality gets a sigmoid gate g = sigmoid(W x) that multiplicatively
    modulates its own features (g * x) before fusion, so the model dynamically
    scales each feature's contribution per sample and dimension. (The previous
    formulation concatenated the raw sigmoid outputs as features — a plain
    sigmoid MLP that never gated anything.)
    """

    def __init__(self, embed_size: int, hidden_size: Optional[int] = None, dropout: float = 0.1):
        """Initialize the gate expert

        Args:
            embed_size: Dimension of the input embedding vectors
            hidden_size: Output dimension of the fusion layer, defaults to embed_size
            dropout: Dropout probability (applied to the gated features)
        """
        super(GateExpert, self).__init__()

        if hidden_size is None:
            hidden_size = embed_size

        # Gates emit per-dimension multipliers in (0, 1); the gate output stays
        # embed_size so g * feature is well-defined.
        self.id_gate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )
        self.txt_gate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )
        self.vis_gate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )

        # Dropout regularizes the gated features AFTER the sigmoid gate:
        # dropping a pre-sigmoid activation silently turns a gate into 0.5.
        self.dropout = nn.Dropout(dropout)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_size * 3, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters"""
        for name, p in self.named_parameters():
            if 'weight' in name:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, id_feat: torch.Tensor, txt_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """Fuse three embedding vectors using gating mechanism

        Args:
            id_feat: ID feature vector, shape [batch_size, embed_size]
            txt_feat: Text feature vector, shape [batch_size, embed_size]
            vis_feat: Visual feature vector, shape [batch_size, embed_size]

        Returns:
            Fused embedding vector, shape [batch_size, hidden_size]
        """
        # Gate values multiplicatively modulate their own modality features
        id_gated = self.id_gate(id_feat) * id_feat       # [batch_size, embed_size]
        txt_gated = self.txt_gate(txt_feat) * txt_feat   # [batch_size, embed_size]
        vis_gated = self.vis_gate(vis_feat) * vis_feat   # [batch_size, embed_size]

        # Concatenate gated features
        gated_features = self.dropout(
            torch.cat([id_gated, txt_gated, vis_gated], dim=1)
        )  # [batch_size, embed_size*3]

        # Fuse
        output = self.fusion(gated_features)  # [batch_size, hidden_size]

        return output


class CrossAttentionExpert(nn.Module):
    """Cross-attention expert: fuses three embedding vectors using cross-attention.

    Each feature vector acts as a query while the other two act as keys and
    values, implementing cross-attention between features to capture more
    complex feature interactions.
    """

    def __init__(self, embed_size: int, num_heads: int = 4, dropout: float = 0.1):
        """Initialize the cross-attention expert

        Args:
            embed_size: Dimension of the input embedding vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossAttentionExpert, self).__init__()

        # Ensure embed_size is divisible by num_heads
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        # Three cross-attention layers, one for each feature acting as query
        self.cross_attn_id = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_txt = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_vis = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),
            nn.LayerNorm(embed_size)
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, id_feat: torch.Tensor, txt_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """Fuse three embedding vectors using cross-attention

        Args:
            id_feat: ID feature vector, shape [batch_size, embed_size]
            txt_feat: Text feature vector, shape [batch_size, embed_size]
            vis_feat: Visual feature vector, shape [batch_size, embed_size]

        Returns:
            Fused embedding vector, shape [batch_size, embed_size]
        """
        # Reshape to sequence form with sequence length 1
        id_seq = id_feat.unsqueeze(1)    # [batch_size, 1, embed_size]
        txt_seq = txt_feat.unsqueeze(1)  # [batch_size, 1, embed_size]
        vis_seq = vis_feat.unsqueeze(1)  # [batch_size, 1, embed_size]

        # Concatenate the other two features as key-value pairs
        kv_for_id = torch.cat([txt_seq, vis_seq], dim=1)   # [batch_size, 2, embed_size]
        kv_for_txt = torch.cat([id_seq, vis_seq], dim=1)   # [batch_size, 2, embed_size]
        kv_for_vis = torch.cat([id_seq, txt_seq], dim=1)   # [batch_size, 2, embed_size]

        # Apply cross-attention
        id_attn_out, _ = self.cross_attn_id(id_seq, kv_for_id, kv_for_id)   # [batch_size, 1, embed_size]
        txt_attn_out, _ = self.cross_attn_txt(txt_seq, kv_for_txt, kv_for_txt)  # [batch_size, 1, embed_size]
        vis_attn_out, _ = self.cross_attn_vis(vis_seq, kv_for_vis, kv_for_vis)  # [batch_size, 1, embed_size]

        # Remove sequence dimension
        id_attn_out = id_attn_out.squeeze(1)   # [batch_size, embed_size]
        txt_attn_out = txt_attn_out.squeeze(1)  # [batch_size, embed_size]
        vis_attn_out = vis_attn_out.squeeze(1)  # [batch_size, embed_size]

        # Concatenate attention outputs
        concat_features = torch.cat([id_attn_out, txt_attn_out, vis_attn_out], dim=1)  # [batch_size, embed_size*3]

        # Fuse
        output = self.fusion(concat_features)  # [batch_size, embed_size]

        return output


def get_expert(expert_type: str, embed_size: int, **kwargs) -> nn.Module:
    """Get an expert module of the specified type

    Args:
        expert_type: Expert type, one of 'sum', 'mlp', 'attention', 'gate', 'cross'
        embed_size: Dimension of the embedding vectors
        **kwargs: Additional keyword arguments passed to the expert module

    Returns:
        Expert module instance
    """
    expert_map = {
        'sum': SumExpert,
        'mlp': MLPExpert,
        'attention': MultiHeadAttentionExpert,
        'gate': GateExpert,
        'cross': CrossAttentionExpert
    }

    if expert_type not in expert_map:
        raise ValueError(f"Unsupported expert type: {expert_type}, available options: {list(expert_map.keys())}")

    expert_class = expert_map[expert_type]

    if expert_type == 'sum':
        return expert_class()
    else:
        return expert_class(embed_size, **kwargs)
