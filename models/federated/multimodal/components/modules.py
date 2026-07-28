import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import SumExpert, MLPExpert, GateExpert, get_expert


class GatingNetwork(nn.Module):
    """Gating network: assigns weights for a mixture-of-experts system.

    Computes per-expert weights based on the input features, implementing
    a dynamic routing mechanism.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, latent_dim: int = 128):
        """Initialize the gating network

        Args:
            in_dim: Input feature dimension
            out_dim: Output dimension (number of experts)
            dropout: Dropout probability
            latent_dim: Hidden layer dimension
        """
        super(GatingNetwork, self).__init__()

        # Two-layer feed-forward network
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters"""
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expert weights

        Args:
            x: Input features, shape [batch_size, in_dim]

        Returns:
            Expert weights, shape [batch_size, out_dim] (per-sample routing)
        """
        # Forward pass
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)

        # Softmax to ensure weights sum to 1, PER SAMPLE. Returning per-sample
        # routing weights [batch, out_dim] keeps each item's expert mixture
        # independent; collapsing to a batch-mean vector made every sample share
        # one mixture and coupled samples within a batch (train/eval inconsistent).
        weights = F.softmax(out, dim=1)
        return weights


class SwitchingFusionModule(nn.Module):
    """Switching fusion module: dynamically selects the most suitable expert.

    Combines multiple expert models and uses a gating network to dynamically
    assign weights, implementing a mixture-of-experts system.
    """

    def __init__(self, in_dim: int, embed_dim: int, dropout: float = 0.1,
                 latent_dim: int = 128):
        """Initialize the switching fusion module

        Args:
            in_dim: Input feature dimension
            embed_dim: Embedding vector dimension
            dropout: Dropout probability
            latent_dim: Hidden layer dimension
        """
        super(SwitchingFusionModule, self).__init__()

        # Gating network for assigning expert weights (server-side routing)
        self.router = GatingNetwork(embed_dim * 3, 3, dropout, latent_dim)

        # List of expert modules (dropout follows config, not the hardcoded default)
        self.experts = nn.ModuleList([
            SumExpert(),                                       # Sum expert
            MLPExpert(embed_dim, dropout=dropout),             # MLP expert
            GateExpert(embed_dim, embed_dim, dropout=dropout)  # Gate expert
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Fuse three feature vectors

        Args:
            x: First feature vector, shape [batch_size, embed_dim]
            y: Second feature vector, shape [batch_size, embed_dim]
            z: Third feature vector, shape [batch_size, embed_dim]

        Returns:
            [batch_size, embed_dim]
        """
        # Feed input to each expert module
        expert_outputs = [expert(x, y, z) for expert in self.experts]

        # Concatenate all expert outputs
        combined_output = torch.cat(expert_outputs, dim=1)  # [batch_size, embed_dim*3]

        # Compute per-sample expert weights using router
        weights = self.router(combined_output)  # [batch_size, num_experts]

        # Per-sample weighted combination of expert outputs
        output = torch.zeros_like(x)
        for i, expert_output in enumerate(expert_outputs):
            output += weights[:, i:i + 1] * expert_output

        return output


class FusionLayer(nn.Module):
    """Fusion layer: fuses multimodal features into a unified representation.

    First maps each modality's features to the same latent space, then
    applies the specified fusion module to merge them.
    """

    def __init__(self, in_dim: int, fusion_module: str = 'moe', latent_dim: int = 128,
                 project_id: bool = True, dropout: float = 0.1,
                 visual_dim: int = None):
        """Initialize the fusion layer

        Args:
            in_dim: Input feature dimension (text/ID modality)
            fusion_module: Fusion module type, one of 'moe', 'sum', 'mlp', 'attention', 'gate', 'cross'
            latent_dim: Latent space dimension
            project_id: If True (default), apply a linear projection to ID features.
                If False, ID features are assumed to already be in latent_dim and only
                LayerNorm is applied.
            visual_dim: Visual feature dimension. The text and visual projections share a
                single ``in_dim``, so this must equal ``in_dim``; pass it to fail fast
                with a clear error on asymmetric encoders.
        """
        super(FusionLayer, self).__init__()

        if visual_dim is not None and visual_dim != in_dim:
            raise ValueError(
                f"FusionLayer requires visual_dim == text/in_dim (shared projection), "
                f"got visual_dim={visual_dim} vs in_dim={in_dim}."
            )

        self.project_id = project_id

        # Feature mapping layers to project each modality to a common dimension
        if project_id:
            self.id_affine = nn.Linear(in_dim, latent_dim)
        self.txt_affine = nn.Linear(in_dim, latent_dim)
        self.vis_affine = nn.Linear(in_dim, latent_dim)

        # Feature normalization layers
        self.id_norm = nn.LayerNorm(latent_dim)
        self.txt_norm = nn.LayerNorm(latent_dim)
        self.vis_norm = nn.LayerNorm(latent_dim)

        # Select fusion module by type
        if fusion_module == 'moe':
            self.fusion = SwitchingFusionModule(
                latent_dim, latent_dim, dropout=dropout,
                latent_dim=latent_dim,
            )
        elif fusion_module in ['sum', 'mlp', 'attention', 'gate', 'cross']:
            self.fusion = get_expert(fusion_module, latent_dim, dropout=dropout)
        else:
            raise ValueError(f'Invalid fusion module: {fusion_module}, currently support: '
                            f'moe, sum, mlp, attention, gate, cross')

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize Linear modules only; LayerNorms keep their PyTorch defaults.

        The previous catch-all over named_parameters() normal-initialized every
        1-D 'weight' tensor — i.e. exactly the LayerNorm gammas — to N(0, 0.01),
        collapsing the fused representation ~100x at init (the hosts' later
        apply(xavier_normal_initialization) touches only Embedding/Linear and
        does not repair them).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, id_feat: torch.Tensor, txt_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """Fuse multimodal features

        Args:
            id_feat: ID features, shape [batch_size, in_dim]
            txt_feat: Text features, shape [batch_size, in_dim]
            vis_feat: Visual features, shape [batch_size, in_dim]

        Returns:
            Fused features, shape [batch_size, latent_dim]
        """
        # Feature mapping and normalization
        if self.project_id:
            id_feat = self.id_norm(self.id_affine(id_feat))
        else:
            id_feat = self.id_norm(id_feat)
        txt_feat = self.txt_norm(self.txt_affine(txt_feat))
        vis_feat = self.vis_norm(self.vis_affine(vis_feat))

        # Feature fusion
        return self.fusion(id_feat, txt_feat, vis_feat)
