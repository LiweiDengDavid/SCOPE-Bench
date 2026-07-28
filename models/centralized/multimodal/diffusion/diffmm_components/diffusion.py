# coding: utf-8
"""
Diffusion Model Module - DiffMM Core Component
===============================================

Implements the core components of the diffusion model:
- NoiseScheduler: noise scheduling
- DenoisingNetwork: denoising network
- DiffusionProcess: complete diffusion process management

Based on the DDPM theoretical framework, adapted for recommendation system tasks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class NoiseScheduler(nn.Module):
    """Noise scheduler.

    Manages noise scheduling in the diffusion process, supporting multiple strategies:
    - Linear: linear schedule
    - Cosine: cosine schedule (smoother)
    """

    def __init__(self,
                 num_steps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule_type: str = "linear"):
        """Initialize the noise scheduler.

        Args:
            num_steps: number of diffusion steps
            beta_start: starting noise variance
            beta_end: ending noise variance
            schedule_type: schedule type ("linear", "cosine")
        """
        super(NoiseScheduler, self).__init__()

        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        # Compute noise schedule
        if schedule_type == "linear":
            betas = self._linear_schedule()
        elif schedule_type == "cosine":
            betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Compute related parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Precomputed values for the forward (noising) process q_sample.
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _linear_schedule(self) -> torch.Tensor:
        """Linear noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)

    def _cosine_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule (smoother scheduling strategy)."""
        steps = self.num_steps + 1
        x = torch.linspace(0, self.num_steps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / self.num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward process: add noise to x_0 to obtain x_t.

        Args:
            x_start: clean input [batch_size, ...]
            t: timestep [batch_size]
            noise: optional noise; if None, noise is sampled randomly

        Returns:
            x_t: noisy input
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract values from tensor a at timestep t and reshape to be compatible with x_shape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class TimeEmbedding(nn.Module):
    """Timestep embedding module.

    Encodes timestep t into a high-dimensional embedding vector for conditioning the denoising network.
    """

    def __init__(self, embedding_dim: int, max_positions: int = 10000):
        """Initialize time embedding.

        Args:
            embedding_dim: embedding dimension
            max_positions: maximum positional encoding
        """
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions

        # Positional encoding parameters
        half_dim = embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb)

        # MLP layer for further processing of time embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            t: timestep [batch_size]

        Returns:
            time embedding [batch_size, embedding_dim]
        """
        # Sinusoidal positional encoding
        emb = t[:, None].float() * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Append zero if embedding dimension is odd
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return self.time_mlp(emb)


class DenoisingNetwork(nn.Module):
    """Denoising network.

    Takes noisy embedding x_t, timestep t, and conditioning information c as input,
    and outputs the predicted noise. Uses an MLP architecture combined with multimodal conditioning.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [512, 256, 128],
                 condition_dim: int = 128,
                 time_embed_dim: int = 128,
                 dropout: float = 0.1,
                 activation: str = "silu"):
        """Initialize the denoising network.

        Args:
            input_dim: input dimension (embedding dimension)
            hidden_dims: list of hidden layer dimensions
            condition_dim: conditioning information dimension
            time_embed_dim: time embedding dimension
            dropout: dropout rate
            activation: activation function type
        """
        super(DenoisingNetwork, self).__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)

        # Conditioning information projection layer
        self.condition_proj = nn.Linear(condition_dim, time_embed_dim)

        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Backbone network
        layers = []
        total_dim = hidden_dims[0] + time_embed_dim  # input features + time condition

        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            layers.extend([
                nn.Linear(total_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            total_dim = hidden_dim + time_embed_dim  # each layer also incorporates the time condition

        self.layers = nn.ModuleList(layers)

        # Output layer
        self.output_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dims[0]),
            self._get_activation(activation),
            nn.Linear(hidden_dims[0], input_dim)
        )

        # Residual connection weight
        self.residual_proj = nn.Linear(input_dim, input_dim)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function; unknown names fail loud (mirrors DiffRec)."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "swish": nn.SiLU(),  # SiLU is also known as Swish
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        if activation not in activations:
            raise ValueError(
                f"DenoisingNetwork: unsupported activation {activation!r}; "
                f"choose one of {sorted(activations)}"
            )
        return activations[activation]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x_t: noisy input [batch_size, input_dim]
            t: timestep [batch_size]
            condition: conditioning information [batch_size, condition_dim]

        Returns:
            predicted noise [batch_size, input_dim]
        """
        # Time embedding
        time_emb = self.time_embed(t)

        # Process conditioning information
        if condition is not None:
            condition_emb = self.condition_proj(condition)
            time_emb = time_emb + condition_emb

        # Input projection
        h = self.input_proj(x_t)

        # Process layer by layer, incorporating time condition at each layer.
        # Layers are always exact triples (Linear, Activation, Dropout-or-Identity).
        for i in range(0, len(self.layers), 3):
            # Incorporate time condition
            h_with_time = torch.cat([h, time_emb], dim=-1)
            h = self.layers[i + 2](self.layers[i + 1](self.layers[i](h_with_time)))

        # Final incorporation of time condition and output
        h_final = torch.cat([h, time_emb], dim=-1)
        output = self.output_proj(h_final)

        # Optional residual connection
        residual = self.residual_proj(x_t)
        return output + residual


class DiffusionProcess(nn.Module):
    """Complete diffusion process manager.

    Manages the forward process (adding noise) and the reverse process (sampling) in a unified way.
    """

    def __init__(self,
                 denoising_network: DenoisingNetwork,
                 noise_scheduler: NoiseScheduler):
        """Initialize the diffusion process.

        Args:
            denoising_network: denoising network
            noise_scheduler: noise scheduler
        """
        super(DiffusionProcess, self).__init__()

        self.denoising_network = denoising_network
        self.noise_scheduler = noise_scheduler

    def training_losses(self, x_start: torch.Tensor,
                       condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute training losses.

        Args:
            x_start: clean input [batch_size, input_dim]
            condition: conditioning information [batch_size, condition_dim]

        Returns:
            Dictionary containing various losses
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Randomly sample timesteps
        t = torch.randint(0, self.noise_scheduler.num_steps, (batch_size,), device=device).long()

        # Add noise
        noise = torch.randn_like(x_start)
        x_t = self.noise_scheduler.q_sample(x_start, t, noise)

        # Predict noise
        predicted_noise = self.denoising_network(x_t, t, condition)

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')

        return {
            'diffusion_loss': loss,
            'predicted_noise': predicted_noise,
            'true_noise': noise,
            'timesteps': t
        }


# Convenience function
def create_diffusion_process(config: Dict[str, Any], input_dim: int,
                           condition_dim: int) -> DiffusionProcess:
    """Create a DiffusionProcess instance.

    Args:
        config: configuration dictionary
        input_dim: input dimension
        condition_dim: conditioning dimension

    Returns:
        DiffusionProcess instance
    """
    # Get diffusion configuration
    diffusion_config = config['diffusion']

    # Create noise scheduler
    noise_scheduler = NoiseScheduler(
        num_steps=diffusion_config['num_steps'],
        beta_start=diffusion_config['noise_min'],
        beta_end=diffusion_config['noise_max'],
        schedule_type=diffusion_config['beta_schedule']
    )

    # Create denoising network
    denoising_network = DenoisingNetwork(
        input_dim=input_dim,
        hidden_dims=diffusion_config['hidden_dims'],
        condition_dim=condition_dim,
        time_embed_dim=diffusion_config['time_embed_dim'],
        dropout=config['dropout_rate']
    )

    # Create complete diffusion process
    diffusion_process = DiffusionProcess(
        denoising_network=denoising_network,
        noise_scheduler=noise_scheduler
    )

    return diffusion_process
