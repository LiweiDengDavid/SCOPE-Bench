# coding: utf-8
"""
DiffRec (Diffusion Recommendation) - simplified DDPM baseline
=============================================================

A simplified DiffRec-style diffusion baseline for NexusRec. It follows the
RecBole DiffRec structure (small-noise forward process + MLP denoiser + reverse
sampling for ranking) but uses a simplified DDPM training objective: uniform
timesteps and a plain unweighted MSE. It does NOT port RecBole's SNR/importance
sampling or VLB/rescaled loss variants.

Reference:
    Wang, W., Yao, Y., Chen, X., et al. (2023).
    "DiffRec: A Diffusion Model for Sequential Recommendation."
    In SIGIR 2023.

RecBole Reference Implementation:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/diffrec.py
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from core.base import RecommenderBase

logger = logging.getLogger("nexusrec")

# Removed unnecessary imports


class MLPLayers(nn.Module):
    """RecBole standard MLP layer implementation - aligned with the RecBole architecture.

    Standard multi-layer perceptron used for the DiffRec denoising network.
    Time information is incorporated via concatenation, which is simple and efficient.
    """

    _ACT = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(self, layers, dropout=0.0, activation="tanh"):
        super().__init__()
        self.layers = nn.ModuleList()
        if activation not in self._ACT:
            raise ValueError(
                f"DiffRec MLPLayers: unsupported activation {activation!r}; "
                f"choose one of {sorted(self._ACT)}"
            )
        self.activation = self._ACT[activation]()
        self.dropout = nn.Dropout(dropout)

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x


class DiffRec(RecommenderBase):
    """DiffRec baseline model - simplified DDPM objective.

    Follows the RecBole DiffRec structure:
    - Forward diffusion: gradually add small noise to user interactions
    - Reverse diffusion: MLP denoising network progressively restores clean interactions
    - Inference sampling: reverse-denoise the corrupted interaction vector to rank items
    - Small-noise design: key innovation for retaining personalized information

    Training uses a simplified DDPM loss (uniform timesteps, unweighted MSE); it
    does NOT port RecBole's SNR/importance sampling or VLB loss variants. Serves
    as the diffusion model baseline in the NexusRec framework.
    """

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        # DiffRec core parameters - aligned with RecBole defaults
        # Architecture is generated from num_layers + hidden_dim (both YAML-driven).
        self.min_hidden_dim = int(config["min_hidden_dim"])
        self.dims_dnn = self._generate_dims_dnn(config["num_layers"], config["hidden_dim"])

        # Diffusion process parameters - ensure numeric type conversion
        self.steps = int(config["steps"])
        self.noise_scale = float(config["noise_scale"])
        self.noise_min = float(config["noise_min"])
        self.noise_max = float(config["noise_max"])
        self.sampling_steps = int(config["sampling_steps"])
        # Deterministic inference by default (RecBole default): when False, reverse
        # sampling injects no stochastic noise, so validation/test metrics are
        # reproducible. When True, Gaussian noise is added at each reverse step.
        self.sampling_noise = bool(config["sampling_noise"])

        # Prediction and training parameters
        self.mean_type = config["mean_type"]

        # Network architecture parameters
        self.activation = config["activation"]
        self.dropout_prob = config["dropout_rate"]

        # Pre-compute user-item interaction matrix (same pattern as LightGCN)
        inter_coo = dataloader.inter_matrix(form="coo").astype(np.float32)
        inter_dense = torch.from_numpy(inter_coo.toarray())  # [n_users, n_items]
        self.register_buffer("user_interaction_matrix", inter_dense)

        # Build denoising network - using RecBole standard architecture
        # Input dimension: n_items + 1 (timestep information)
        # Network structure: [n_items+1] -> dims_dnn -> [n_items]
        input_dim = self.n_items + 1
        output_dim = self.n_items

        # Build complete network structure: input -> num_layers-generated stack ->
        # output. An empty dims_dnn collapses to [input_dim, output_dim], so no
        # separate branch is needed.
        size = [input_dim] + self.dims_dnn + [output_dim]

        self.mlp = MLPLayers(size, self.dropout_prob, self.activation)

        # Diffusion schedule + posterior coefficients computed in FLOAT64, then
        # registered as float32 buffers. Critical: betas ~ 1e-9..5e-7 (noise_scale
        # 1e-4), so cumprod(1-betas) is ~ 1 - 1e-9 for small t; computing
        # (1 - alphas_cumprod) in float32 rounds that to EXACTLY 0 (float32 has
        # ~1e-7 relative precision near 1.0), making p_sample/q_sample divide by
        # sqrt(0) and produce NaN scores. Precomputing the derived terms in float64 keeps
        # (1 - alphas_cumprod) ~ 1e-9 (finite) and stores it without the lossy
        # near-1 subtraction.
        betas = torch.linspace(
            self.noise_min, self.noise_max, self.steps, dtype=torch.float64
        ) * self.noise_scale
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]]
        )
        one_minus_acp = (1.0 - alphas_cumprod).clamp_min(1e-12)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        # q_sample coefficients (sqrt over float64-derived terms, so small-t no
        # longer underflows the (1-acp) factor to 0).
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(one_minus_acp).float()
        )
        # eps->x0 recovery coefficients.
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float()
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0).float()
        )
        # Reverse-step posterior coefficients (never re-derived in float32).
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / one_minus_acp).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / one_minus_acp).float(),
        )
        self.register_buffer(
            "posterior_variance",
            (betas * (1.0 - alphas_cumprod_prev) / one_minus_acp).float(),
        )

        # Set inference step count
        if self.sampling_steps <= 0:
            self.sampling_steps = self.steps

    def _generate_dims_dnn(self, num_layers, hidden_dim):
        """Generate MLP architecture from number of layers and first-layer dimension.

        Args:
            num_layers (int): number of hidden layers
            hidden_dim (int): first hidden layer dimension

        Returns:
            list: list of hidden layer dimensions; subsequent layers are halved in size
        """
        dims = []
        current_dim = hidden_dim

        for i in range(num_layers):
            dims.append(current_dim)
            # Halve subsequent layer dimensions, floored by config min_hidden_dim
            current_dim = max(current_dim // 2, self.min_hidden_dim)

        return dims

    def _get_user_interaction_vector(self, user_ids):
        """Look up pre-computed user interaction vectors.

        Returns:
            Tensor of shape [batch_size, n_items] with 1.0 for interacted items.
        """
        return self.user_interaction_matrix[user_ids]

    def forward(self, user_interactions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Diffusion model forward pass - RecBole standard implementation.

        Args:
            user_interactions: user interaction matrix [batch_size, n_items]
            t: timestep [batch_size]

        Returns:
            Predicted noise or x_0
        """
        # Normalize time information to [0, 1]
        t_normalized = t.float() / self.steps
        t_expanded = t_normalized.unsqueeze(1)  # [batch_size, 1]

        # Concatenate time information with the interaction vector
        x_with_t = torch.cat(
            [user_interactions, t_expanded], dim=1
        )  # [batch_size, n_items+1]

        # Pass through the MLP network
        return self.mlp(x_with_t)

    def calculate_loss(self, interaction):
        """Calculate the diffusion training loss (simplified DDPM objective).

        This is NOT a full RecBole port: timesteps are sampled uniformly and the
        loss is a plain unweighted MSE between the model output and the target
        (x_0 or eps). It does NOT implement SNR/importance-sampling reweighting
        or the VLB/rescaled loss variants from the RecBole DiffRec.
        """
        # Parse interaction format
        if isinstance(interaction, dict):
            user_ids = interaction[self.USER_ID]
        else:
            user_ids = interaction[0]  # NexusRec tensor format

        batch_size = len(user_ids)
        device = self.device

        # Get user rating matrix (corresponds to x_start in RecBole)
        x_start = self._get_user_interaction_vector(user_ids)

        # Sample timesteps uniformly (no importance/SNR reweighting)
        ts = self._sample_timesteps(batch_size, device)

        # Generate noise
        noise = torch.randn_like(x_start)

        # Add noise
        if self.noise_scale != 0.0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        # Model prediction
        model_output = self.forward(x_t, ts)

        # Target selection
        if self.mean_type == "x0":
            target = x_start
        else:  # mean_type == 'eps'
            target = noise

        # Ensure shape consistency
        assert (
            model_output.shape == target.shape == x_start.shape
        ), f"Shape mismatch: model_output {model_output.shape}, target {target.shape}, x_start {x_start.shape}"

        # Compute MSE loss (averaged over the batch dimension)
        mse = (target - model_output) ** 2
        mse = mse.view(mse.shape[0], -1).mean(dim=1)
        return mse.mean()

    def _sample_timesteps(self, batch_size, device):
        """Sample timesteps - standard uniform sampling."""
        ts = torch.randint(
            0, self.steps, (batch_size,), device=device, dtype=torch.long
        )
        return ts

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process - add noise (uses float64-derived buffers)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_acp = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_acp = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_acp * x_start + sqrt_one_minus_acp * noise

    def p_sample(self, x_t, t):
        """Single reverse diffusion step (posterior coefficients precomputed in
        float64; no float32 near-1 subtraction here)."""
        model_output = self.forward(x_t, t)

        if self.mean_type == "x0":
            x_start_pred = model_output
        else:
            # eps prediction -> recover x_start
            x_start_pred = (
                self.sqrt_recip_alphas_cumprod[t].unsqueeze(-1) * x_t
                - self.sqrt_recipm1_alphas_cumprod[t].unsqueeze(-1) * model_output
            )

        # For t == 0, return the prediction directly (no noise added)
        if t[0].item() == 0:
            return x_start_pred

        # Posterior mean from precomputed coefficients
        mean = (
            self.posterior_mean_coef1[t].unsqueeze(-1) * x_start_pred
            + self.posterior_mean_coef2[t].unsqueeze(-1) * x_t
        )

        # Deterministic by default (sampling_noise=false) for reproducible eval;
        # only inject Gaussian noise when explicitly enabled.
        if not self.sampling_noise:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(self.posterior_variance[t].unsqueeze(-1)) * noise

    def p_sample_loop(self, x_start):
        """Full reverse diffusion chain - mirrors RecBole DiffRec.p_sample.

        The reverse trajectory always walks the FULL reversed schedule
        ``reversed(range(self.steps))``. Inference truncation is controlled by
        the *initial* noise level: x_T is the (corrupted) interaction vector
        ``q_sample(x_start, sampling_steps - 1)`` (RecBole), NOT pure Gaussian
        noise and NOT a shortened loop. Initializing from randn / iterating only
        the lowest ``sampling_steps`` timesteps would feed the model inputs whose
        noise level does not match the schedule it denoises over.
        """
        batch_size = x_start.shape[0]
        # sampling_steps is normalized to >= 1 in __init__ (0 -> self.steps), so
        # x_T is always the interaction vector corrupted to t = sampling_steps-1.
        t_init = torch.full(
            (batch_size,), self.sampling_steps - 1, device=self.device, dtype=torch.long
        )
        # Deterministic inference by default: corrupt x_start with ZERO noise so the
        # initial x_T is reproducible; only draw random init noise when sampling_noise.
        init_noise = None if self.sampling_noise else torch.zeros_like(x_start)
        x_t = self.q_sample(x_start, t_init, noise=init_noise)
        for step in reversed(range(self.steps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t)
        return x_t

    def predict(self, interaction):
        """Predict scores using iterative reverse diffusion."""
        if isinstance(interaction, dict):
            user_ids = interaction[self.USER_ID]
        else:
            user_ids = interaction[0]

        user_interactions = self._get_user_interaction_vector(user_ids)

        with torch.no_grad():
            scores = self.p_sample_loop(user_interactions)

        # If a specific item is specified, return the corresponding score
        if isinstance(interaction, dict) and self.ITEM_ID in interaction:
            item_ids = interaction[self.ITEM_ID]
            batch_size = len(user_ids)
            return scores[torch.arange(batch_size, device=scores.device), item_ids]
        elif not isinstance(interaction, dict) and len(interaction) > 1:
            item_ids = interaction[1]
            batch_size = len(user_ids)
            return scores[torch.arange(batch_size, device=scores.device), item_ids]
        else:
            return scores.mean(dim=-1)

    def full_sort_predict(self, interaction):
        """Full-sort prediction using iterative reverse diffusion."""
        if isinstance(interaction, dict):
            user_ids = interaction[self.USER_ID]
        else:
            user_ids = interaction[0]

        user_interactions = self._get_user_interaction_vector(user_ids)

        with torch.no_grad():
            recommendations = self.p_sample_loop(user_interactions)

        # History masking is owned by the evaluator, which overwrites seen-item
        # positions with -inf (core/base/trainer.py); masking here would only
        # corrupt the saved score artifact for an already-suppressed ranking.
        return recommendations
