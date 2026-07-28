"""
CFDiff: Collaborative Filtering based on Diffusion Models
Original paper: "Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity"
SIGIR 2024
"""

import torch
from core.base import RecommenderBase
from .cfdiff_components.gaussian_diffusion import GaussianDiffusion, ModelMeanType
from .cfdiff_components.cam_ae import CAM_AE
from .cfdiff_components.cam_ae_multihops import CAM_AE_multihops
from .cfdiff_components.diffusion_utils import get_named_beta_schedule


class CFDiff(RecommenderBase):
    """
    CFDiff model implementation for FedVLR framework
    
    Based on: Collaborative Filtering Based on Diffusion Models: 
              Unveiling the Potential of High-Order Connectivity (SIGIR 2024)
    """
    
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.diffusion_steps = config["diffusion_steps"]
        self.noise_schedule = config["noise_schedule"]
        # Small-noise forward-process parameters (DiffRec/CF-Diff design): betas =
        # noise_scale * linspace(noise_min, noise_max). Required for the 'linear'
        # schedule so the diffusion retains the personalized signal.
        self.noise_scale = config["noise_scale"]
        self.noise_min = config["noise_min"]
        self.noise_max = config["noise_max"]
        self.model_mean_type = config["model_mean_type"]
        # Deterministic reverse process at inference (default). When false, the
        # DDPM reverse loop draws no init/per-step Gaussian noise, so eval is
        # reproducible and independent of global RNG state. Mirrors DiffRec.
        self.sampling_noise = bool(config["sampling_noise"])
        self.d_model = config["d_model"]
        self.num_heads = config["num_attention_heads"]
        self.num_layers = config["num_layers"]
        self.max_hops = config["max_hops"]
        self.hop_fusion_weights = config["hop_fusion_weights"]
        self.cross_fusion_weight = config["cross_fusion_weight"]
        self._align_item_dim(dataloader)
        self._setup_diffusion()
        self._setup_model()
        self._setup_multihop_processor(dataloader)

    @staticmethod
    def _resolve_dataset(dataloader):
        """Accept either a dataloader wrapper or a raw dataset."""
        return getattr(dataloader, 'dataset', dataloader)

    def _align_item_dim(self, dataloader):
        """Keep the diffusion input size aligned with the actual dataset."""
        dataset = self._resolve_dataset(dataloader)
        if hasattr(dataset, 'get_item_num'):
            self.in_dims = dataset.get_item_num()
            self.n_items = self.in_dims
            return
        self.in_dims = self.n_items

    def _setup_diffusion(self):
        """Build the Gaussian diffusion process."""
        betas = get_named_beta_schedule(
            self.noise_schedule,
            self.diffusion_steps,
            noise_scale=self.noise_scale,
            noise_min=self.noise_min,
            noise_max=self.noise_max,
        )
        if self.model_mean_type == "epsilon":
            model_mean_type = ModelMeanType.EPSILON
        elif self.model_mean_type == "start_x":
            model_mean_type = ModelMeanType.START_X
        else:
            raise ValueError(
                f"Unknown model_mean_type '{self.model_mean_type}': expected 'epsilon' or 'start_x'"
            )
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            loss_type="mse"
        )

    def _setup_model(self):
        """Build the denoiser backbone."""
        common = dict(
            in_dims=self.in_dims,
            emb_size=self.d_model,
            norm=True,
            dropout=self.dropout_rate,
            d_model=self.d_model,
            n_heads=self.num_heads,
            n_layers=self.num_layers,
        )
        if self.max_hops > 2:
            self.model = CAM_AE_multihops(**common, hop_fusion_weights=self.hop_fusion_weights)
        else:
            self.model = CAM_AE(**common, cross_fusion_weight=self.cross_fusion_weight)

    def _setup_multihop_processor(self, dataloader):
        """Build the multi-hop processor."""
        from .cfdiff_components.multihop_processor import MultiHopProcessor

        dataset = self._resolve_dataset(dataloader)
        self.multihop_processor = MultiHopProcessor(
            dataset=dataset,
            max_hops=self.max_hops,
            cache_enabled=self.config["cache_multihop"],
        )

    def _get_multihop_features(self, users, items, device):
        """Skip multi-hop lookup when the model runs in one-hop mode."""
        if self.max_hops <= 1:
            return None
        return self.multihop_processor.get_multihop_features(users, items, device=device)

    @staticmethod
    def _build_model_kwargs(multihop_features):
        """Attach optional multi-hop conditioning for diffusion calls."""
        if multihop_features is None:
            return {}
        return {"x_sec_hop": multihop_features}
        
    def forward(self, users, items):
        """Compute training-time scores for one user-item batch."""
        batch_size = users.size(0)
        device = users.device

        if not self.training:
            raise RuntimeError(
                "CFDiff.forward() must not be called in eval mode; "
                "use full_sort_predict() for ranking."
            )

        history = self._build_interaction_vectors(users)
        multihop_features = self._get_multihop_features(users, items, device)
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
        if multihop_features is not None:
            predictions = self.model(history, t, x_sec_hop=multihop_features)
        else:
            predictions = self.model(history, t)
        return predictions.gather(1, items.unsqueeze(1)).squeeze(1)

    def calculate_loss(self, interaction):
        """Compute the diffusion training loss for one batch."""
        users = interaction[0]
        pos_items = interaction[1]
        batch_size = users.size(0)
        device = users.device
        x_start = self._build_interaction_vectors(users)
        multihop_features = self._get_multihop_features(users, pos_items, device)
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
        loss_dict = self.diffusion.training_losses(
            model=self.model,
            x_start=x_start,
            t=t,
            model_kwargs=self._build_model_kwargs(multihop_features),
        )
        return loss_dict["loss"].mean()

    def _build_interaction_vectors(self, users):
        """Encode each user's history as a dense interaction vector."""
        batch_size = users.size(0)
        device = users.device
        vectors = torch.zeros(batch_size, self.n_items, device=device)
        batch_indices = []
        item_indices = []

        for i, user in enumerate(users):
            user_interactions = self.multihop_processor.get_user_interactions(user.item())
            if len(user_interactions) > 0:
                batch_indices.extend([i] * len(user_interactions))
                item_indices.extend(user_interactions)

        if len(batch_indices) > 0:
            batch_idx_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
            item_idx_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)
            vectors[batch_idx_tensor, item_idx_tensor] = 1.0

        return vectors
    
    def predict_all(self, users):
        """Generate one full ranking distribution per user.

        Seed the reverse chain from the user's corrupted interaction-history vector
        (x_T = q_sample(history, diffusion_steps - 1)), where history is the dense
        per-user interaction vector built by _build_interaction_vectors. This mirrors
        DiffRec and the reference CF_Diff: starting the reverse process from a
        zero/Gaussian latent instead discards the interaction-history signal the
        denoiser was trained to reconstruct (training corrupts q_sample(history, t)),
        leaving rankings un-personalized.
        """
        device = users.device
        batch_size = users.size(0)

        with torch.no_grad():
            history = self._build_interaction_vectors(users)
            dummy_items = torch.zeros(batch_size, dtype=torch.long, device=device)
            multihop_features = self._get_multihop_features(users, dummy_items, device)
            shape = (batch_size, self.n_items)
            # Deterministic inference by default: corrupt history with ZERO noise so
            # x_T is reproducible; only draw random init noise when sampling_noise.
            t_init = torch.full(
                (batch_size,), self.diffusion_steps - 1, device=device, dtype=torch.long
            )
            init_noise = None if self.sampling_noise else torch.zeros_like(history)
            x_T = self.diffusion.q_sample(history, t_init, noise=init_noise)
            u_0 = self.diffusion.p_sample_loop(
                model=self.model,
                shape=shape,
                noise=x_T,
                clip_denoised=False,
                model_kwargs=self._build_model_kwargs(multihop_features),
                device=device,
                progress=False,
                sampling_noise=self.sampling_noise,
            )

        return u_0

    def full_sort_predict(self, interaction):
        """Return scores for all items for all users in the batch.

        Ranking is the only eval entry point for CFDiff: it delegates straight to
        predict_all(), which runs the DDPM denoising loop. This avoids the base
        class fallback that scores via predict()->forward() (forward() raises in
        eval mode).

        Args:
            interaction: list of tensors; interaction[0] is user indices [batch_size]

        Returns:
            scores: [batch_size, n_items]
        """
        users = interaction[0]
        return self.predict_all(users)
