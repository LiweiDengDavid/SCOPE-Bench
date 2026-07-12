"""
Diffusion utility functions for CFDiff
"""

import math
import numpy as np
import torch


def get_named_beta_schedule(
    schedule_name,
    num_diffusion_timesteps,
    *,
    noise_scale=None,
    noise_min=None,
    noise_max=None,
):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == "linear":
        # CF-Diff / DiffRec small-noise forward process: betas =
        # noise_scale * linspace(noise_min, noise_max), keeping alphas_cumprod
        # near 1 so the interaction signal remains present during noising.
        if noise_scale is None or noise_min is None or noise_max is None:
            raise ValueError(
                "'linear' schedule requires noise_scale, noise_min, noise_max"
            )
        return np.linspace(
            noise_scale * noise_min,
            noise_scale * noise_max,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    elif schedule_name == "cosine":
        def alpha_bar_cosine(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        return betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar_cosine)
    elif schedule_name == "binomial":
        # Binomial beta schedule
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            # Linear interpolation between t1 and t2
            alpha_bar_t1 = (1 - t1) ** 2  
            alpha_bar_t2 = (1 - t2) ** 2
            if alpha_bar_t1 > 1e-8:  # Avoid division by zero
                beta = 1 - alpha_bar_t2 / alpha_bar_t1
            else:
                beta = 0.999
            betas.append(np.clip(beta, 1e-6, 0.9999))
        return np.array(betas)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        alpha_bar_t1 = alpha_bar(t1)
        alpha_bar_t2 = alpha_bar(t2)
        if alpha_bar_t1 > 1e-8:
            beta = min(1 - alpha_bar_t2 / alpha_bar_t1, max_beta)
        else:
            beta = max_beta
        betas.append(max(beta, 1e-6))
    return np.array(betas)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Canonical definition shared by the CAM_AE denoisers.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
