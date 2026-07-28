# coding: utf-8
"""
Multimodal Utility Functions
============================
"""

from __future__ import annotations

import torch


def resolve_multimodal_ablation(config):
    """Resolve canonical multimodal ablation settings from runtime config."""
    ablation = config["multimodal_ablation"]
    if not isinstance(ablation, dict):
        raise TypeError("Config key 'multimodal_ablation' must be a dict.")

    def normalize_mode(mode):
        if isinstance(mode, str) and mode.lower() == "none":
            return None
        return mode

    txt_mode = normalize_mode(ablation["text"])
    vis_mode = normalize_mode(ablation["visual"])
    id_mode = normalize_mode(ablation["id"])

    valid_modes = [None, "remove", "noise"]
    for name, mode in [
        ("multimodal_ablation.text", txt_mode),
        ("multimodal_ablation.visual", vis_mode),
        ("multimodal_ablation.id", id_mode),
    ]:
        if mode not in valid_modes:
            raise ValueError(f"Invalid {name}: {mode!r}, supported values are: {valid_modes}")

    valid_noise_types = ["gaussian", "uniform"]
    for name, noise_type in [
        ("multimodal_ablation.text_noise_type", ablation["text_noise_type"]),
        ("multimodal_ablation.visual_noise_type", ablation["visual_noise_type"]),
        ("multimodal_ablation.id_noise_type", ablation["id_noise_type"]),
    ]:
        if noise_type not in valid_noise_types:
            raise ValueError(
                f"Invalid {name}: {noise_type!r}, supported values are: {valid_noise_types}"
            )

    # Noise parameters are required when the corresponding mode is "noise".
    # Experiment configs must provide perturbation values explicitly.
    if txt_mode == "noise":
        for key in ("text_noise_scale", "text_noise_type"):
            if key not in ablation:
                raise KeyError(
                    f"multimodal_ablation.{key} is required when text mode is 'noise'"
                )
    if vis_mode == "noise":
        for key in ("visual_noise_scale", "visual_noise_type"):
            if key not in ablation:
                raise KeyError(
                    f"multimodal_ablation.{key} is required when visual mode is 'noise'"
                )
    if id_mode == "noise":
        for key in ("id_noise_scale", "id_noise_type"):
            if key not in ablation:
                raise KeyError(
                    f"multimodal_ablation.{key} is required when id mode is 'noise'"
                )

    return {
        "txt_mode": txt_mode,
        "vis_mode": vis_mode,
        "id_mode": id_mode,
        "txt_noise_scale": ablation["text_noise_scale"],
        "vis_noise_scale": ablation["visual_noise_scale"],
        "id_noise_scale": ablation["id_noise_scale"],
        "txt_noise_type": ablation["text_noise_type"],
        "vis_noise_type": ablation["visual_noise_type"],
        "id_noise_type": ablation["id_noise_type"],
        # Drives the dedicated noise generator so the "noise" ablation is a
        # run-stable perturbation independent of the global RNG state.
        "seed": config["seed"],
    }


def modal_ablation(
    item_embed,
    txt_embed,
    vision_embed,
    *,
    seed,
    txt_mode=None,
    vis_mode=None,
    id_mode=None,
    txt_noise_scale=1.0,
    vis_noise_scale=1.0,
    id_noise_scale=1.0,
    txt_noise_type="gaussian",
    vis_noise_type="gaussian",
    id_noise_type="gaussian",
    device=None,
):
    """Apply modal ablations or perturbations for multimodal studies.

    The "noise" ablation is a run-stable perturbation: a dedicated
    ``torch.Generator`` is re-seeded from ``seed`` on every call, so each modality's
    noise is deterministic across forwards (training and eval) and independent of
    the global RNG state.
    """

    def normalize_mode(mode):
        if isinstance(mode, str) and mode.lower() == "none":
            return None
        return mode

    txt_mode = normalize_mode(txt_mode)
    vis_mode = normalize_mode(vis_mode)
    id_mode = normalize_mode(id_mode)

    valid_modes = [None, "remove", "noise"]
    valid_noise_types = ["gaussian", "uniform"]

    for name, mode in [
        ("txt_mode", txt_mode),
        ("vis_mode", vis_mode),
        ("id_mode", id_mode),
    ]:
        if mode not in valid_modes:
            raise ValueError(f"Invalid {name}: {mode!r}, supported values are: {valid_modes}")

    for name, noise_type in [
        ("txt_noise_type", txt_noise_type),
        ("vis_noise_type", vis_noise_type),
        ("id_noise_type", id_noise_type),
    ]:
        if noise_type not in valid_noise_types:
            raise ValueError(
                f"Invalid {name}: {noise_type!r}, supported values are: {valid_noise_types}"
            )

    if device is None:
        device = item_embed.device
    dtype = item_embed.dtype

    # Dedicated, freshly-seeded generator: the noise is a deterministic function of
    # (seed, shape, scale, type) and the per-modality draw order below, so it is
    # identical run-to-run and call-to-call without disturbing the global RNG.
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    def generate_noise(tensor, noise_type, noise_scale):
        shape = tensor.shape
        if noise_type == "gaussian":
            return torch.randn(shape, generator=generator, device=device, dtype=dtype) * noise_scale
        return (torch.rand(shape, generator=generator, device=device, dtype=dtype) * 2 - 1) * noise_scale

    processed_id = item_embed
    if id_mode == "remove":
        # Zero the ID embedding, consistent with text/visual "remove".
        processed_id = torch.zeros_like(item_embed, dtype=dtype)
    elif id_mode == "noise":
        processed_id = generate_noise(item_embed, id_noise_type, id_noise_scale)

    processed_txt = txt_embed
    if txt_mode == "remove":
        processed_txt = torch.zeros_like(txt_embed, dtype=dtype)
    elif txt_mode == "noise":
        processed_txt = generate_noise(txt_embed, txt_noise_type, txt_noise_scale)

    processed_vision = vision_embed
    if vis_mode == "remove":
        processed_vision = torch.zeros_like(vision_embed, dtype=dtype)
    elif vis_mode == "noise":
        processed_vision = generate_noise(
            vision_embed, vis_noise_type, vis_noise_scale
        )

    return processed_id, processed_txt, processed_vision
