"""Shared torch full-sort ranking helpers for the eval paths.

These are the single finiteness gate and tie-break convention used by the
centralized, federated, and sequential torch eval paths.

The sequential evaluator's NumPy ranking path keeps its own implementation by
design, but shares this finiteness gate.
"""

from __future__ import annotations

import torch


def assert_finite_scores(scores: torch.Tensor, model) -> None:
    """Validate that all scores are finite before ranking.

    ``argsort`` would otherwise turn NaN/Inf into index-order rankings and make
    every metric meaningless.
    """
    if not torch.isfinite(scores).all():
        raise ValueError(
            f"{type(model).__name__}.full_sort_predict produced non-finite "
            "scores (NaN/Inf); rankings would be meaningless."
        )


def topk_from_scores(scores, history_mask, topk, *, score_components=None):
    """Mask seen items with ``-inf`` and return the stable top-k item indices.

    Applies the ``-inf`` history mask to ``scores`` and to every score-component
    tensor in place, then ranks
    with a stable descending ``argsort``. Stable sort breaks ties by lower item
    id deterministically on both CPU and CUDA — unlike ``torch.topk``, which is
    implementation-defined for ties and nondeterministic on CUDA.

    Args:
        scores: ``[batch, n_items]`` score tensor (mutated in place if masked).
        history_mask: ``None`` or a ``[2, n]`` tensor whose row 0 is batch-local
            user indices and row 1 is item ids to suppress.
        topk: number of top indices to return per row.
        score_components: optional dict of named ``[batch, n_items]`` tensors to
            mask identically (also mutated in place).

    Returns:
        ``[batch, topk]`` long tensor of ranked item indices.
    """
    if history_mask is not None:
        scores[history_mask[0], history_mask[1]] = -float("inf")
        if score_components:
            for component_scores in score_components.values():
                component_scores[history_mask[0], history_mask[1]] = -float("inf")
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :topk]
