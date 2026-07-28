# coding: utf-8

"""Base package with a minimal lazy export surface."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "RecommenderBase": (".recommender", "RecommenderBase"),
    "TrainerBase": (".trainer", "TrainerBase"),
    "xavier_normal_initialization": (".init", "xavier_normal_initialization"),
    "BPRLoss": (".loss", "BPRLoss"),
    "EmbLoss": (".loss", "EmbLoss"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
