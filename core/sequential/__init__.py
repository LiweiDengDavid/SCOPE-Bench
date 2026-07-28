# coding: utf-8

"""Sequential package with a minimal lazy export surface."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "SequentialRecommender": (".recommender", "SequentialRecommender"),
    "SequentialTrainer": (".trainer", "SequentialTrainer"),
    "SequentialEvaluator": (".evaluator", "SequentialEvaluator"),
    "SequentialDataset": (".dataset", "SequentialDataset"),
    "SequentialDataLoader": (".dataloader", "SequentialDataLoader"),
    "SequentialTorchDataset": (".dataloader", "SequentialTorchDataset"),
    "auto_setup": (".integration", "auto_setup"),
    "detect_sequential_model": (".integration", "detect_sequential_model"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
