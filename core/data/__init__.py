# coding: utf-8

"""Data package with a minimal lazy export surface."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "RecDataset": (".dataset", "RecDataset"),
    "TrainDataLoader": (".dataloader", "TrainDataLoader"),
    "EvalDataLoader": (".dataloader", "EvalDataLoader"),
    "setup_centralized_features": (".features", "setup_centralized_features"),
    "setup_federated_features": (".features", "setup_federated_features"),
    "create_loaders": (".pipeline", "create_loaders"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
