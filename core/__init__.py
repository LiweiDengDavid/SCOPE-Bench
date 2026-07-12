# coding: utf-8

"""NexusRec core package with a minimal eager import surface."""

from .config import ConfigManager
from .package_exports import export_names, lazy_getattr


_EXPORTS = {
    "quick_start": (".training.interface", "quick_start"),
    "run_training": (".training.interface", "run_training"),
    "prepare_env": (".training.environment", "prepare_env"),
    "run_unified_hpo": (".hpo.engine", "run_unified_hpo"),
}

__all__ = ["ConfigManager", *export_names(_EXPORTS)]


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
