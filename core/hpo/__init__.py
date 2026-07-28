# coding: utf-8

"""HPO package with a minimal lazy export surface."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "run_unified_hpo": (".engine", "run_unified_hpo"),
    "UnifiedHPOManager": (".engine", "UnifiedHPOManager"),
    "ParameterGenerator": (".parameters", "ParameterGenerator"),
    "suggest_parameters": (".optuna_backend", "suggest_parameters"),
    "run_parallel_hpo": (".parallel", "run_parallel_hpo"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
