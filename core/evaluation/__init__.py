# coding: utf-8

"""Evaluation package with a minimal lazy export surface."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "TopKEvaluator": (".evaluator", "TopKEvaluator"),
    "novelty_": (".topk_kernel", "novelty_"),
    "diversity_": (".topk_kernel", "diversity_"),
    "coverage_": (".topk_kernel", "coverage_"),
    "build_cds_gain_table": (".lcds", "build_cds_gain_table"),
    "build_lcds_result_dict": (".lcds", "build_lcds_result_dict"),
    "lcds_metric_arrays": (".lcds", "lcds_metric_arrays"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
