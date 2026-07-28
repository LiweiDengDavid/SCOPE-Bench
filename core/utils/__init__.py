"""Utility package with a minimal lazy export surface."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "get_local_time": (".training", "get_local_time"),
    "early_stopping": (".training", "early_stopping"),
    "dict2str": (".training", "dict2str"),
    "init_seed": (".training", "init_seed"),
    "get_model": ("..model_registry", "get_model"),
    "get_trainer": ("..model_registry", "get_trainer"),
    "build_knn_neighbourhood": (".graph", "build_knn_neighbourhood"),
    "compute_normalized_laplacian": (".graph", "compute_normalized_laplacian"),
    "build_sim": (".graph", "build_sim"),
    "build_knn_normalized_graph": (".graph", "build_knn_normalized_graph"),
    "build_norm_adj_matrix": (".graph", "build_norm_adj_matrix"),
    "modal_ablation": (".multimodal", "modal_ablation"),
    "resolve_multimodal_ablation": (".multimodal", "resolve_multimodal_ablation"),
    "Recommendation": (".recommendation", "Recommendation"),
    "Result": (".result", "Result"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
