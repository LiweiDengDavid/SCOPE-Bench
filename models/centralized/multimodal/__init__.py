"""Centralized multimodal model families."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "contrastive": ".contrastive",
    "diffusion": ".diffusion",
    "factorization": ".factorization",
    "graph": ".graph",
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
