"""Centralized ID graph models."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "LightGCN": (".lightgcn", "LightGCN"),
    "SGL": (".sgl", "SGL"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
