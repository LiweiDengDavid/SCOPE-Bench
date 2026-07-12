# coding: utf-8

"""Training package with lazy exports to avoid package-level import cycles."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "quick_start": (".interface", "quick_start"),
    "prepare_env": (".environment", "prepare_env"),
    "Components": (".factory", "Components"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
