# coding: utf-8

"""Model package namespace with family-level lazy exports."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "centralized": ".centralized",
    "federated": ".federated",
    "sequential": ".sequential",
    "templates": ".templates",
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
