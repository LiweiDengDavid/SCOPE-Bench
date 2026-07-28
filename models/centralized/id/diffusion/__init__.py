"""Centralized ID diffusion models."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "CFDiff": (".cfdiff", "CFDiff"),
    "DiffRec": (".diffrec", "DiffRec"),
    "cfdiff_components": ".cfdiff_components",
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
