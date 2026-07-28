"""Sequential recommendation model families."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "id": ".id",
    "multimodal": ".multimodal",
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
