"""Reference templates for adding new models."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "recommender_template": ".recommender_template",
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
