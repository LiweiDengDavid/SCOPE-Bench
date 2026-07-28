"""Sequential multimodal model implementations."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "HM4SR": (".hm4sr", "HM4SR"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
