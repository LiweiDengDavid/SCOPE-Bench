"""Centralized multimodal contrastive models."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "BM3": (".bm3", "BM3"),
    "IDFREE": (".idfree", "IDFREE"),
    "SLMRec": (".slmrec", "SLMRec"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
