"""Sequential ID-based model implementations."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "BERT4Rec": (".bert4rec", "BERT4Rec"),
    "GRU4Rec": (".gru4rec", "GRU4Rec"),
    "SASRec": (".sasrec", "SASRec"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
