"""Federated multimodal model implementations."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "MMFCF": (".mmfcf", "MMFCF"),
    "MMFedAvg": (".mmfedavg", "MMFedAvg"),
    "MMFedNCF": (".mmfedncf", "MMFedNCF"),
    "MMFedRAP": (".mmfedrap", "MMFedRAP"),
    "MMPFedRec": (".mmpfedrec", "MMPFedRec"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
