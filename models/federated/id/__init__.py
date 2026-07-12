"""Federated ID-based model implementations."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "FCF": (".fcf", "FCF"),
    "FedAvg": (".fedavg", "FedAvg"),
    "FedNCF": (".fedncf", "FedNCF"),
    "FedRAP": (".fedrap", "FedRAP"),
    "PFedRec": (".pfedrec", "PFedRec"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
