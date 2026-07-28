"""Centralized ID autoencoder models."""

from core.package_exports import export_names, lazy_getattr


_EXPORTS = {
    "MultiVAE": (".multivae", "MultiVAE"),
    "RecVAE": (".recvae", "RecVAE"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
