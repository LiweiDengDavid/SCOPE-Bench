# coding: utf-8

"""Runtime infrastructure package with a minimal lazy export surface."""

from ..package_exports import export_names, lazy_getattr


_EXPORTS = {
    "get_system_status": (".monitor", "get_system_status"),
    "init_logger": (".logger", "init_logger"),
    "TrainLogger": (".logger", "TrainLogger"),
}

__all__ = export_names(_EXPORTS)


def __getattr__(name):
    return lazy_getattr(__name__, _EXPORTS, globals(), name)
