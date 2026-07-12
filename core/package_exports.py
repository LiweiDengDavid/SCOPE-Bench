"""Small helper for consistent lazy package export surfaces."""

from __future__ import annotations

from importlib import import_module


def export_names(exports: dict) -> list:
    """Return the public export names in declaration order."""
    return list(exports.keys())


def lazy_getattr(
    package_name: str,
    exports: dict,
    namespace: dict,
    name: str,
):
    """Resolve one lazy export and cache it in the caller namespace.

    Mutates *namespace* (the caller's globals()) so that subsequent attribute
    lookups find the cached object directly and bypass __getattr__ entirely.
    """
    if name not in exports:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

    export_target = exports[name]
    if isinstance(export_target, str):
        module_name = export_target
        module = import_module(module_name, package_name)
        value = module
    else:
        module_name, attr_name = export_target
        module = import_module(module_name, package_name)
        value = getattr(module, attr_name)

    namespace[name] = value
    return value
