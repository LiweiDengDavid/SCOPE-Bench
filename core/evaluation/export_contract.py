# coding: utf-8
"""Shared export artifact contract."""

from __future__ import annotations

from typing import Any, Dict, Type

from ..utils.recommendation import Recommendation

SUPPORTED_FORMATS = Recommendation.FORMATS
EXPORT_KEYS = frozenset(
    {
        "enabled",
        "formats",
        "include_scores",
        "split",
        "topk",
        "path",
    }
)


def require_bool(value: Any, key: str, error_cls: Type[Exception]) -> None:
    if not isinstance(value, bool):
        raise error_cls(f"{key} must be boolean, got {value!r}")


def require_flag(value: Any, key: str, error_cls: Type[Exception]) -> None:
    require_bool(value, f"output.export.{key}", error_cls)


def require_string(
    value: Any,
    key: str,
    error_cls: Type[Exception],
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, str):
        raise error_cls(f"output.export.{key} must be a string, got {value!r}")
    if not allow_empty and value.strip() == "":
        raise error_cls(f"output.export.{key} must be a non-empty string.")


def require_formats(formats: Any, error_cls: Type[Exception]) -> None:
    if not isinstance(formats, list) or not formats:
        raise error_cls("output.export.formats must be a non-empty list.")

    non_string = [repr(fmt) for fmt in formats if not isinstance(fmt, str)]
    if non_string:
        raise error_cls(
            f"output.export.formats must contain only strings: {non_string}"
        )

    seen = set()
    duplicates = []
    for fmt in formats:
        if fmt in seen and fmt not in duplicates:
            duplicates.append(fmt)
        seen.add(fmt)
    if duplicates:
        raise error_cls(
            f"output.export.formats contains duplicate format(s): {duplicates}"
        )

    unsupported = sorted(set(formats) - SUPPORTED_FORMATS)
    if unsupported:
        raise error_cls(f"Unsupported output.export format(s): {unsupported}")


def require_topk(topk: Any, error_cls: Type[Exception]) -> None:
    if topk is None:
        return
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
        raise error_cls(
            f"output.export.topk must be null or a positive integer, got {topk!r}"
        )


def validate_section(
    config: Dict[str, Any],
    error_cls: Type[Exception],
    *,
    legacy_conflict_message: str,
) -> None:
    section = config["export"]
    if not isinstance(section, dict):
        raise error_cls("output.export must be a mapping.")
    extra_keys = sorted(set(section) - EXPORT_KEYS)
    if extra_keys:
        raise error_cls(f"Unsupported output.export key(s): {extra_keys}")
    require_bool(
        config["save_recommended_topk"],
        "output.save_recommended_topk",
        error_cls,
    )
    require_flag(section["enabled"], "enabled", error_cls)
    require_flag(section["include_scores"], "include_scores", error_cls)
    if section["enabled"] and config["save_recommended_topk"]:
        raise error_cls(legacy_conflict_message)
    if section["split"] != "test":
        raise error_cls("output.export currently supports split='test' only.")
    require_string(section["path"], "path", error_cls, allow_empty=True)
    require_formats(section["formats"], error_cls)
    require_topk(section["topk"], error_cls)
