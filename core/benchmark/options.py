# coding: utf-8
"""Shared benchmark reporting option normalization."""

from __future__ import annotations

from typing import Any, Dict, List


_ALLOWED_REPORTING_KEYS = {
    "metrics",
    "significance_baseline",
    "significance_test",
    "significance_pair_field",
}
_ALLOWED_SIGNIFICANCE_TESTS = {"wilcoxon", "paired_t"}
# Significance pairs a baseline model's runs against a candidate model's runs.
# Only fields shared 1:1 across the two models per pairing group are valid:
#   seed    — the canonical pairing key (same seeds run for both models)
#   comment — usable when the spec varies comment per seed
# output_comment and run_id ENCODE the model, so cross-model keys never intersect
# (they would deterministically fail to pair) — excluded.
_ALLOWED_SIGNIFICANCE_PAIR_FIELDS = {"seed", "comment"}


def _normalize_metrics(metrics: List[str] | None) -> List[str] | None:
    if metrics is None:
        return None
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Benchmark reporting metrics must be a non-empty list when provided.")
    return [str(metric) for metric in metrics]


def _normalize_significance_baseline(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Benchmark reporting significance_baseline must be a non-empty string.")
    return value


def _normalize_significance_test(value: str | None) -> str:
    if value is None:
        return "wilcoxon"
    if value not in _ALLOWED_SIGNIFICANCE_TESTS:
        raise ValueError(
            "Benchmark reporting significance_test must be one of "
            f"{sorted(_ALLOWED_SIGNIFICANCE_TESTS)}."
        )
    return value


def _normalize_significance_pair_field(value: str | None) -> str:
    if value is None:
        return "seed"
    if value not in _ALLOWED_SIGNIFICANCE_PAIR_FIELDS:
        raise ValueError(
            "Benchmark reporting significance_pair_field must be one of "
            f"{sorted(_ALLOWED_SIGNIFICANCE_PAIR_FIELDS)}."
        )
    return value


def normalize_reporting_config(raw_reporting: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normalize optional benchmark reporting config from a manifest."""
    if raw_reporting is None:
        raw_reporting = {}
    if not isinstance(raw_reporting, dict):
        raise ValueError("Benchmark reporting config must be a mapping.")

    unknown_keys = sorted(set(raw_reporting) - _ALLOWED_REPORTING_KEYS)
    if unknown_keys:
        raise ValueError(
            "Benchmark reporting config contains unsupported keys: "
            + ", ".join(unknown_keys)
        )

    normalized = {
        "metrics": None,
        "significance_baseline": None,
        "significance_test": "wilcoxon",
        "significance_pair_field": "seed",
    }

    if "metrics" in raw_reporting:
        normalized["metrics"] = _normalize_metrics(raw_reporting["metrics"])
    if "significance_baseline" in raw_reporting:
        normalized["significance_baseline"] = _normalize_significance_baseline(
            raw_reporting["significance_baseline"]
        )
    if "significance_test" in raw_reporting:
        normalized["significance_test"] = _normalize_significance_test(
            raw_reporting["significance_test"]
        )
    if "significance_pair_field" in raw_reporting:
        normalized["significance_pair_field"] = _normalize_significance_pair_field(
            raw_reporting["significance_pair_field"]
        )

    return normalized


def resolve_reporting_config(
    plan_reporting: Dict[str, Any] | None,
    metrics: List[str] | None = None,
    significance_baseline: str | None = None,
    significance_test: str | None = None,
    significance_pair_field: str | None = None,
) -> Dict[str, Any]:
    """Resolve plan defaults with explicit CLI-style overrides."""
    resolved = normalize_reporting_config(plan_reporting)
    if metrics is not None:
        resolved["metrics"] = _normalize_metrics(metrics)
    if significance_baseline is not None:
        resolved["significance_baseline"] = _normalize_significance_baseline(
            significance_baseline
        )
    if significance_test is not None:
        resolved["significance_test"] = _normalize_significance_test(significance_test)
    if significance_pair_field is not None:
        resolved["significance_pair_field"] = _normalize_significance_pair_field(
            significance_pair_field
        )
    return resolved


__all__ = ["normalize_reporting_config", "resolve_reporting_config"]
