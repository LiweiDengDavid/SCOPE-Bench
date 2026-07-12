# coding: utf-8
"""Metric extraction utilities."""

from typing import Any, Dict, Optional


def _find_metric(result_dict: Dict[str, Any], target_metric: str) -> Optional[float]:
    """Return a metric value from the canonical nested metric structure."""
    if not result_dict:
        return None

    for wrapper_key in ("valid_metrics", "test_metrics"):
        if wrapper_key in result_dict:
            inner = result_dict[wrapper_key]
            if isinstance(inner, dict):
                found = _find_metric(inner, target_metric)
                if found is not None:
                    return found

    if target_metric in result_dict:
        return float(result_dict[target_metric])

    target_lower = target_metric.lower()
    for key, value in result_dict.items():
        if key.lower() == target_lower:
            return float(value)

    return None


def extract_target_metric(result_dict: Dict[str, Any], target_metric: str) -> float:
    """Extract a target metric from the canonical result contract."""
    result = _find_metric(result_dict, target_metric)
    if result is None:
        raise ValueError(f"Metric '{target_metric}' not found in evaluation results")
    return result
