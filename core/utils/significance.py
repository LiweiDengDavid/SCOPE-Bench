"""Utilities for strict paired significance testing on repeated experiment results."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .result import Result

_META_COLUMNS = {"model", "dataset", "type", "comment"}
_TIE_ATOL = 1.0e-8


def _record_label(record: Dict[str, Any], fallback: str) -> str:
    if "__path__" in record:
        return str(record["__path__"])
    return fallback


def _ties_mask(baseline_values: np.ndarray, candidate_values: np.ndarray) -> np.ndarray:
    return np.isclose(candidate_values, baseline_values, rtol=0.0, atol=_TIE_ATOL)


def expand_result_inputs(inputs: Sequence[str]) -> List[Path]:
    """Expand files, directories, and glob patterns into concrete CSV paths."""
    resolved_paths: List[Path] = []
    seen: set[Path] = set()

    for raw_input in inputs:
        matches: List[Path] = []
        candidate = Path(raw_input)

        if any(token in raw_input for token in ["*", "?", "["]):
            matches = [Path(path).resolve() for path in sorted(glob.glob(raw_input, recursive=True))]
        elif candidate.is_dir():
            matches = sorted(path.resolve() for path in candidate.rglob("*.csv"))
        elif candidate.is_file():
            matches = [candidate.resolve()]

        if not matches:
            raise FileNotFoundError(f"No result CSV files found for input: {raw_input}")

        for path in matches:
            if path.suffix.lower() != ".csv":
                continue
            if path not in seen:
                seen.add(path)
                resolved_paths.append(path)

    return resolved_paths


def load_result_record(path: Path) -> Dict[str, Any]:
    """Load one result CSV that must contain exactly one experiment row."""
    record = Result.load(
        path,
        required_columns=sorted(_META_COLUMNS),
        name="Result CSV for significance testing",
    )
    record["__path__"] = str(path)
    record["__stem__"] = path.stem
    return record


def load_result_records(inputs: Sequence[str]) -> List[Dict[str, Any]]:
    """Load result records from files, directories, or glob patterns."""
    return [load_result_record(path) for path in expand_result_inputs(inputs)]


def validate_group_consistency(records: Sequence[Dict[str, Any]], label: str) -> Dict[str, str]:
    """Ensure one result group represents a single repeated experiment family."""
    if not records:
        raise ValueError(f"{label} result group is empty.")

    summary: Dict[str, str] = {}
    for field in ("model", "dataset", "type"):
        values = {str(record[field]) for record in records}
        if len(values) != 1:
            raise ValueError(
                f"{label} results must share the same '{field}', but found: {sorted(values)}"
            )
        summary[field] = next(iter(values))
    return summary


def align_paired_records(
    baseline_records: Sequence[Dict[str, Any]],
    candidate_records: Sequence[Dict[str, Any]],
    pair_field: str,
) -> Tuple[Dict[str, str], List[Tuple[str, Dict[str, Any], Dict[str, Any]]]]:
    """Align two repeated-run result groups with a strict one-to-one key."""
    baseline_summary = validate_group_consistency(baseline_records, "Baseline")
    candidate_summary = validate_group_consistency(candidate_records, "Candidate")

    if baseline_summary["dataset"] != candidate_summary["dataset"]:
        raise ValueError(
            "Baseline and candidate results must use the same dataset, "
            f"got {baseline_summary['dataset']} vs {candidate_summary['dataset']}."
        )
    if baseline_summary["type"] != candidate_summary["type"]:
        raise ValueError(
            "Baseline and candidate results must use the same run type, "
            f"got {baseline_summary['type']} vs {candidate_summary['type']}."
        )

    baseline_by_key: Dict[str, Dict[str, Any]] = {}
    for record in baseline_records:
        key = Result.pair_key(
            record,
            pair_field,
            _record_label(record, "baseline record"),
        )
        if key in baseline_by_key:
            raise ValueError(f"Duplicate baseline pair key '{key}' detected.")
        baseline_by_key[key] = record

    candidate_by_key: Dict[str, Dict[str, Any]] = {}
    for record in candidate_records:
        key = Result.pair_key(
            record,
            pair_field,
            _record_label(record, "candidate record"),
        )
        if key in candidate_by_key:
            raise ValueError(f"Duplicate candidate pair key '{key}' detected.")
        candidate_by_key[key] = record

    baseline_keys = set(baseline_by_key)
    candidate_keys = set(candidate_by_key)
    if baseline_keys != candidate_keys:
        missing_in_candidate = sorted(baseline_keys - candidate_keys)
        missing_in_baseline = sorted(candidate_keys - baseline_keys)
        raise ValueError(
            "Pair keys do not match between baseline and candidate results. "
            f"Missing in candidate: {missing_in_candidate}. Missing in baseline: {missing_in_baseline}."
        )

    ordered_keys = sorted(baseline_keys)
    pairs = [(key, baseline_by_key[key], candidate_by_key[key]) for key in ordered_keys]
    if len(pairs) < 2:
        raise ValueError("At least two paired runs are required for significance testing.")

    metadata = {
        "baseline_model": baseline_summary["model"],
        "candidate_model": candidate_summary["model"],
        "dataset": baseline_summary["dataset"],
        "type": baseline_summary["type"],
        "pair_field": pair_field,
        "n_pairs": str(len(pairs)),
    }
    return metadata, pairs


def infer_metric_columns(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Infer metric columns from one result group."""
    metrics: List[str] = []
    for column in records[0].keys():
        if column in _META_COLUMNS or column.startswith("__"):
            continue
        metrics.append(column)
    return metrics


def extract_metric_arrays(
    pairs: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract baseline/candidate metric arrays from aligned pairs."""
    baseline_values: List[float] = []
    candidate_values: List[float] = []

    for _key, baseline_record, candidate_record in pairs:
        if metric not in baseline_record:
            raise ValueError(
                f"Metric '{metric}' not found in baseline record {baseline_record['__path__']}"
            )
        if metric not in candidate_record:
            raise ValueError(
                f"Metric '{metric}' not found in candidate record {candidate_record['__path__']}"
            )
        baseline_values.append(float(baseline_record[metric]))
        candidate_values.append(float(candidate_record[metric]))

    return np.asarray(baseline_values, dtype=np.float64), np.asarray(candidate_values, dtype=np.float64)


def run_paired_test(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    test_name: str,
) -> Tuple[float, float]:
    """Run a strict paired statistical test."""
    if np.all(_ties_mask(baseline_values, candidate_values)):
        return 0.0, 1.0

    if test_name == "wilcoxon":
        statistic, p_value = stats.wilcoxon(candidate_values, baseline_values, zero_method="wilcox")
        return float(statistic), float(p_value)

    if test_name == "paired_t":
        statistic, p_value = stats.ttest_rel(candidate_values, baseline_values)
        return float(statistic), float(p_value)

    raise ValueError(f"Unsupported paired test: {test_name}")


def summarize_metric(
    metric: str,
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    test_name: str,
) -> Dict[str, Any]:
    """Summarize paired significance results for one metric."""
    statistic, p_value = run_paired_test(baseline_values, candidate_values, test_name)
    differences = candidate_values - baseline_values

    # Partition the pairs into mutually exclusive win/loss/tie so the three counts
    # sum to n_pairs. Ties use the same isclose tolerance that short-circuits the
    # p-value above; without excluding ties from the strict >/< comparisons a
    # near-equal pair would be double-counted as both a win/loss and a tie.
    ties_mask = _ties_mask(baseline_values, candidate_values)

    return {
        "metric": metric,
        "n_pairs": len(baseline_values),
        "test": test_name,
        "baseline_mean": float(np.mean(baseline_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "mean_diff": float(np.mean(differences)),
        "baseline_std": float(np.std(baseline_values, ddof=1)),
        "candidate_std": float(np.std(candidate_values, ddof=1)),
        "statistic": statistic,
        "p_value": p_value,
        "wins": int(np.sum((candidate_values > baseline_values) & ~ties_mask)),
        "losses": int(np.sum((candidate_values < baseline_values) & ~ties_mask)),
        "ties": int(np.sum(ties_mask)),
    }


def _holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down family-wise correction.

    Controls the family-wise error rate over the set of tested metrics so that
    reporting many per-metric p-values does not inflate the false-positive rate.
    Returns adjusted p-values in the original order (each >= its raw p-value).
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, float(p_values[idx]) * (m - rank))
        running_max = max(running_max, adjusted)
        corrected[idx] = running_max
    return corrected


def compare_paired_results(
    baseline_records: Sequence[Dict[str, Any]],
    candidate_records: Sequence[Dict[str, Any]],
    metrics: Sequence[str] | None,
    pair_field: str,
    test_name: str,
    correction: str = "holm",
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Compare two repeated-run result groups with strict pairing.

    Per-metric p-values are corrected for multiple comparisons across the metric
    family (``correction``: "holm" or "none") and attached as ``p_value_corrected``.
    """
    metadata, pairs = align_paired_records(baseline_records, candidate_records, pair_field)

    selected_metrics = list(metrics) if metrics else infer_metric_columns(baseline_records)
    if not selected_metrics:
        raise ValueError("No metric columns available for significance testing.")

    summaries = []
    for metric in selected_metrics:
        baseline_values, candidate_values = extract_metric_arrays(pairs, metric)
        summaries.append(
            summarize_metric(metric, baseline_values, candidate_values, test_name)
        )

    if correction == "holm":
        corrected = _holm_correction([s["p_value"] for s in summaries])
    elif correction == "none":
        corrected = [s["p_value"] for s in summaries]
    else:
        raise ValueError(f"Unsupported correction: {correction!r}. Use 'holm' or 'none'.")
    for summary, p_corrected in zip(summaries, corrected):
        summary["p_value_corrected"] = float(p_corrected)
    metadata["correction"] = correction

    return metadata, summaries


def format_summary_table(
    metadata: Dict[str, str],
    summaries: Iterable[Dict[str, Any]],
) -> str:
    """Format the significance summary as a readable text table."""
    summary_frame = pd.DataFrame(list(summaries))
    if summary_frame.empty:
        raise ValueError("No significance summaries to format.")

    render_frame = summary_frame.copy()
    float_columns = [
        "baseline_mean",
        "candidate_mean",
        "mean_diff",
        "baseline_std",
        "candidate_std",
        "statistic",
        "p_value",
    ]
    for column in float_columns:
        render_frame[column] = render_frame[column].map(lambda value: f"{value:.6f}")

    header = [
        "Paired Significance Test",
        f"Baseline Model: {metadata['baseline_model']}",
        f"Candidate Model: {metadata['candidate_model']}",
        f"Dataset: {metadata['dataset']}",
        f"Run Type: {metadata['type']}",
        f"Pair Field: {metadata['pair_field']}",
        f"Paired Runs: {metadata['n_pairs']}",
        "",
    ]
    return "\n".join(header) + render_frame.to_string(index=False)


__all__ = [
    "align_paired_records",
    "compare_paired_results",
    "expand_result_inputs",
    "format_summary_table",
    "infer_metric_columns",
    "load_result_record",
    "load_result_records",
]
