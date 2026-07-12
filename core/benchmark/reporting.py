# coding: utf-8
"""Benchmark summary and reporting utilities."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .options import resolve_reporting_config
from ..hpo.parameters import select_best_index
from ..utils.result import Result
from ..utils.significance import compare_paired_results
from .runner import get_benchmark_paths, load_benchmark_latest_rows


_RESULT_META_COLUMNS = {"model", "dataset", "type", "comment"}


def _parse_metric_mapping(raw_value: Any, source_path: Path, field_name: str) -> Dict[str, float]:
    if pd.isna(raw_value):
        raise ValueError(f"Missing '{field_name}' in benchmark artifact: {source_path}")

    parsed = raw_value
    if isinstance(raw_value, str):
        parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Benchmark artifact field '{field_name}' must parse to a mapping: {source_path}"
        )

    metrics: Dict[str, float] = {}
    for metric_name, metric_value in parsed.items():
        metrics[str(metric_name)] = float(metric_value)
    return metrics


def _load_train_completed_artifact(result_path: Path) -> Dict[str, Any]:
    result_row = Result.load(
        result_path,
        required_columns=sorted(_RESULT_META_COLUMNS),
        name="Benchmark result CSV",
    )
    metrics = Result.metrics(result_row, _RESULT_META_COLUMNS)

    return {
        "metrics": metrics,
        "metrics_source": "result_csv",
        "metrics_source_file": str(result_path),
        "hpo_strategy": None,
        "best_trial_num": None,
        "best_target_score": None,
    }


def _find_hpo_history_csv(run: Dict[str, Any]) -> Path:
    hyper_search_dir = Path(run["hyper_search_dir"])
    if not hyper_search_dir.exists():
        raise FileNotFoundError(f"HPO artifact directory not found: {hyper_search_dir}")

    candidates = [
        path
        for path in sorted(hyper_search_dir.glob("*.csv"))
        if run["output_comment"] in path.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No HPO history CSV matched output_comment='{run['output_comment']}' in {hyper_search_dir}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one HPO history CSV for output_comment='{run['output_comment']}', "
            f"found {len(candidates)} in {hyper_search_dir}"
        )
    return candidates[0]


def _load_hpo_completed_artifact(run: Dict[str, Any]) -> Dict[str, Any]:
    history_path = _find_hpo_history_csv(run)
    frame = pd.read_csv(history_path)
    if frame.empty:
        raise ValueError(f"HPO history CSV is empty: {history_path}")
    if "status" not in frame.columns:
        raise ValueError(f"HPO history CSV is missing required column 'status': {history_path}")
    if "target_score" not in frame.columns:
        raise ValueError(f"HPO history CSV is missing required column 'target_score': {history_path}")
    if "test_metrics" not in frame.columns:
        raise ValueError(f"HPO history CSV is missing required column 'test_metrics': {history_path}")

    completed = frame[frame["status"].astype(str).str.lower() == "completed"].copy()
    if completed.empty:
        raise ValueError(f"HPO history CSV has no completed trials: {history_path}")

    # target_score is the raw validation metric, NOT normalized to a single
    # direction (see core/hpo/optuna_backend.py and engine.py, which both branch
    # on the objective). Mirror that direction-aware selection here, reading the
    # objective from the run's overrides (default 'maximize', matching
    # configs/overall.yaml optimization.objective). A non-default 'minimize'
    # objective only ever reaches a run via a per-run override (so it is present
    # here in overrides_json); no committed model/dataset YAML sets it, and
    # validate_training (core/config.py) fails fast if objective disagrees with
    # valid_metric_bigger, so the merged direction is unambiguous. Resolving via a
    # full ConfigManager here is deliberately avoided: it would run set_paths
    # (os.makedirs) and assign_runtime_device (torch.cuda.set_device) as a side
    # effect of read-only report generation.
    overrides = json.loads(run["overrides_json"])
    objective = "maximize"
    if "optimization" in overrides and "objective" in overrides["optimization"]:
        objective = overrides["optimization"]["objective"]

    completed["target_score"] = pd.to_numeric(completed["target_score"])
    best_index = select_best_index(completed["target_score"], objective)
    best_row = completed.loc[best_index]
    metrics = _parse_metric_mapping(best_row["test_metrics"], history_path, "test_metrics")

    strategy = None
    if "strategy" in best_row.index and not pd.isna(best_row["strategy"]):
        strategy = str(best_row["strategy"])

    best_trial_num = None
    if "trial_num" in best_row.index and not pd.isna(best_row["trial_num"]):
        best_trial_num = int(best_row["trial_num"])

    return {
        "metrics": metrics,
        "metrics_source": "hpo_csv",
        "metrics_source_file": str(history_path),
        "hpo_strategy": strategy,
        "best_trial_num": best_trial_num,
        "best_target_score": float(best_row["target_score"]),
    }


def _load_completed_artifact(run: Dict[str, Any], latest_row: Dict[str, Any]) -> Dict[str, Any]:
    if run["mode"] == "train":
        result_path = Path(latest_row["result_file"])
        if not result_path.exists():
            raise FileNotFoundError(
                f"Completed benchmark run is missing its result CSV: {result_path}"
            )
        return _load_train_completed_artifact(result_path)

    if run["mode"] == "hpo":
        return _load_hpo_completed_artifact(run)

    raise ValueError(
        "Benchmark summary does not support unsupported mode "
        f"'{run['mode']}' in experiment '{run['experiment_name']}'."
    )


def _resolve_metric_columns(
    completed_artifacts: List[Dict[str, Any]],
    requested_metrics: List[str] | None,
) -> List[str]:
    if not completed_artifacts:
        return list(requested_metrics) if requested_metrics is not None else []

    metric_sets = []
    for artifact in completed_artifacts:
        metric_sets.append(list(artifact["metrics"].keys()))

    reference = metric_sets[0]
    for metric_set in metric_sets[1:]:
        if metric_set != reference:
            raise ValueError(
                "Completed benchmark result rows in the same manifest must share an identical metric schema."
            )

    if requested_metrics is None:
        return reference

    for metric in requested_metrics:
        if metric not in reference:
            raise ValueError(f"Requested metric '{metric}' not found in benchmark result schema.")
    return requested_metrics


def _build_run_summary_rows(
    plan: Dict[str, Any],
    latest_rows: Dict[str, Dict[str, Any]],
    completed_artifacts: Dict[str, Dict[str, Any]],
    metrics: List[str],
) -> List[Dict[str, Any]]:
    run_rows: List[Dict[str, Any]] = []

    for run in plan["runs"]:
        latest = latest_rows.get(run["run_id"])
        if latest is None:
            latest = dict(run)

        summary_row = {
            "experiment_name": run["experiment_name"],
            "run_id": run["run_id"],
            "model": run["model"],
            "dataset": run["dataset"],
            "seed": run["seed"],
            "mode": run["mode"],
            "type": run["type"],
            "comment": run["comment"],
            "output_comment": run["output_comment"],
            "status": latest["status"],
            "attempt": latest["attempt"],
            "return_code": latest["return_code"],
            "failure_state": latest["failure_state"],
            "metrics_source": None,
            "metrics_source_file": None,
            "hpo_strategy": None,
            "best_trial_num": None,
            "best_target_score": None,
            "result_file": latest["result_file"],
            "started_at": latest["started_at"],
            "finished_at": latest["finished_at"],
        }

        if latest["status"] == "completed":
            artifact = completed_artifacts[run["run_id"]]
            summary_row["metrics_source"] = artifact["metrics_source"]
            summary_row["metrics_source_file"] = artifact["metrics_source_file"]
            summary_row["hpo_strategy"] = artifact["hpo_strategy"]
            summary_row["best_trial_num"] = artifact["best_trial_num"]
            summary_row["best_target_score"] = artifact["best_target_score"]
            for metric in metrics:
                if metric not in artifact["metrics"]:
                    raise ValueError(
                        f"Metric '{metric}' not found in benchmark artifact for run_id={run['run_id']}: "
                        f"{artifact['metrics_source_file']}"
                    )
                summary_row[metric] = float(artifact["metrics"][metric])
        else:
            for metric in metrics:
                summary_row[metric] = None

        run_rows.append(summary_row)

    return run_rows


def _aggregate_group_summary(run_rows: List[Dict[str, Any]], metrics: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for row in run_rows:
        key = (row["experiment_name"], row["model"], row["dataset"], row["mode"], row["type"])
        groups.setdefault(key, []).append(row)

    group_rows: List[Dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        completed_rows = [row for row in rows if row["status"] == "completed"]
        group_row: Dict[str, Any] = {
            "experiment_name": key[0],
            "model": key[1],
            "dataset": key[2],
            "mode": key[3],
            "type": key[4],
            "planned_runs": len(rows),
            "completed_runs": len(completed_rows),
            "failed_or_incomplete_runs": len(rows) - len(completed_rows),
        }

        for metric in metrics:
            if completed_rows:
                series = pd.Series([float(row[metric]) for row in completed_rows], dtype="float64")
                group_row[f"{metric}_mean"] = float(series.mean())
                group_row[f"{metric}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            else:
                group_row[f"{metric}_mean"] = None
                group_row[f"{metric}_std"] = None

        group_rows.append(group_row)

    return group_rows


def _build_failed_or_incomplete_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        row for row in run_rows
        if row["status"] != "completed"
    ]


def _build_significance_summary(
    run_rows: List[Dict[str, Any]],
    baseline_model: str,
    metrics: List[str],
    test_name: str,
    pair_field: str,
) -> List[Dict[str, Any]]:
    grouped_rows: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in run_rows:
        if row["status"] == "completed":
            key = (row["experiment_name"], row["dataset"], row["mode"], row["type"])
            grouped_rows.setdefault(key, []).append(row)

    significance_rows: List[Dict[str, Any]] = []
    baseline_found = False
    for group_key in sorted(grouped_rows):
        rows = grouped_rows[group_key]
        baseline_rows = [row for row in rows if row["model"] == baseline_model]
        if not baseline_rows:
            continue
        baseline_found = True

        candidate_models = sorted({row["model"] for row in rows if row["model"] != baseline_model})
        for candidate_model in candidate_models:
            candidate_rows = [row for row in rows if row["model"] == candidate_model]
            if not candidate_rows:
                continue

            def _row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
                record = {
                    key: row[key]
                    for key in ("model", "dataset", "type", "comment", "output_comment", "run_id", "seed")
                }
                for metric in metrics:
                    record[metric] = row[metric]
                return record

            baseline_records = [_row_to_record(row) for row in baseline_rows]
            candidate_records = [_row_to_record(row) for row in candidate_rows]

            metadata, summaries = compare_paired_results(
                baseline_records=baseline_records,
                candidate_records=candidate_records,
                metrics=metrics,
                pair_field=pair_field,
                test_name=test_name,
            )
            for summary in summaries:
                significance_rows.append(
                    {
                        "experiment_name": group_key[0],
                        "dataset": group_key[1],
                        "mode": group_key[2],
                        "type": group_key[3],
                        "baseline_model": metadata["baseline_model"],
                        "candidate_model": metadata["candidate_model"],
                        **summary,
                    }
                )

    if not baseline_found:
        raise ValueError(
            f"Requested significance baseline '{baseline_model}' was not found in completed benchmark runs."
        )

    return significance_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    return pd.DataFrame(rows).to_string(index=False)


def _write_summary_markdown(
    path: Path,
    plan: Dict[str, Any],
    run_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    failed_rows: List[Dict[str, Any]],
    significance_rows: List[Dict[str, Any]],
) -> None:
    completed_runs = len([row for row in run_rows if row["status"] == "completed"])
    lines = [
        f"# Benchmark Summary: {plan['manifest_name']}",
        "",
        f"- Manifest Hash: `{plan['manifest_hash']}`",
        f"- Planned Runs: {len(run_rows)}",
        f"- Completed Runs: {completed_runs}",
        f"- Failed/Incomplete Runs: {len(failed_rows)}",
        "",
        "## Group Summary",
        "",
        "```text",
        _render_table(group_rows),
        "```",
        "",
        "## Failed or Incomplete Runs",
        "",
        "```text",
        _render_table(
            [
                {
                    "run_id": row["run_id"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "status": row["status"],
                    "return_code": row["return_code"],
                }
                for row in failed_rows
            ]
        ),
        "```",
    ]

    if significance_rows:
        lines.extend(
            [
                "",
                "## Significance",
                "",
                "```text",
                _render_table(significance_rows),
                "```",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_benchmark_summary(
    plan: Dict[str, Any],
    output_root: str | Path,
    significance_baseline: str | None = None,
    significance_metrics: List[str] | None = None,
    significance_test: str | None = None,
    significance_pair_field: str | None = None,
) -> Dict[str, Any]:
    reporting_options = resolve_reporting_config(
        plan["reporting"] if "reporting" in plan else None,
        metrics=significance_metrics,
        significance_baseline=significance_baseline,
        significance_test=significance_test,
        significance_pair_field=significance_pair_field,
    )
    latest_rows = load_benchmark_latest_rows(
        output_root,
        plan["manifest_name"],
        plan["manifest_hash"],
    )

    completed_artifacts: Dict[str, Dict[str, Any]] = {}
    for run in plan["runs"]:
        latest = latest_rows.get(run["run_id"])
        if latest is not None and latest["status"] == "completed":
            completed_artifacts[run["run_id"]] = _load_completed_artifact(run, latest)

    metrics = _resolve_metric_columns(
        list(completed_artifacts.values()),
        reporting_options["metrics"],
    )
    run_rows = _build_run_summary_rows(plan, latest_rows, completed_artifacts, metrics)
    group_rows = _aggregate_group_summary(run_rows, metrics)
    failed_rows = _build_failed_or_incomplete_rows(run_rows)

    significance_rows: List[Dict[str, Any]] = []
    if reporting_options["significance_baseline"] is not None:
        significance_rows = _build_significance_summary(
            run_rows,
            baseline_model=reporting_options["significance_baseline"],
            metrics=metrics,
            test_name=reporting_options["significance_test"],
            pair_field=reporting_options["significance_pair_field"],
        )

    paths = get_benchmark_paths(output_root, plan["manifest_name"], plan["manifest_hash"])
    summary_runs_csv = paths["output_dir"] / "summary_runs.csv"
    summary_groups_csv = paths["output_dir"] / "summary_groups.csv"
    summary_markdown = paths["output_dir"] / "summary.md"
    summary_significance_csv = paths["output_dir"] / "summary_significance.csv"

    _write_csv(summary_runs_csv, run_rows)
    _write_csv(summary_groups_csv, group_rows)
    _write_summary_markdown(
        summary_markdown,
        plan,
        run_rows,
        group_rows,
        failed_rows,
        significance_rows,
    )
    if reporting_options["significance_baseline"] is not None:
        _write_csv(summary_significance_csv, significance_rows)

    return {
        "summary_runs_csv": str(summary_runs_csv),
        "summary_groups_csv": str(summary_groups_csv),
        "summary_markdown": str(summary_markdown),
        "summary_significance_csv": (
            str(summary_significance_csv)
            if reporting_options["significance_baseline"] is not None
            else None
        ),
        "metrics": metrics,
        "completed_runs": len([row for row in run_rows if row["status"] == "completed"]),
        "failed_or_incomplete_runs": len(failed_rows),
        "reporting": reporting_options,
    }


__all__ = ["build_benchmark_summary"]
