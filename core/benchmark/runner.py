# coding: utf-8
"""Benchmark planning, execution, and ledger utilities."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, List

import yaml

from ..config import ConfigManager, deep_merge_dict
from .options import normalize_reporting_config


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Strategies whose trial CSV lands in hyper_search_dir, where --summarize looks
# (Tutorial 07). Serial grid writes its CSV through the save path under an
# "experiment" alias (core/hpo/engine._csv_path), so a grid benchmark would
# train to completion and then fail summarization — reject it at plan time
# (mirrors core/hpo/parallel.build_parallel_shards' strategy whitelist).
_SUMMARIZABLE_HPO_STRATEGIES = frozenset({"bayesian", "tpe", "random"})
_LEDGER_FIELDS = [
    "manifest_name",
    "manifest_hash",
    "experiment_name",
    "run_id",
    "attempt",
    "model",
    "dataset",
    "seed",
    "mode",
    "type",
    "comment",
    "output_comment",
    "status",
    "return_code",
    "failure_state",
    "recorded_at",
    "started_at",
    "finished_at",
    "command",
    "result_file",
    "log_dir",
    "checkpoint_dir",
    "hyper_search_dir",
    "overrides_json",
]


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark spec root must be a mapping: {path}")
    return payload


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_normalize_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_benchmark_spec(spec_path: str | Path) -> Dict[str, Any]:
    path = Path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark spec not found: {path}")

    spec = _load_yaml_mapping(path)
    if "experiments" not in spec:
        raise ValueError(f"Benchmark spec must define an 'experiments' list: {path}")
    if not isinstance(spec["experiments"], list):
        raise ValueError(f"Benchmark spec 'experiments' must be a list: {path}")
    return spec


def build_manifest_hash(spec: Dict[str, Any]) -> str:
    execution_spec = {
        "experiments": spec["experiments"],
    }
    canonical = _canonical_json(execution_spec)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _as_list(entry: Dict[str, Any], singular_key: str, plural_key: str) -> List[Any]:
    name = entry["name"] if "name" in entry else "unnamed"
    if plural_key in entry:
        values = entry[plural_key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Benchmark experiment '{name}' must provide a non-empty list for '{plural_key}'")
        return values
    if singular_key in entry:
        return [entry[singular_key]]
    raise ValueError(f"Benchmark experiment '{name}' must provide '{singular_key}' or '{plural_key}'")


def _normalize_experiment_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Each benchmark experiment entry must be a mapping")
    if "command" in entry:
        raise ValueError(
            "Command-based queue entries are not supported by the dry-run benchmark planner. "
            "Use structured fields like models/datasets/seeds instead."
        )

    name = entry["name"] if "name" in entry else "benchmark"
    models = _as_list(entry, "model", "models")
    datasets = _as_list(entry, "dataset", "datasets")
    seeds = entry["seeds"] if "seeds" in entry else None
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Benchmark experiment '{name}' must provide a non-empty 'seeds' list")

    mode = entry["mode"] if "mode" in entry else "train"
    if mode not in {"train", "hpo"}:
        raise ValueError(f"Benchmark experiment '{name}' has unsupported mode '{mode}'")

    run_type = entry["type"] if "type" in entry else "benchmark"
    comment = entry["comment"] if "comment" in entry else name
    overrides = entry["overrides"] if "overrides" in entry else {}
    if not isinstance(overrides, dict):
        raise ValueError(f"Benchmark experiment '{name}' overrides must be a mapping")

    hpo = entry["hpo"] if "hpo" in entry else {}
    if mode == "hpo":
        if not isinstance(hpo, dict):
            raise ValueError(f"Benchmark experiment '{name}' hpo settings must be a mapping")
    elif hpo:
        raise ValueError(f"Benchmark experiment '{name}' may only define hpo settings when mode='hpo'")

    return {
        "name": name,
        "models": models,
        "datasets": datasets,
        "seeds": seeds,
        "mode": mode,
        "type": run_type,
        "comment": comment,
        "overrides": overrides,
        "hpo": hpo,
    }


def _config_overrides_for_run(entry: Dict[str, Any], seed: int) -> Dict[str, Any]:
    overrides = deep_merge_dict({}, entry["overrides"])
    overrides["seed"] = seed
    return overrides


def build_run_id(
    manifest_hash: str,
    model: str,
    dataset: str,
    seed: int,
    mode: str,
    run_type: str,
    comment: str,
    overrides: Dict[str, Any],
) -> str:
    payload = {
        "manifest_hash": manifest_hash,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "type": run_type,
        "comment": comment,
        "overrides": overrides,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def _output_comment(comment: str, seed: int, run_id: str) -> str:
    return f"{comment}.seed{seed}.{run_id[:8]}"


def _path_contract(model: str, dataset: str, mode: str, run_type: str, output_comment: str, overrides: Dict[str, Any], hpo: Dict[str, Any]) -> Dict[str, str]:
    config_dict = deep_merge_dict({}, overrides)
    config_dict["type"] = run_type
    config_dict["comment"] = output_comment
    if mode == "hpo":
        config_dict["smart_hpo"] = True
        optimization = {}
        if "strategy" in hpo:
            optimization["strategy"] = hpo["strategy"]
        if "budget" in hpo:
            optimization["budget"] = hpo["budget"]
        if optimization:
            config_dict["optimization"] = optimization

    config = ConfigManager(model, dataset, config_dict)
    if mode == "hpo":
        # Validate the RESOLVED strategy (hpo block, overrides, or YAML default)
        # so an unsupported one fails before any training runs.
        strategy = config["optimization"]["strategy"]
        if strategy not in _SUMMARIZABLE_HPO_STRATEGIES:
            raise ValueError(
                f"Benchmark mode='hpo' for {model}/{dataset} resolved unsupported HPO "
                f"strategy '{strategy}'. Supported: "
                f"{', '.join(sorted(_SUMMARIZABLE_HPO_STRATEGIES))}. Grid writes its "
                "trial CSV outside hyper_search_dir, so --summarize cannot find it; "
                "run grid searches via main.py --smart_hpo --strategy grid instead."
            )
    return {
        "result_file": config["result_file_name"],
        "log_dir": config["log_dir"],
        "checkpoint_dir": config["checkpoint_dir"],
        "hyper_search_dir": config["hpo_dir"],
    }


def _argv_for_run(
    model: str,
    dataset: str,
    mode: str,
    run_type: str,
    output_comment: str,
    overrides: Dict[str, Any],
    hpo: Dict[str, Any],
) -> List[str]:
    command_parts = [
        sys.executable,
        str((_REPO_ROOT / "main.py").resolve()),
        "--model",
        model,
        "--dataset",
        dataset,
        "--type",
        run_type,
        "--comment",
        output_comment,
    ]
    if mode == "hpo":
        command_parts.append("--smart_hpo")
        if "strategy" in hpo:
            command_parts.extend(["--strategy", str(hpo["strategy"])])
        if "budget" in hpo:
            command_parts.extend(["--hpo_budget", str(hpo["budget"])])
        if "verbose" in hpo and hpo["verbose"]:
            command_parts.append("--verbose")
        if "resume" in hpo and not hpo["resume"]:
            command_parts.append("--no-resume")
    if overrides:
        command_parts.extend(
            [
                "--param_overrides",
                json.dumps(_normalize_jsonable(overrides), sort_keys=True, separators=(",", ":")),
            ]
        )
    return command_parts


def _command_for_run(
    model: str,
    dataset: str,
    mode: str,
    run_type: str,
    output_comment: str,
    overrides: Dict[str, Any],
    hpo: Dict[str, Any],
) -> str:
    argv = _argv_for_run(
        model=model,
        dataset=dataset,
        mode=mode,
        run_type=run_type,
        output_comment=output_comment,
        overrides=overrides,
        hpo=hpo,
    )
    # shlex.quote every token: the always-present compact --param_overrides JSON
    # has no whitespace but contains shell-special characters ({}, ", brace
    # expansion in bash), so the pasted reproduction command must be POSIX-safe.
    return " ".join(shlex.quote(token) for token in argv)


def build_benchmark_plan(spec_path: str | Path) -> Dict[str, Any]:
    spec_path = Path(spec_path)
    spec = load_benchmark_spec(spec_path)
    manifest_name = spec_path.stem
    manifest_hash = build_manifest_hash(spec)
    reporting = normalize_reporting_config(
        spec["reporting"] if "reporting" in spec else None
    )

    runs: List[Dict[str, Any]] = []
    for raw_entry in spec["experiments"]:
        entry = _normalize_experiment_entry(raw_entry)
        for model in entry["models"]:
            for dataset in entry["datasets"]:
                for seed in entry["seeds"]:
                    overrides = _config_overrides_for_run(entry, seed)
                    run_id = build_run_id(
                        manifest_hash=manifest_hash,
                        model=model,
                        dataset=dataset,
                        seed=seed,
                        mode=entry["mode"],
                        run_type=entry["type"],
                        comment=entry["comment"],
                        overrides=overrides,
                    )
                    output_comment = _output_comment(entry["comment"], seed, run_id)
                    outputs = _path_contract(
                        model=model,
                        dataset=dataset,
                        mode=entry["mode"],
                        run_type=entry["type"],
                        output_comment=output_comment,
                        overrides=overrides,
                        hpo=entry["hpo"],
                    )
                    runs.append(
                        {
                            "manifest_name": manifest_name,
                            "manifest_hash": manifest_hash,
                            "experiment_name": entry["name"],
                            "run_id": run_id,
                            "attempt": 0,
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "mode": entry["mode"],
                            "type": entry["type"],
                            "comment": entry["comment"],
                            "output_comment": output_comment,
                            "status": "planned",
                            "return_code": None,
                            "failure_state": None,
                            "started_at": None,
                            "finished_at": None,
                            "command": _command_for_run(
                                model=model,
                                dataset=dataset,
                                mode=entry["mode"],
                                run_type=entry["type"],
                                output_comment=output_comment,
                                overrides=overrides,
                                hpo=entry["hpo"],
                            ),
                            "result_file": outputs["result_file"],
                            "log_dir": outputs["log_dir"],
                            "checkpoint_dir": outputs["checkpoint_dir"],
                            "hyper_search_dir": outputs["hyper_search_dir"],
                            "hpo_json": json.dumps(_normalize_jsonable(entry["hpo"]), sort_keys=True),
                            "overrides_json": json.dumps(_normalize_jsonable(overrides), sort_keys=True),
                        }
                    )

    return {
        "spec_path": str(spec_path.resolve()),
        "manifest_name": manifest_name,
        "manifest_hash": manifest_hash,
        "reporting": reporting,
        "runs": runs,
    }


def _benchmark_output_dir(output_root: str | Path, manifest_name: str, manifest_hash: str) -> Path:
    return Path(output_root) / manifest_name / manifest_hash[:12]


def get_benchmark_paths(output_root: str | Path, manifest_name: str, manifest_hash: str) -> Dict[str, Path]:
    output_dir = _benchmark_output_dir(output_root, manifest_name, manifest_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": output_dir,
        "ledger_jsonl": output_dir / "ledger.jsonl",
        "ledger_csv": output_dir / "ledger.csv",
        "plan_json": output_dir / "plan.json",
    }


def load_benchmark_history(output_root: str | Path, manifest_name: str, manifest_hash: str) -> List[Dict[str, Any]]:
    ledger_jsonl = get_benchmark_paths(output_root, manifest_name, manifest_hash)["ledger_jsonl"]
    if not ledger_jsonl.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in ledger_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest_rows_by_run_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["run_id"]] = row
    return latest


def load_benchmark_latest_rows(
    output_root: str | Path,
    manifest_name: str,
    manifest_hash: str,
) -> Dict[str, Dict[str, Any]]:
    return _latest_rows_by_run_id(
        load_benchmark_history(output_root, manifest_name, manifest_hash)
    )


def _next_attempt(latest_row: Dict[str, Any] | None) -> int:
    if latest_row is None or "attempt" not in latest_row:
        return 1
    return int(latest_row["attempt"]) + 1


def _append_ledger_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as ledger_file:
        ledger_file.write(json.dumps(row, sort_keys=True) + "\n")


def _rewrite_ledger_csv(path: Path, latest_rows: Dict[str, Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_LEDGER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(latest_rows.values())


def _write_plan_json(path: Path, plan: Dict[str, Any]) -> None:
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")


def write_benchmark_ledger(plan: Dict[str, Any], output_root: str | Path) -> Dict[str, str]:
    paths = get_benchmark_paths(output_root, plan["manifest_name"], plan["manifest_hash"])
    history = load_benchmark_history(output_root, plan["manifest_name"], plan["manifest_hash"])

    recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()
    latest_rows = _latest_rows_by_run_id(history)
    for row in plan["runs"]:
        ledger_row = dict(row)
        ledger_row["recorded_at"] = recorded_at
        _append_ledger_row(paths["ledger_jsonl"], ledger_row)
        latest_rows[ledger_row["run_id"]] = ledger_row

    _rewrite_ledger_csv(paths["ledger_csv"], latest_rows)
    _write_plan_json(paths["plan_json"], plan)

    return {
        "output_dir": str(paths["output_dir"]),
        "ledger_jsonl": str(paths["ledger_jsonl"]),
        "ledger_csv": str(paths["ledger_csv"]),
        "plan_json": str(paths["plan_json"]),
    }


def execute_benchmark_plan(
    plan: Dict[str, Any],
    output_root: str | Path,
    resume_enabled: bool = True,
) -> Dict[str, Any]:
    paths = get_benchmark_paths(output_root, plan["manifest_name"], plan["manifest_hash"])
    _write_plan_json(paths["plan_json"], plan)

    history = load_benchmark_history(output_root, plan["manifest_name"], plan["manifest_hash"])
    latest_rows = _latest_rows_by_run_id(history)

    executed_runs = 0
    skipped_runs = 0

    for run in plan["runs"]:
        latest_row = latest_rows.get(run["run_id"])
        if resume_enabled and latest_row is not None and latest_row["status"] == "completed":
            skipped_runs += 1
            continue

        attempt = _next_attempt(latest_row)
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        running_row = dict(run)
        running_row.update(
            {
                "attempt": attempt,
                "status": "running",
                "return_code": None,
                "failure_state": None,
                "recorded_at": started_at,
                "started_at": started_at,
                "finished_at": None,
            }
        )
        _append_ledger_row(paths["ledger_jsonl"], running_row)
        latest_rows[run["run_id"]] = running_row

        argv = _argv_for_run(
            model=run["model"],
            dataset=run["dataset"],
            mode=run["mode"],
            run_type=run["type"],
            output_comment=run["output_comment"],
            overrides=json.loads(run["overrides_json"]),
            hpo=json.loads(run["hpo_json"]),
        )
        completed = subprocess.run(argv, cwd=_REPO_ROOT, shell=False)
        executed_runs += 1

        finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        terminal_row = dict(run)
        terminal_row.update(
            {
                "attempt": attempt,
                "status": "completed" if completed.returncode == 0 else "failed",
                "return_code": completed.returncode,
                "failure_state": None if completed.returncode == 0 else "process_exit",
                "recorded_at": finished_at,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        )
        _append_ledger_row(paths["ledger_jsonl"], terminal_row)
        latest_rows[run["run_id"]] = terminal_row

        if completed.returncode != 0:
            _rewrite_ledger_csv(paths["ledger_csv"], latest_rows)
            raise RuntimeError(
                f"Benchmark run failed for run_id={run['run_id']} with exit code {completed.returncode}"
            )

    _rewrite_ledger_csv(paths["ledger_csv"], latest_rows)
    return {
        "manifest_name": plan["manifest_name"],
        "manifest_hash": plan["manifest_hash"],
        "planned_runs": len(plan["runs"]),
        "executed_runs": executed_runs,
        "skipped_runs": skipped_runs,
        "output_dir": str(paths["output_dir"]),
        "ledger_jsonl": str(paths["ledger_jsonl"]),
        "ledger_csv": str(paths["ledger_csv"]),
        "plan_json": str(paths["plan_json"]),
    }
