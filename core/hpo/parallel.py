# coding: utf-8
"""Single-node parallel HPO execution.

Parallel HPO here means trial-level parallelism: launch independent normal HPO
processes, one visible GPU per process, then merge their trial CSVs. It does not
add a distributed training dependency to models or trainers.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..config import deep_merge_dict
from .parameters import _coerce_numeric, select_best_index


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HPOParallelShard:
    index: int
    gpu_id: str
    budget: int
    comment: str
    command: List[str]
    log_file: Path
    hpo_dir: Path
    checkpoint_dir: Path
    env: Dict[str, str]


_CHILD_CLI_KEYS = {
    "model",
    "dataset",
    "gpu_id",
    "type",
    "comment",
    "smart_hpo",
    "verbose",
}


_CHILD_RUNTIME_KEYS = {
    "device",
    "paths",
    "default_parameters",
    "log_file_name",
    "result_file_name",
    "model_dir",
    "log_dir",
    "hpo_dir",
    "checkpoint_dir",
}


def parse_gpu_ids(raw_gpus: str) -> List[str]:
    gpu_ids = [part.strip() for part in raw_gpus.split(",") if part.strip()]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--hpo_gpus contains duplicate GPU ids: {raw_gpus!r}")
    return gpu_ids


def resolve_parallel_gpu_ids(raw_gpus: str) -> List[str]:
    gpu_ids = parse_gpu_ids(raw_gpus)
    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_ids:
        if gpu_count == 0:
            raise ValueError(
                "--hpo_gpus was set but CUDA is not available to the parent process."
            )
        for gpu_id in gpu_ids:
            if not gpu_id.isdigit():
                raise ValueError(f"--hpo_gpus must contain numeric GPU ids, got {gpu_id!r}")
            if int(gpu_id) >= gpu_count:
                raise ValueError(
                    f"--hpo_gpus contains GPU id {gpu_id}, but only {gpu_count} CUDA GPU(s) "
                    "are visible to the parent process."
                )
        return gpu_ids

    if gpu_count == 0:
        raise ValueError(
            "parallel HPO needs at least one GPU. Set --hpo_gpus, or run on a node "
            "where CUDA GPUs are visible."
        )
    return [str(index) for index in range(gpu_count)]


def split_trial_budget(total_trials: int, shard_count: int) -> List[int]:
    if total_trials <= 0:
        raise ValueError("optimization.budget must be positive for parallel HPO")
    if shard_count <= 0:
        raise ValueError("parallel HPO requires at least one shard")

    active_shards = min(total_trials, shard_count)
    base_budget = total_trials // active_shards
    remainder = total_trials % active_shards
    return [base_budget + (1 if index < remainder else 0) for index in range(active_shards)]


def _copy_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    return copy.deepcopy(value)


def _child_base_overrides(input_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in input_config.items()
        if key not in _CHILD_CLI_KEYS and key not in _CHILD_RUNTIME_KEYS
    }


def _base_sampler_seed(runtime_config: Dict[str, Any]) -> int:
    # Parallel HPO only runs bayesian/tpe (build_parallel_shards rejects other
    # strategies), so the base sampler seed is the explicit sampler_seed if set,
    # else the TPE random_state.
    optimization = _copy_mapping(runtime_config["optimization"], "optimization")
    configured_seed = optimization["sampler_seed"]
    if configured_seed is not None:
        return int(configured_seed)

    return int(runtime_config["tpe"]["random_state"])


def _shared_study_name(
    model_name: str,
    dataset_name: str,
    strategy: str,
    target_metric: str,
    run_type: str,
    comment: str,
) -> str:
    # target_metric is part of the study identity (same contract as the serial
    # study name in optuna_backend.py): resuming after a valid_metric change
    # must start a NEW shared study instead of mixing metric values.
    if not target_metric:
        return f"{model_name}_{dataset_name}_{strategy}_{run_type}_{comment}"
    return f"{model_name}_{dataset_name}_{strategy}_{target_metric}_{run_type}_{comment}"


def _combined_csv_path(
    model_name: str,
    dataset_name: str,
    runtime_config: Dict[str, Any],
    strategy: str,
) -> Path:
    return (
        Path(runtime_config["hpo_dir"]).resolve()
        / f"[{model_name}]-[{dataset_name}]-[{strategy}.{runtime_config['type']}.{runtime_config['comment']}.parallel].csv"
    )


def _load_shared_study(storage_dir: Path, study_name: str):
    journal = storage_dir / "optuna_journal.log"
    if not journal.exists():
        return None

    import optuna
    from optuna.storages.journal import JournalFileBackend

    storage = optuna.storages.JournalStorage(JournalFileBackend(str(journal)))
    # A missing study is a legitimate "nothing to resume" case, so check
    # existence explicitly. Other load failures, such as corrupted journals,
    # should surface to the caller.
    if study_name not in set(optuna.get_all_study_names(storage)):
        return None
    return optuna.load_study(study_name=study_name, storage=storage)


def _completed_trials_in_shared_study(storage_dir: Path, study_name: str) -> int:
    study = _load_shared_study(storage_dir, study_name)
    if study is None:
        return 0
    import optuna

    completed = study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,),
    )
    return len(completed)


def _trial_row_from_optuna_trial(trial: Any, strategy: str) -> Dict[str, Any]:
    params = dict(trial.user_attrs.get("params", trial.params.copy()))
    metrics = trial.user_attrs.get("metrics", {"valid_metrics": {}, "test_metrics": {}})
    status = {
        "COMPLETE": "completed",
        "FAIL": "failed",
        "PRUNED": "pruned",
    }.get(trial.state.name, trial.state.name.lower())
    row: Dict[str, Any] = {}
    row.update(params)
    if isinstance(metrics, dict):
        row.update(metrics)
    row["trial_num"] = trial.number + 1
    row["strategy"] = strategy
    row["duration"] = trial.duration.total_seconds() if trial.duration is not None else 0.0
    row["status"] = status
    row["target_score"] = trial.value if trial.state.name == "COMPLETE" else float("nan")
    for key in (
        "parallel_shard_index",
        "parallel_shard_count",
        "parallel_target_budget",
        "sampler_seed",
        "duplicate_of",
        "target_metric",
    ):
        if key in trial.user_attrs:
            row[key] = trial.user_attrs[key]
    return row


def _attach_shard_columns(combined: pd.DataFrame, shards: List[HPOParallelShard], base_comment: str) -> pd.DataFrame:
    fallback = (
        combined["source_shard_index"]
        if "source_shard_index" in combined
        else pd.Series(-1, index=combined.index)
    )
    if "parallel_shard_index" in combined:
        combined["shard_index"] = pd.to_numeric(
            combined["parallel_shard_index"], errors="coerce"
        ).fillna(fallback).astype(int)
    else:
        combined["shard_index"] = fallback.astype(int)
    shard_comments = {shard.index: shard.comment for shard in shards}
    combined["shard_comment"] = combined["shard_index"].map(
        lambda index: shard_comments.get(index, f"{base_comment}_shard{index:02d}" if index >= 0 else "")
    )
    return combined


def _result_from_combined_frame(
    combined: pd.DataFrame,
    combined_path: Path,
    runtime_config: Dict[str, Any],
    strategy: str,
    shards: List[HPOParallelShard],
) -> Dict[str, Any]:
    completed = combined[combined["status"] == "completed"].copy()
    if completed.empty:
        raise ValueError("Parallel HPO completed without any successful trials")
    completed["target_score_numeric"] = pd.to_numeric(completed["target_score"])
    objective = str(runtime_config["optimization"]["objective"])
    best_index = select_best_index(completed["target_score_numeric"], objective)
    best_row = completed.loc[best_index]
    hyper_parameters = (
        list(runtime_config["hyper_parameters"])
        if "hyper_parameters" in runtime_config
        else []
    )
    best_configuration = {
        param: _coerce_numeric(best_row[param])
        for param in hyper_parameters
        if param in best_row and not pd.isna(best_row[param])
    }

    return {
        "parallel": True,
        "strategy": strategy,
        "target_metric": runtime_config["valid_metric"] if "valid_metric" in runtime_config else "",
        "total_trials": int(len(combined)),
        "successful_trials": int(len(completed)),
        "csv_file": str(combined_path),
        "best_configuration": best_configuration,
        "best_score": float(best_row["target_score_numeric"]),
        "best_trial_num": int(best_row["trial_num"]),
        "best_shard_index": int(best_row["shard_index"]),
        "best_shard_comment": str(best_row["shard_comment"]),
        "shards": [
            {
                "index": shard.index,
                "gpu_id": shard.gpu_id,
                "budget": shard.budget,
                "comment": shard.comment,
                "log_file": str(shard.log_file),
                "hpo_dir": str(shard.hpo_dir),
                "checkpoint_dir": str(shard.checkpoint_dir),
            }
            for shard in shards
        ],
    }


def build_parallel_shards(
    model_name: str,
    dataset_name: str,
    input_config: Dict[str, Any],
    runtime_config: Dict[str, Any],
    strategy: str,
    target_metric: str = "",
    total_trials: int = 0,
    resume: bool = True,
    verbose: bool = False,
) -> List[HPOParallelShard]:
    if strategy not in {"bayesian", "tpe"}:
        raise ValueError(
            "parallel HPO currently supports bayesian and tpe. "
            "Use non-parallel mode for grid/random searches."
        )

    optimization = _copy_mapping(runtime_config["optimization"], "optimization")
    run_type = str(runtime_config["type"])
    base_comment = str(runtime_config["comment"])
    base_hpo_dir = Path(runtime_config["hpo_dir"]).resolve()
    base_checkpoint_dir = Path(runtime_config["checkpoint_dir"]).resolve()
    base_log_dir = Path(runtime_config["log_dir"]).resolve()
    base_sampler_seed = _base_sampler_seed(runtime_config)
    resolved_target_metric = target_metric
    if not resolved_target_metric and "valid_metric" in runtime_config:
        resolved_target_metric = runtime_config["valid_metric"]
    shared_study_name = _shared_study_name(
        model_name, dataset_name, strategy, resolved_target_metric, run_type, base_comment
    )
    shard_offset = (
        int(optimization["parallel_shard_offset"])
        if "parallel_shard_offset" in optimization
        else 0
    )
    global_shard_count = (
        int(optimization["parallel_global_shard_count"] or 0)
        if "parallel_global_shard_count" in optimization
        else 0
    )

    already_completed = (
        _completed_trials_in_shared_study(base_hpo_dir, shared_study_name)
        if resume else 0
    )
    remaining_trials = max(0, total_trials - already_completed)
    if remaining_trials == 0:
        return []

    gpu_ids = resolve_parallel_gpu_ids(str(optimization["parallel_gpus"]))
    budgets = split_trial_budget(remaining_trials, len(gpu_ids))

    eval_final_test = (
        bool(optimization["eval_final_test"])
        if "eval_final_test" in optimization
        else True
    )
    shards = []
    for index, budget in enumerate(budgets):
        global_index = shard_offset + index
        physical_gpu = gpu_ids[index]
        shard_comment = f"{base_comment}_shard{global_index:02d}"
        shard_hpo_dir = base_hpo_dir / "parallel" / run_type / shard_comment
        shard_checkpoint_dir = base_checkpoint_dir / "parallel" / run_type / shard_comment
        shard_log_file = base_log_dir / "parallel" / run_type / f"{shard_comment}.stdout.log"

        child_config = _child_base_overrides(input_config)
        child_optimization = child_config["optimization"] if "optimization" in child_config else {}
        if not isinstance(child_optimization, dict):
            raise ValueError("optimization override must be a dict")
        child_config["optimization"] = deep_merge_dict(
            child_optimization,
            {
                "budget": budget,
                "strategy": strategy,
                "parallel": False,
                "parallel_gpus": "",
                "parallel_dry_run": False,
                "parallel_storage_dir": str(base_hpo_dir),
                "parallel_study_name": shared_study_name,
                "parallel_shard_index": global_index,
                "parallel_shard_count": global_shard_count or len(budgets),
                "parallel_target_budget": total_trials,
                "sampler_seed": base_sampler_seed + global_index,
                "save_model": False,
                "print_model_info": False,
                "eval_final_test": eval_final_test,
                "final_train": {"enabled": False},
            },
        )
        child_config["save_model"] = False
        child_config["hpo_dir"] = str(shard_hpo_dir)
        child_config["checkpoint_dir"] = str(shard_checkpoint_dir)

        command = [
            sys.executable,
            "main.py",
            "--model",
            model_name,
            "--dataset",
            dataset_name,
            "--gpu_id",
            "0",
            "--type",
            run_type,
            "--comment",
            shard_comment,
            "--smart_hpo",
            "--strategy",
            strategy,
            "--hpo_budget",
            str(budget),
            "--param_overrides",
            json.dumps(child_config, sort_keys=True),
        ]
        if not resume:
            command.append("--no-resume")
        if verbose:
            command.append("--verbose")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        shards.append(
            HPOParallelShard(
                index=global_index,
                gpu_id=physical_gpu,
                budget=budget,
                comment=shard_comment,
                command=command,
                log_file=shard_log_file,
                hpo_dir=shard_hpo_dir,
                checkpoint_dir=shard_checkpoint_dir,
                env=env,
            )
        )

    return shards


def _shard_csv_path(shard: HPOParallelShard, model_name: str, dataset_name: str, strategy: str, run_type: str) -> Path:
    return shard.hpo_dir / f"[{model_name}]-[{dataset_name}]-[{strategy}.{run_type}.{shard.comment}].csv"


def _merge_shard_csvs(
    shards: List[HPOParallelShard],
    model_name: str,
    dataset_name: str,
    runtime_config: Dict[str, Any],
    strategy: str,
) -> Dict[str, Any]:
    run_type = str(runtime_config["type"])
    base_comment = str(runtime_config["comment"])
    frames = []

    for shard in shards:
        csv_path = _shard_csv_path(shard, model_name, dataset_name, strategy, run_type)
        if not csv_path.exists():
            raise FileNotFoundError(f"Parallel HPO shard did not produce CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame["source_shard_index"] = shard.index
        frame["shard_csv"] = str(csv_path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    # Shards share one Optuna study, so every shard CSV lists all trials —
    # de-duplicate by trial number before reporting/merging.
    combined = combined.drop_duplicates(subset=["trial_num"], keep="first").reset_index(drop=True)
    combined = _attach_shard_columns(combined, shards, base_comment)
    combined_path = _combined_csv_path(model_name, dataset_name, runtime_config, strategy)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)

    return _result_from_combined_frame(combined, combined_path, runtime_config, strategy, shards)


def _merge_shared_study(
    model_name: str,
    dataset_name: str,
    runtime_config: Dict[str, Any],
    strategy: str,
    target_metric: str,
) -> Dict[str, Any]:
    run_type = str(runtime_config["type"])
    base_comment = str(runtime_config["comment"])
    study_name = _shared_study_name(
        model_name, dataset_name, strategy, target_metric, run_type, base_comment
    )
    study = _load_shared_study(Path(runtime_config["hpo_dir"]).resolve(), study_name)
    if study is None:
        raise FileNotFoundError(f"Shared Optuna study not found: {study_name}")

    trials = [trial for trial in study.trials if trial.state.is_finished()]
    combined = pd.DataFrame([_trial_row_from_optuna_trial(trial, strategy) for trial in trials])
    if combined.empty:
        raise ValueError("Shared Optuna study has no finished trials to merge")
    combined = _attach_shard_columns(combined, [], base_comment)
    combined_path = _combined_csv_path(model_name, dataset_name, runtime_config, strategy)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    return _result_from_combined_frame(combined, combined_path, runtime_config, strategy, [])


def run_parallel_hpo(
    model_name: str,
    dataset_name: str,
    input_config: Dict[str, Any],
    runtime_config: Dict[str, Any],
    strategy: str,
    target_metric: str,
    resume: bool,
    verbose: bool,
) -> Dict[str, Any]:
    optimization = _copy_mapping(runtime_config["optimization"], "optimization")
    total_trials = int(optimization["budget"])
    dry_run = bool(optimization["parallel_dry_run"])
    shards = build_parallel_shards(
        model_name=model_name,
        dataset_name=dataset_name,
        input_config=input_config,
        runtime_config=runtime_config,
        strategy=strategy,
        target_metric=target_metric,
        total_trials=total_trials,
        resume=resume,
        verbose=verbose,
    )

    print(
        f"Parallel HPO: model={model_name} dataset={dataset_name} "
        f"strategy={strategy} target={target_metric} total_trials={total_trials} "
        f"shards={len(shards)}"
    )
    for shard in shards:
        print(
            f"[shard {shard.index:02d}] gpu={shard.gpu_id} trials={shard.budget} "
            f"comment={shard.comment}"
        )
        print(f"  log: {shard.log_file}")
        print(f"  hpo: {shard.hpo_dir}")

    if dry_run:
        return {
            "parallel": True,
            "dry_run": True,
            "total_trials": total_trials,
            "shards": [
                {
                    "index": shard.index,
                    "gpu_id": shard.gpu_id,
                    "budget": shard.budget,
                    "comment": shard.comment,
                    "log_file": str(shard.log_file),
                    "hpo_dir": str(shard.hpo_dir),
                    "checkpoint_dir": str(shard.checkpoint_dir),
                }
                for shard in shards
            ],
        }
    if not shards:
        print("Parallel HPO: target budget already satisfied in the shared Optuna study.")
        return _merge_shared_study(
            model_name, dataset_name, runtime_config, strategy, target_metric
        )

    # On a fresh (--no-resume) parallel run, reset the shared study journal so
    # shards start from the requested budget and study state.
    if not resume:
        journal = Path(runtime_config["hpo_dir"]).resolve() / "optuna_journal.log"
        if journal.exists():
            journal.unlink()

    running = []
    for shard in shards:
        shard.log_file.parent.mkdir(parents=True, exist_ok=True)
        shard.hpo_dir.mkdir(parents=True, exist_ok=True)
        shard.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_handle = shard.log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(
            shard.command,
            cwd=REPO_ROOT,
            env=shard.env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        running.append((shard, process, log_handle))

    # Poll all shards; on the first failure stop the survivors immediately
    # rather than waiting for slow shards to finish.
    failed = []
    pending = list(running)
    while pending and not failed:
        still_running = []
        for shard, process, log_handle in pending:
            code = process.poll()
            if code is None:
                still_running.append((shard, process, log_handle))
                continue
            log_handle.close()
            print(f"[shard {shard.index:02d}] finished return_code={code}")
            if code != 0:
                failed.append((shard.index, code, str(shard.log_file)))
        pending = still_running
        if pending and not failed:
            time.sleep(1.0)

    # Terminate any survivors on failure; close every still-open log handle.
    for shard, process, log_handle in pending:
        process.terminate()
        process.wait()
        log_handle.close()

    if failed:
        raise RuntimeError(f"Parallel HPO shard failures: {failed}")

    return _merge_shard_csvs(
        shards=shards,
        model_name=model_name,
        dataset_name=dataset_name,
        runtime_config=runtime_config,
        strategy=strategy,
    )
