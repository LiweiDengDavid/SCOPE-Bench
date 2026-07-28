# coding: utf-8
"""Optuna-backed HPO helpers — uses RDB storage for automatic resume."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


from .parameters import _coerce_numeric, is_better, worst_score


def suggest_parameters(manager, trial) -> Dict[str, Any]:
    """Convert parameter_space config to Optuna suggestions."""
    params = {}
    hyper_params = manager.base_config["hyper_parameters"]
    parameter_space = manager.base_config["parameter_space"]

    def _coerce_list(lst: List[Any]) -> List[Any]:
        return [_coerce_numeric(v) for v in lst]

    for param_name in hyper_params:
        if param_name in parameter_space:
            param_config = parameter_space[param_name]
            param_type = param_config["type"]

            if param_type == "choice":
                values = param_config["values"]
                if not values:
                    raise ValueError(
                        f"Parameter '{param_name}' of type 'choice' must define 'values'"
                    )
                values = _coerce_list(values) if isinstance(values, list) else values
                params[param_name] = trial.suggest_categorical(param_name, values)
            elif param_type == "uniform":
                low, high = _coerce_numeric(param_config["low"]), _coerce_numeric(param_config["high"])
                params[param_name] = trial.suggest_float(param_name, low, high)
            elif param_type in ("logscale", "loguniform"):
                low, high = _coerce_numeric(param_config["low"]), _coerce_numeric(param_config["high"])
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_type == "int":
                low, high = int(_coerce_numeric(param_config["low"])), int(_coerce_numeric(param_config["high"]))
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == "logint":
                low, high = int(_coerce_numeric(param_config["low"])), int(_coerce_numeric(param_config["high"]))
                params[param_name] = trial.suggest_int(param_name, low, high, log=True)
            else:
                raise ValueError(
                    f"Unsupported parameter type '{param_type}' for '{param_name}'. "
                    "Expected one of: choice, uniform, loguniform, logscale, int, logint."
                )
        else:
            base_value = manager.base_config[param_name]
            if isinstance(base_value, list):
                params[param_name] = trial.suggest_categorical(param_name, _coerce_list(base_value))
            else:
                # Fixed (non-searched) parameter: keep it out of the Optuna
                # search space so importance analysis is not polluted by a
                # constant single-value distribution.
                params[param_name] = _coerce_numeric(base_value)

    return params


def _build_storage(manager):
    """Return a file-based Optuna JournalStorage for study persistence.

    JournalStorage uses file locking and is safe for concurrent multi-process
    writers — required for parallel-HPO shards that share one study. SQLite is
    not (it raises "database is locked" under concurrent writes).
    """
    import optuna
    from optuna.storages.journal import JournalFileBackend

    parallel_dir = manager.base_config["optimization"]["parallel_storage_dir"]
    storage_dir = Path(parallel_dir) if parallel_dir else manager._hpo_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    return optuna.storages.JournalStorage(
        JournalFileBackend(str(storage_dir / "optuna_journal.log"))
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _parallel_shard_index(base_config: Dict[str, Any]) -> int:
    optimization = base_config["optimization"]
    return int(optimization["parallel_shard_index"])


def _resolve_sampler_seed(strategy: str, base_config: Dict[str, Any]) -> int:
    optimization = base_config["optimization"]
    configured_seed = optimization["sampler_seed"]
    if configured_seed is not None:
        return int(configured_seed)

    # engine.py routes "random" to enumeration; keep this branch so the seed
    # resolver remains well-defined for direct helper tests.
    if strategy == "random":
        return int(base_config["seed"]) + _parallel_shard_index(base_config)

    return int(base_config["tpe"]["random_state"]) + _parallel_shard_index(base_config)


def _resolve_tpe_constant_liar(base_config: Dict[str, Any]) -> bool:
    configured = base_config["tpe"]["constant_liar"]
    if isinstance(configured, str) and configured.strip().lower() == "auto":
        return bool(base_config["optimization"]["parallel_storage_dir"])
    return _as_bool(configured)


def _resolve_duplicate_guard(optimization: Dict[str, Any]) -> bool:
    configured = optimization["duplicate_guard"]
    if isinstance(configured, str) and configured.strip().lower() == "auto":
        return bool(optimization["parallel_storage_dir"])
    return _as_bool(configured, default=False)


def _build_sampler(optuna, strategy: str, base_config: Dict[str, Any]):
    """Build a deterministic Optuna sampler from the explicit config seed.

    All TPE sampler hyperparameters live in the `tpe:` block of
    configs/overall.yaml (single source of truth); they are read directly so a
    missing key raises instead of falling back to a Python-literal default.
    """
    seed = _resolve_sampler_seed(strategy, base_config)
    # engine.py routes "random" to enumeration; keep this branch for direct
    # sampler-construction tests.
    if strategy == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    tpe_config = base_config["tpe"]
    # Optuna's gamma is a Callable[[n_trials], n_good] (default: ceil(0.1*n)
    # capped at 25). Honor the configured fraction and cap so both the tpe.gamma
    # and tpe.gamma_cap YAML values drive the search instead of Python literals.
    gamma_fraction = float(tpe_config["gamma"])
    gamma_cap = int(tpe_config["gamma_cap"])
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(tpe_config["n_startup_jobs"]),
        n_ei_candidates=int(tpe_config["n_ei_candidates"]),
        prior_weight=float(tpe_config["prior_weight"]),
        gamma=lambda n_trials: min(math.ceil(gamma_fraction * n_trials), gamma_cap),
        constant_liar=_resolve_tpe_constant_liar(base_config),
    )


def _canonical_params(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def _find_duplicate_trial(
    study,
    current_trial_number: int,
    params: Dict[str, Any],
    states,
    suggested_params: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    current_key = _canonical_params(params)
    suggested_key = _canonical_params(suggested_params) if suggested_params is not None else None
    for existing in reversed(study.get_trials(deepcopy=False, states=states)):
        if existing.number >= current_trial_number:
            continue
        existing_params = existing.user_attrs.get("params") if hasattr(existing, "user_attrs") else None
        if existing_params is not None and _canonical_params(dict(existing_params)) == current_key:
            return int(existing.number)
        if existing_params is None and suggested_key is not None and _canonical_params(dict(existing.params)) == suggested_key:
            return int(existing.number)
    return None


def _complete_trial_count(optuna, study) -> int:
    return len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))


def run_optuna_optimization(
    manager,
    strategy: str,
    target_metric: str,
    max_trials: int,
    resume: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Run optimization using Optuna with RDB storage for persistence."""
    import optuna
    from optuna import logging as opt_logging

    opt_logging.set_verbosity(opt_logging.WARNING)

    sampler = _build_sampler(optuna, strategy, manager.base_config)

    exp_type = manager.base_config["type"]
    comment = manager.base_config["comment"]
    optimization = manager.base_config["optimization"]
    # Parallel-HPO shards share one persistent study (set by the parent);
    # a non-parallel run persists only when resuming.
    is_parallel_shard = bool(optimization["parallel_storage_dir"])
    if optimization["parallel_study_name"]:
        study_name = optimization["parallel_study_name"]
    else:
        # target_metric is part of the study identity: resuming after a
        # valid_metric change must start a NEW study instead of silently
        # ranking old-metric and new-metric trial values against each other.
        study_name = (
            f"{manager.model_name}_{manager.dataset_name}_{strategy}_"
            f"{target_metric}_{exp_type}_{comment}"
        )

    sampler_seed = _resolve_sampler_seed(strategy, manager.base_config)
    duplicate_guard = _resolve_duplicate_guard(optimization)
    # RandomSampler ignores constant_liar, so report False whenever this helper
    # is called directly with strategy="random".
    constant_liar = _resolve_tpe_constant_liar(manager.base_config) if strategy != "random" else False
    manager.logger.info(
        "Optuna sampler: strategy=%s seed=%s constant_liar=%s duplicate_guard=%s "
        "parallel_shard=%s/%s study=%s",
        strategy,
        sampler_seed,
        constant_liar,
        duplicate_guard,
        optimization["parallel_shard_index"],
        optimization["parallel_shard_count"],
        study_name,
    )

    direction = optimization["objective"]
    use_storage = resume or is_parallel_shard
    storage = _build_storage(manager) if use_storage else None
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage,
        load_if_exists=use_storage,
    )

    best_params: Dict[str, Any] = {}
    best_score = worst_score(direction)
    best_metrics: Dict[str, Any] = {}
    best_trial_num = 0

    # Restore best from existing trials if resuming
    already_completed = _complete_trial_count(optuna, study)
    if already_completed > 0:
        best_trial = study.best_trial
        # user_attrs["params"] holds the full runtime config; best_trial.params
        # holds only Optuna-suggested values.
        best_params = dict(best_trial.user_attrs.get("params", best_trial.params))
        best_score = best_trial.value
        best_metrics = best_trial.user_attrs["metrics"]
        best_trial_num = best_trial.number + 1
        manager.best_score = best_score
        manager.best_trial_num = best_trial_num
        manager.logger.info(
            f"Resuming: {already_completed} trials completed, best score={best_score:.4f}"
        )

    if is_parallel_shard:
        target_budget = int(optimization["parallel_target_budget"] or max_trials)
        remaining_trials = min(max_trials, max(0, target_budget - already_completed))
    else:
        target_budget = max_trials
        remaining_trials = max(0, max_trials - already_completed)

    _trial_counter = 0
    _trained_counter = 0
    _duplicate_counter = 0

    def objective(trial):
        nonlocal best_params, best_score, best_metrics, best_trial_num
        nonlocal _trial_counter, _trained_counter, _duplicate_counter
        _trial_counter += 1

        params = suggest_parameters(manager, trial)
        trial_num = int(trial.number) + 1
        trial.set_user_attr("params", params)
        # Recorded per trial so CSV rows rebuilt from the study carry the
        # target_metric column (engine._trial_result_from_optuna_trial).
        trial.set_user_attr("target_metric", target_metric)
        if is_parallel_shard:
            trial.set_user_attr("parallel_shard_index", int(optimization["parallel_shard_index"]))
            trial.set_user_attr("parallel_shard_count", int(optimization["parallel_shard_count"]))
            trial.set_user_attr("parallel_target_budget", target_budget)
            trial.set_user_attr("sampler_seed", sampler_seed)
        if duplicate_guard:
            duplicate_of = _find_duplicate_trial(
                trial.study,
                int(trial.number),
                params,
                (
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.RUNNING,
                    optuna.trial.TrialState.WAITING,
                ),
                suggested_params=dict(trial.params),
            )
            if duplicate_of is not None:
                _duplicate_counter += 1
                trial.set_user_attr("duplicate_of", duplicate_of)
                manager.logger.info(
                    "Trial %s - Duplicate params of trial %s; pruning before training",
                    trial_num,
                    duplicate_of + 1,
                )
                raise optuna.TrialPruned(f"duplicate params of trial {duplicate_of + 1}")

        trial_info = (
            f"{manager.model_name} on {manager.dataset_name}: "
            f"{strategy.upper()} [Trial: {trial_num}/{max_trials}]"
        )
        result = manager._run_single_trial(
            params, trial_info, target_metric, verbose,
            show_model_info=(_trained_counter == 0),
            save_if_best=True,
        )
        target_score = result["score"]
        all_metrics = result["metrics"]

        manager._log_trial(trial_num, params, all_metrics)

        if is_better(target_score, best_score, direction):
            best_params = params.copy()
            best_score = target_score
            best_metrics = all_metrics.copy()
            best_trial_num = trial_num
            manager.best_score = best_score
            manager.best_trial_num = best_trial_num
            manager._log_new_best(trial_num, best_params, best_metrics)

        manager._log_best(trial_num, best_params, best_metrics)

        trial.set_user_attr("metrics", all_metrics)

        manager.trial_results.append({
            "trial": trial_num, "params": params.copy(),
            "score": target_score, "metrics": all_metrics.copy(),
        })
        _trained_counter += 1

        return target_score

    if remaining_trials > 0:
        manager.logger.info(f"Running {remaining_trials} trials ({already_completed} already done)")
        attempts_multiplier = int(optimization["max_attempts_multiplier"])
        attempts_floor = int(optimization["max_attempts_floor"])
        max_attempts = max(remaining_trials * attempts_multiplier, remaining_trials + attempts_floor)
        while _trained_counter < remaining_trials and _trial_counter < max_attempts:
            if is_parallel_shard and _complete_trial_count(optuna, study) >= target_budget:
                break
            study.optimize(objective, n_trials=1)
        if _trained_counter < remaining_trials:
            manager.logger.warning(
                "Stopped after %s Optuna attempts with %s/%s completed trainings "
                "(duplicates pruned: %s). The search space may be exhausted.",
                _trial_counter,
                _trained_counter,
                remaining_trials,
                _duplicate_counter,
            )
    else:
        manager.logger.info(f"All {max_trials} trials already completed")

    # Extract final best
    has_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0
    if has_completed:
        final_best = study.best_trial
        # Read the full runtime config from user_attrs, not only Optuna-suggested values.
        best_params = dict(final_best.user_attrs.get("params", final_best.params))
        best_score = final_best.value
        best_metrics = final_best.user_attrs["metrics"]
        best_trial_num = final_best.number + 1

    importance_analysis = {}
    if has_completed:
        importance_analysis = manager._analyze_importance(study, target_metric)

    return {
        "best_configuration": best_params,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_trial_num": best_trial_num,
        "target_metric": target_metric,
        "strategy": strategy,
        "optuna_study": study,
        "hyperparameter_importance": importance_analysis,
    }
