# coding: utf-8
"""
Unified Hyperparameter Optimization Interface
============================================

Strategies: grid, random, bayesian, tpe.
Optuna-based strategies use RDB storage for automatic resume.
"""

import gc
import json
import time
import logging
import copy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..config import _FLATTEN_GROUPS, deep_merge_dict
from ..training.core import train_single
from ..training.environment import prepare_env, setup_hpo_environment
from ..utils.metrics import extract_target_metric
from .optuna_backend import (
    run_optuna_optimization as _run_optuna_optimization_backend,
)
from .parameters import _coerce_numeric, generate_grid_parameter_dicts, is_better, select_best_index, worst_score


_MATERIALIZED_CONFIG_KEYS = {
    "device",
    "paths",
    "default_parameters",
    "log_file_name",
    "result_file_name",
    "model_dir",
}


def _strip_materialized_config(config: Dict[str, Any]) -> Dict[str, Any]:
    skip = _MATERIALIZED_CONFIG_KEYS | set(_FLATTEN_GROUPS)
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in skip
    }


def _parse_metrics_cell(payload: Any) -> Dict[str, Any]:
    """Parse a metrics dict that pandas stringified into a resumed CSV cell."""
    text = str(payload).strip()
    if not text.startswith("{"):
        return {}
    return json.loads(text.replace("'", '"'))


def _expand_dotted_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fold dotted param keys (e.g. "loss_weights.ssl_loss") into nested dicts.

    Suggested params may be keyed by literal dotted strings for nested model
    config. A flat update would land them as top-level keys the model never
    reads (the model reads config["loss_weights"]["ssl_loss"]), so split each
    dotted key and deep-merge it into the nested structure.

    Both branches deep-merge: a non-dotted dict parent (e.g. ``"a": {"c": 2}``)
    must merge with a dotted child (``"a.b": 1``) rather than overwrite it,
    regardless of iteration order.
    """
    expanded: Dict[str, Any] = {}
    for key, value in params.items():
        if "." in key:
            nested: Dict[str, Any] = value
            for part in reversed(key.split(".")):
                nested = {part: nested}
        else:
            nested = {key: value}
        expanded = deep_merge_dict(expanded, nested)
    return expanded


def _disable_trial_export(config: Dict[str, Any]) -> Dict[str, Any]:
    if "export" in config:
        export_config = config["export"]
        if not isinstance(export_config, dict):
            raise ValueError("output.export must be a dict.")
        config["export"] = deep_merge_dict(export_config, {"enabled": False})
    return config


class UnifiedHPOManager:
    """Unified manager for hyperparameter optimization."""

    def __init__(self, model_name: str, dataset_name: str, base_config: Dict[str, Any]):
        self.model_name = model_name
        self.dataset_name = dataset_name

        from ..config import ConfigManager
        full_config = ConfigManager(model_name, dataset_name, base_config)
        self.base_config = dict(full_config)
        self.trial_base_config = _strip_materialized_config(self.base_config)

        self.logger = logging.getLogger("nexusrec")
        self.objective = self.base_config["optimization"]["objective"]
        self.trial_results: List[Dict[str, Any]] = []
        self.best_score = worst_score(self.objective)
        self.best_config: Dict[str, Any] = {}
        self.best_metrics: Dict[str, Any] = {}
        self.best_trial_num = 0
        self._existing_trial_rows: List[Dict[str, Any]] = []

        # Cached data loaders — populated once, reused across all trials
        self._cached_data = None

    def _get_budget(self) -> int:
        return int(self.base_config["optimization"]["budget"])

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def _ensure_cached_data(self, verbose: bool) -> None:
        """Load data once and cache for all trials."""
        if self._cached_data is not None:
            return
        log_config = self.trial_base_config.copy()
        log_config["type"] = "hpo"
        config, train_data, valid_data, test_data = prepare_env(
            self.model_name, self.dataset_name, log_config,
            setup_logging=verbose,
        )
        self._cached_data = (train_data, valid_data, test_data)

    def run_optimization(
        self,
        strategy: str = "grid",
        target_metric: str = None,
        max_trials: int = None,
        resume: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if target_metric is None:
            target_metric = self.base_config["valid_metric"]

        self.logger.info(
            f"Using {strategy.upper()} strategy for HPO"
        )
        self.logger.info(
            f"Starting {strategy.upper()} HPO for {self.model_name} on {self.dataset_name}"
        )

        self._ensure_cached_data(verbose)

        if strategy in ("bayesian", "tpe"):
            if max_trials is None:
                max_trials = self._get_budget()
            return self._run_optuna_path(strategy, target_metric, max_trials, resume, verbose)

        return self._run_enumeration_path(strategy, target_metric, max_trials, resume, verbose)

    # ------------------------------------------------------------------
    # Optuna path (bayesian / tpe) — uses RDB storage for persistence
    # ------------------------------------------------------------------

    def _run_optuna_path(
        self, strategy: str, target_metric: str, max_trials: int, resume: bool, verbose: bool,
    ) -> Dict[str, Any]:
        optuna_results = _run_optuna_optimization_backend(
            self, strategy, target_metric, max_trials, resume, verbose,
        )

        self.best_config = optuna_results["best_configuration"]
        self.best_score = optuna_results["best_score"]
        self.best_metrics = optuna_results["best_metrics"]
        self.best_trial_num = optuna_results["best_trial_num"]

        optuna_study = optuna_results["optuna_study"]
        if optuna_study:
            self.trial_results = []
            for trial in optuna_study.trials:
                if trial.state.is_finished():
                    self.trial_results.append(self._trial_result_from_optuna_trial(trial))

        csv_file = self._save_csv(self.trial_results, strategy)
        self._display_final_results(strategy, target_metric, max_trials, csv_file)

        return {
            "best_configuration": self.best_config,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "target_metric": target_metric,
            "strategy": strategy,
            "best_trial_num": self.best_trial_num,
            "total_trials": len(self.trial_results),
            "trial_history": self.trial_results,
            "csv_file": csv_file,
        }

    # ------------------------------------------------------------------
    # Enumeration path (grid / random) — stateless, skips via CSV
    # ------------------------------------------------------------------

    def _run_enumeration_path(
        self, strategy: str, target_metric: str, max_trials: int, resume: bool, verbose: bool,
    ) -> Dict[str, Any]:
        csv_file = self._csv_path(strategy)
        if resume and csv_file.exists():
            self._restore_enumeration_best_from_csv(csv_file)

        if strategy == "grid":
            existing_csv = str(csv_file) if resume and csv_file.exists() else None
            param_combinations = generate_grid_parameter_dicts(
                self.base_config, existing_results_file=existing_csv,
            )
            self.logger.info(f"Grid Search: {len(param_combinations)} combinations remaining")
        elif strategy == "random":
            if max_trials is None:
                max_trials = self._get_budget()
            # max_trials is a total budget. Draw the full deterministic sequence
            # and slice off completed trials so resume continues the same search.
            already_completed = len(self._existing_trial_rows)
            from .parameters import ParameterGenerator
            all_combinations = ParameterGenerator(
                self.model_name,
                self.base_config,
            ).generate_random_combinations(max_trials)
            param_combinations = all_combinations[already_completed:]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}. Available: grid, random, bayesian, tpe")

        total_trials = len(param_combinations)
        if total_trials == 0:
            self.logger.info("All combinations already completed.")
            self._display_final_results(strategy, target_metric, 0)
            result = self._result_dict()
            result["target_metric"] = target_metric
            result["strategy"] = strategy
            result["csv_file"] = str(csv_file)
            return result

        for trial_num, trial_params in enumerate(param_combinations, 1):
            trial_info = (
                f"{self.model_name} on {self.dataset_name}: "
                f"{strategy.upper()} [Trial: {trial_num}/{total_trials}]"
            )
            trial_result = self._run_single_trial(trial_params, trial_info, target_metric, verbose, save_if_best=True)
            trial_result["trial"] = len(self._existing_trial_rows) + trial_num
            if is_better(trial_result["score"], self.best_score, self.objective):
                self.best_score = trial_result["score"]
                self.best_config = trial_params.copy()
                self.best_metrics = trial_result["metrics"].copy()
                self.best_trial_num = trial_result["trial"]
                self._display_new_best(trial_result)
            self.trial_results.append(trial_result)
            # Persist after EVERY trial so a mid-run kill (walltime / Ctrl-C /
            # OOM) keeps completed rows on disk, matching the per-trial
            # persistence of the optuna journal path. _save_csv rewrites the
            # full frame (existing rows + all results so far), so this is safe
            # to repeat.
            saved_csv = self._save_csv(self.trial_results, strategy)
            self._log_best(trial_num, self.best_config, self.best_metrics)

        self._display_final_results(strategy, target_metric, total_trials, saved_csv)

        return {
            "best_configuration": self.best_config,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "target_metric": target_metric,
            "strategy": strategy,
            "best_trial_num": self.best_trial_num,
            "total_trials": len(self.trial_results),
            "trial_history": self.trial_results,
            "csv_file": saved_csv,
        }

    def _result_dict(self) -> Dict[str, Any]:
        return {
            "best_configuration": self.best_config,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "total_trials": 0,
            "best_trial_num": self.best_trial_num,
        }

    # ------------------------------------------------------------------
    # Trial execution
    # ------------------------------------------------------------------

    def _run_single_trial(
        self,
        params: Dict[str, Any],
        trial_info: str,
        target_metric: str,
        verbose: bool,
        show_model_info: bool = True,
        save_if_best: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Build config with trial-specific hyperparameters (no data reload)
        from ..config import ConfigManager

        if not hasattr(self, "trial_base_config"):
            self.trial_base_config = _strip_materialized_config(self.base_config)
        trial_config = deep_merge_dict(self.trial_base_config, _expand_dotted_params(params))
        trial_config = _disable_trial_export(trial_config)
        trial_config["type"] = "hpo_trial"
        trial_config["comment"] = f"trial_{int(time.time())}"
        
        self.logger.info("="*40 + f" {trial_info} " + "="*40)

        config = ConfigManager(
            self.model_name, self.dataset_name, trial_config,
            trial_info=trial_info,
        )

        setup_hpo_environment(config)
        config["print_model_info"] = show_model_info
        save_trial_checkpoint = save_if_best and config["save_model"]

        # Reuse cached data loaders
        train_data, valid_data, test_data = self._cached_data

        test_result, valid_result, trainer = train_single(
            config, train_data, valid_data, test_data,
            return_trainer=save_trial_checkpoint,
        )
        score = extract_target_metric(valid_result, target_metric)

        # Save the trial checkpoint while the trainer still owns the best state.
        if save_trial_checkpoint and trainer is not None and is_better(score, self.best_score, self.objective):
            trainer.save_checkpoint(self._checkpoint_dir())

        del trainer

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if valid_result is None:
            valid_result = {}
        if test_result is None:
            test_result = {}

        return {
            "params": params.copy(),
            "metrics": {"valid_metrics": valid_result, "test_metrics": test_result},
            "score": score,
            "target_metric": target_metric,
            "duration": time.time() - start_time,
            "status": "completed",
        }

    # ------------------------------------------------------------------
    # Results I/O (replaces HPOResultManager)
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_trial_metrics() -> Dict[str, Dict[str, Any]]:
        return {"valid_metrics": {}, "test_metrics": {}}

    def _trial_result_from_optuna_trial(self, trial: Any) -> Dict[str, Any]:
        trial_state = trial.state.name
        duration_seconds = 0.0
        if trial.duration is not None:
            duration_seconds = trial.duration.total_seconds()

        def _with_trial_attrs(row: Dict[str, Any]) -> Dict[str, Any]:
            # target_metric is set by the optuna objective (optuna_backend.py)
            # so rebuilt rows carry the same column enumeration rows do.
            for key in (
                "target_metric",
                "parallel_shard_index",
                "parallel_shard_count",
                "parallel_target_budget",
                "sampler_seed",
                "duplicate_of",
            ):
                if key in trial.user_attrs:
                    row[key] = trial.user_attrs[key]
            return row

        if trial_state == "COMPLETE":
            return _with_trial_attrs({
                "trial": trial.number + 1,
                "params": dict(trial.user_attrs["params"]),
                "score": trial.value,
                "metrics": trial.user_attrs["metrics"],
                "duration": duration_seconds,
                "status": "completed",
            })

        if trial_state == "FAIL":
            return _with_trial_attrs({
                "trial": trial.number + 1,
                "params": dict(trial.user_attrs.get("params", trial.params.copy())),
                "score": float("nan"),
                "metrics": self._empty_trial_metrics(),
                "duration": duration_seconds,
                "status": "failed",
            })

        if trial_state == "PRUNED":
            return _with_trial_attrs({
                "trial": trial.number + 1,
                "params": dict(trial.user_attrs.get("params", trial.params.copy())),
                "score": float("nan"),
                "metrics": self._empty_trial_metrics(),
                "duration": duration_seconds,
                "status": "pruned",
            })

        raise ValueError(f"Unsupported finished Optuna trial state: {trial_state}")

    def _restore_enumeration_best_from_csv(self, csv_path: Path) -> None:
        rows = pd.read_csv(csv_path).to_dict("records")
        self._existing_trial_rows = rows
        if not rows:
            return

        frame = pd.DataFrame(rows)
        completed = frame[frame["status"] == "completed"].copy()
        if completed.empty:
            return

        completed["target_score_numeric"] = pd.to_numeric(completed["target_score"])
        best_index = select_best_index(completed["target_score_numeric"], self.objective)
        best_row = completed.loc[best_index]
        self.best_score = float(best_row["target_score_numeric"])
        self.best_trial_num = int(best_row["trial_num"])
        self.best_config = {
            param: _coerce_numeric(best_row[param])
            for param in self.base_config["hyper_parameters"]
            if param in best_row
        }
        self.best_metrics = {
            "valid_metrics": _parse_metrics_cell(best_row["valid_metrics"])
            if "valid_metrics" in best_row else {},
            "test_metrics": _parse_metrics_cell(best_row["test_metrics"])
            if "test_metrics" in best_row else {},
        }
        self.logger.info(
            "Resuming enumeration HPO: %d existing rows, best score=%.4f",
            len(rows),
            self.best_score,
        )

    def _checkpoint_dir(self, strategy: str = None) -> Path:
        """Return the checkpoint directory for the best HPO trial."""
        if strategy is None:
            if "optimization" in self.base_config:
                strategy = self.base_config["optimization"]["strategy"]
            else:
                strategy = "hpo"
        exp_type = self.base_config["type"]
        comment = self.base_config["comment"]
        return Path(self.base_config["checkpoint_dir"]) / "hpo" / strategy / exp_type / comment

    def _hpo_dir(self) -> Path:
        """Return the HPO output directory from config."""
        return Path(self.base_config["hpo_dir"])

    def _csv_path(self, strategy: str) -> Path:
        exp_type = self.base_config["type"]
        comment = self.base_config["comment"]
        strategy_name = strategy if strategy != "grid" else "experiment"
        filename = f"[{self.model_name}]-[{self.dataset_name}]-[{strategy_name}.{exp_type}.{comment}].csv"
        if strategy == "grid":
            return Path(self.base_config["paths"]["save"]) / filename
        return self._hpo_dir() / filename

    def _save_csv(self, trial_results: List[Dict[str, Any]], strategy: str) -> str:
        existing_rows = []
        if hasattr(self, "_existing_trial_rows"):
            existing_rows = [dict(row) for row in self._existing_trial_rows]
        if not trial_results and not existing_rows:
            return ""
        rows = existing_rows
        for trial in trial_results:
            row = {}
            row.update(trial["params"])
            row.update(trial["metrics"])
            for key, value in trial.items():
                if key not in {"trial", "params", "metrics", "score", "duration", "status"}:
                    row[key] = value
            row["trial_num"] = trial["trial"]
            row["strategy"] = strategy
            row["duration"] = trial["duration"]
            row["status"] = trial["status"]
            row["target_score"] = trial["score"]
            rows.append(row)

        filepath = self._csv_path(strategy)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(filepath, index=False)
        self.logger.info(f"{strategy.upper()} HPO results saved to: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_params(params: Dict[str, Any]) -> str:
        items = [f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()]
        return " | ".join(items) if items else "No params"

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        if isinstance(metrics, dict) and "valid_metrics" in metrics:
            parts = []
            for label, key in [("Valid", "valid_metrics"), ("Test", "test_metrics")]:
                sub = metrics[key]
                if sub:
                    items = [f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in sub.items()]
                    parts.append(f"{label}: {' | '.join(items)}")
            return "\n  ".join(parts) if parts else "No metrics"
        if metrics is None:
            return "No metrics"
        items = [f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items()]
        return " | ".join(items) if items else "No metrics"

    def _log_trial(self, trial_num, params, metrics):
        self.logger.info("Trial %s - Params: %s", trial_num, self._format_params(params))
        self.logger.info("Trial %s - Results: \n %s", trial_num, self._format_metrics(metrics))

    def _log_new_best(self, trial_num, best_params, best_metrics):
        self.logger.info(">>> NEW GLOBAL BEST at Trial %s!", trial_num)
        self.logger.info(">>> Params:  %s", self._format_params(best_params))
        self.logger.info(">>> Results: \n%s", self._format_metrics(best_metrics))

    def _log_best(self, trial_num, best_params, best_metrics):
        best_idx = self.best_trial_num or trial_num
        self.logger.info(">>> BEST-SO-FAR: Trial %s (after %s trials)", best_idx, trial_num)
        self.logger.info("\tParams: %s", self._format_params(best_params))
        self.logger.info("\tResults: \n%s", self._format_metrics(best_metrics))

    def _display_new_best(self, trial_result: Dict[str, Any]):
        self.logger.info("NEW GLOBAL BEST - Params: %s", self._format_params(trial_result["params"]))

    def _display_final_results(
        self, strategy: str, target_metric: str, total_trials: int, csv_file: str = None,
    ):
        strategy_name = "GRID SEARCH" if strategy == "grid" else f"{strategy.upper()} HPO"
        header = f" {strategy_name} RESULTS ".center(80, "=")
        self.logger.info(header)
        self.logger.info(f"Model: {self.model_name} | Dataset: {self.dataset_name}")
        self.logger.info(f"Strategy: {strategy_name} | Total Trials: {total_trials}")
        self.logger.info(f"Target Metric: {target_metric.upper()}")

        if self.best_metrics:
            self.logger.info("Best Metrics: %s", self._format_metrics(self.best_metrics))
        if self.best_config:
            self.logger.info("Best Config: %s", self._format_params(self.best_config))
        if csv_file:
            self.logger.info(f"Results saved to: {csv_file}")

    def _analyze_importance(self, study, target_metric: str) -> Dict[str, Any]:
        completed = [
            trial
            for trial in study.trials
            if trial.state.name == "COMPLETE" and trial.value is not None
        ]
        if len(completed) < 2:
            self.logger.info(
                "Skipping hyperparameter importance: fewer than two completed trials."
            )
            return {}

        target_values = {float(trial.value) for trial in completed}
        if len(target_values) < 2:
            self.logger.info(
                "Skipping hyperparameter importance: completed trials have zero target variance."
            )
            return {}

        param_names = {
            name for trial in completed for name in trial.params
        }
        varied_params = {
            name
            for name in param_names
            if len({trial.params.get(name) for trial in completed}) > 1
        }
        if not varied_params:
            self.logger.info(
                "Skipping hyperparameter importance: completed trials have no varied search parameters."
            )
            return {}

        from optuna.importance import get_param_importances

        importance_data = get_param_importances(study)
        sorted_importance = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        self.logger.info("Hyperparameter Importance:")
        for param, importance in sorted_importance[:5]:
            self.logger.info("  %-20s: %.4f", param, importance)
        return {"parameter_importance": dict(sorted_importance)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_unified_hpo(
    model_name: str,
    dataset_name: str,
    base_config: Dict[str, Any],
    strategy: str = "grid",
    target_metric: str = None,
    max_trials: int = None,
    resume: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Unified hyperparameter optimization entry point."""
    manager = UnifiedHPOManager(model_name, dataset_name, base_config)
    return manager.run_optimization(
        strategy=strategy,
        target_metric=target_metric,
        max_trials=max_trials,
        resume=resume,
        verbose=verbose,
        **kwargs,
    )
