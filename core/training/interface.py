# coding: utf-8
"""
Training Interface — user-facing APIs, flow orchestration, and model persistence.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from ..config import ConfigManager, SEARCHABLE_RANGE_TYPES, deep_merge_dict
from ..utils.result import Result
from ..utils.training import dict2str
from .core import train_single
from .environment import prepare_env


def _has_hpo_ranges(config: Dict[str, Any], hyper_parameters: List[str]) -> bool:
    if not hyper_parameters:
        return False

    parameter_space = config["parameter_space"]
    for param_name in hyper_parameters:
        if param_name in parameter_space:
            param_config = parameter_space[param_name]
            if not isinstance(param_config, dict):
                continue
            # The documented grid-only pattern (Tutorial 02/04): a bare
            # grid_values list with no range type is searchable when it
            # enumerates more than one value (mirrors
            # core/hpo/parameters._resolve_grid_values, which prefers
            # grid_values over any declared type).
            if (
                "grid_values" in param_config
                and isinstance(param_config["grid_values"], list)
                and len(param_config["grid_values"]) > 1
            ):
                return True
            if "type" not in param_config:
                continue
            param_type = param_config["type"]
            if param_type == "choice" and isinstance(param_config["values"], list):
                if len(param_config["values"]) > 1:
                    return True
            elif param_type in SEARCHABLE_RANGE_TYPES:
                return True

    return False


def _resolve_hpo_strategy(config: Dict[str, Any]) -> str:
    return config["optimization"]["strategy"]


def _hpo_lineage(hpo_result: Dict[str, Any]) -> Dict[str, Any]:
    lineage = {}
    if "csv_file" in hpo_result:
        lineage["source_csv"] = hpo_result["csv_file"]
    if "strategy" in hpo_result:
        lineage["strategy"] = hpo_result["strategy"]
    if "target_metric" in hpo_result:
        lineage["target_metric"] = hpo_result["target_metric"]
    if "best_trial_num" in hpo_result:
        lineage["best_trial_num"] = hpo_result["best_trial_num"]
    if "best_score" in hpo_result:
        lineage["best_score"] = hpo_result["best_score"]
    if "best_configuration" in hpo_result:
        lineage["best_configuration"] = json.dumps(
            hpo_result["best_configuration"],
            sort_keys=True,
        )
    return lineage


def _build_final_train_config(
    input_config: Dict[str, Any],
    hpo_result: Dict[str, Any],
    final_train_config: Dict[str, Any],
) -> Dict[str, Any]:
    if "best_configuration" not in hpo_result:
        raise ValueError("HPO result does not contain best_configuration for final_train.")
    best_configuration = hpo_result["best_configuration"]
    if not isinstance(best_configuration, dict) or not best_configuration:
        raise ValueError("HPO final_train requires a non-empty best_configuration.")

    from ..hpo.engine import _expand_dotted_params

    final_config = deep_merge_dict(input_config, _expand_dotted_params(best_configuration))
    overrides = final_train_config["overrides"]
    if not isinstance(overrides, dict):
        raise ValueError("optimization.final_train.overrides must be a dict.")
    final_config = deep_merge_dict(final_config, overrides)
    final_config["smart_hpo"] = False

    optimization = final_config["optimization"] if "optimization" in final_config else {}
    if not isinstance(optimization, dict):
        raise ValueError("optimization override must be a dict.")
    optimization = deep_merge_dict(
        optimization,
        {
            "parallel": False,
            "final_train": deep_merge_dict(final_train_config, {"enabled": False}),
        },
    )
    final_config["optimization"] = optimization
    final_config["hpo_lineage"] = _hpo_lineage(hpo_result)
    return final_config


def _maybe_run_final_train(
    model: str,
    dataset: str,
    input_config: Dict[str, Any],
    runtime_config: Dict[str, Any],
    hpo_result: Dict[str, Any],
) -> Dict[str, Any]:
    final_train_config = runtime_config["optimization"]["final_train"]
    if not bool(final_train_config["enabled"]):
        return hpo_result
    if "dry_run" in hpo_result and bool(hpo_result["dry_run"]):
        return hpo_result

    logger = logging.getLogger("nexusrec")
    logger.info("Starting final_train with HPO best_configuration")
    final_config = _build_final_train_config(
        input_config,
        hpo_result,
        final_train_config,
    )
    hpo_result["final_train"] = run_training(
        model=model,
        dataset=dataset,
        config_dict=final_config,
        save_model=None,
    )
    return hpo_result


# ---------------------------------------------------------------------------
# Flows — training and HPO orchestration
# ---------------------------------------------------------------------------


def _run_hpo_flow(
    model: str,
    dataset: str,
    config: Dict[str, Any],
    strategy: str,
    target_metric: str,
    resume: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Run unified HPO from the user-facing training entrypoint."""
    from ..hpo.engine import run_unified_hpo

    logger = logging.getLogger("nexusrec")
    logger.info("Using %s strategy for HPO", strategy.upper())

    return run_unified_hpo(
        model_name=model,
        dataset_name=dataset,
        base_config=dict(config),
        strategy=strategy,
        target_metric=target_metric,
        resume=resume,
        verbose=verbose,
    )


def _run_single(
    config: Dict[str, Any],
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    save_model: bool,
) -> Dict[str, Any]:
    logger = logging.getLogger("nexusrec")
    test_result, valid_result, trainer = train_single(
        config,
        train_data,
        valid_data,
        test_data,
        return_trainer=save_model,
    )

    if save_model and trainer is not None:
        trainer.save_checkpoint(config["checkpoint_dir"])

    logger.info("Training Complete")
    logger.info("Test Results: %s", dict2str(test_result))

    param_dict = {
        "model": config["model"],
        "dataset": config["dataset"],
        "type": config["type"],
        "comment": config["comment"],
        **Result.provenance(config),
    }
    Result.write(config["result_file_name"], {**param_dict, **test_result})

    return {
        "test_result": test_result,
        "valid_result": valid_result,
        "config": config,
    }


def run_training(
    model: str,
    dataset: str,
    config_dict: Dict[str, Any],
    save_model: bool | None,
) -> Dict[str, Any]:
    """Run the standard training flow."""
    runtime_config = dict(config_dict)
    if save_model is not None:
        runtime_config["save_model"] = save_model

    config, train_data, valid_data, test_data = prepare_env(
        model,
        dataset,
        runtime_config,
        setup_logging=True,
    )
    return _run_single(
        config,
        train_data,
        valid_data,
        test_data,
        config["save_model"],
    )


# ---------------------------------------------------------------------------
# User-facing APIs
# ---------------------------------------------------------------------------

def quick_start(
    model: str,
    dataset: str,
    config_dict: Dict[str, Any],
    save_model: bool | None = True,
    resume: bool = True,
    verbose: bool = False,
):
    """User-facing quick_start API backed by the unified runtime."""
    input_config = dict(config_dict)
    config = ConfigManager(model, dataset, input_config)
    if save_model is not None:
        config["save_model"] = save_model

    explicit_hpo = bool(config["smart_hpo"])
    if explicit_hpo:
        hyper_parameters = list(config["hyper_parameters"])
        if not hyper_parameters:
            raise ValueError(
                "smart_hpo=true requires a non-empty hyper_parameters list."
            )
        if not _has_hpo_ranges(config, hyper_parameters):
            raise ValueError(
                "smart_hpo=true requires at least one searchable parameter range in parameter_space."
            )
        strategy = _resolve_hpo_strategy(config)
        optimization = config["optimization"]
        if optimization["parallel"]:
            from ..hpo.parallel import run_parallel_hpo

            runtime_config = {key: value for key, value in config.items()}
            hpo_result = run_parallel_hpo(
                model_name=model,
                dataset_name=dataset,
                input_config=input_config,
                runtime_config=runtime_config,
                strategy=strategy,
                target_metric=config["valid_metric"],
                resume=resume,
                verbose=verbose,
            )
            return _maybe_run_final_train(
                model,
                dataset,
                input_config,
                runtime_config,
                hpo_result,
            )
        hpo_result = _run_hpo_flow(
            model=model,
            dataset=dataset,
            config=input_config,
            strategy=strategy,
            target_metric=config["valid_metric"],
            resume=resume,
            verbose=verbose,
        )
        runtime_config = {key: value for key, value in config.items()}
        return _maybe_run_final_train(
            model,
            dataset,
            input_config,
            runtime_config,
            hpo_result,
        )

    return run_training(
        model=model,
        dataset=dataset,
        config_dict=input_config,
        save_model=save_model,
    )
