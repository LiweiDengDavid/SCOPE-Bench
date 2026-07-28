#!/usr/bin/env python
# coding: utf-8
"""
Validate registered model configs against the repository state.

The source of truth is the repository itself:
- `configs/models/*.yaml` defines the registered model set
- `models/**` provides the implementation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "models"
CONFIGS_ROOT = REPO_ROOT / "configs" / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import BAD_KEYS, BAD_ROOT_KEYS
from core.model_registry import infer_paradigm, get_model_source_path

BAD_MODEL_KEYS = frozenset(
    {
        "n_layers",
        "n_ui_layers",
        "hidden_layers",
        "num_gcn_layers",
        "dropout_prob",
        "fusion_module",
        "learner",
        "scheduler",
        "lr_step_size",
        "lr_gamma",
        "ssl_temp",
        "latent_size",
        "early_stop",
        "client_sample_ratio",
        "k_list",
        "lr_scheduler",
        "max_sequence_length",
        "min_sequence_length",
        "MAX_ITEM_LIST_LENGTH",
        "MIN_ITEM_LIST_LENGTH",
        "hpo_default_strategy",
        "hpo_budget",
        "hpo_timeout",
        "hpo_early_stopping_patience",
        "hpo_convergence_tolerance",
        "hpo_objective",
        "hpo_save_intermediate",
        "hpo_save_plots",
        "hpo_detailed_logs",
        "data_split_strategy",
        "data_split_train_ratio",
        "data_split_validation_ratio",
        "data_split_test_ratio",
        "temporal_split_method",
        "remove_duplicate_interactions",
        "min_interactions_per_user",
        "min_interactions_per_item",
        "data_augmentation_enabled",
        "data_augmentation_strategies",
        "split_strategy",
        "neg_sample_strategy",
        "neg_sample_num",
        "negative_sampling_strategy",
        "num_negatives",
        "negative_sampling_max_attempts",
        "check_duplicate_negatives",
        "filter_test_items_from_negatives",
        "use_neighborhood_loss",
        "federated_eval_enabled",
        "federated_eval_step",
        "differential_privacy_enabled",
        "privacy_epsilon",
        "privacy_delta",
    }
)

# Keep smoke validation lightweight and deterministic.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class ModelValidationResult:
    model: str
    paradigm: str
    dependency: str
    source_ok: bool
    class_ok: bool
    profile_ok: bool
    config_ok: bool
    contract_ok: bool
    static_ok: bool
    smoke_ok: Optional[bool]
    smoke_skip: Optional[str]
    issues: List[str]


def list_registered_models() -> List[str]:
    return sorted(config_path.stem for config_path in CONFIGS_ROOT.glob("*.yaml"))


def find_model_source(model_name: str) -> Optional[Path]:
    if not config_exists_for_model(model_name):
        return None
    source_path = REPO_ROOT / get_model_source_path(model_name)
    if source_path.exists():
        return source_path
    return None


def infer_dependency(file_path: Path) -> str:
    source = file_path.read_text(encoding="utf-8")
    extended_markers = [
        "torch_geometric",
        "torch_scatter",
        "torchdiffeq",
        "diffusion",
    ]
    if any(marker in source for marker in extended_markers):
        return "extended"
    return "core"


def load_model_profile_config(model_name: str) -> Tuple[Dict[str, Any], List[str]]:
    config_path = CONFIGS_ROOT / f"{model_name}.yaml"
    if not config_path.exists():
        return {}, ["missing config file"]

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return {}, ["config root must be a mapping"]
    required_keys = ["is_federated", "is_multimodal_model", "is_sequential"]
    missing = [key for key in required_keys if key not in payload]
    return payload, missing


def validate_model_config_contract(model_name: str) -> List[str]:
    """Ensure model configs follow the canonical baseline/search contract."""
    config_path = CONFIGS_ROOT / f"{model_name}.yaml"
    if not config_path.exists():
        return ["missing config file"]

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return ["config root must be a mapping"]
    issues: List[str] = []

    hyper_parameters = []
    if "hyper_parameters" in payload and payload["hyper_parameters"] is not None:
        hyper_parameters = payload["hyper_parameters"]

    parameter_space = {}
    if "parameter_space" in payload and payload["parameter_space"] is not None:
        parameter_space = payload["parameter_space"]

    unsupported_top_level = sorted(
        key
        for key in payload
        if key in BAD_ROOT_KEYS
        or key in BAD_KEYS
        or key in BAD_MODEL_KEYS
    )
    unsupported_hparams = sorted(
        key
        for key in hyper_parameters
        if key in BAD_MODEL_KEYS or key in BAD_KEYS
    )
    unsupported_parameter_space = sorted(
        key
        for key in parameter_space
        if key in BAD_MODEL_KEYS or key in BAD_KEYS
    )

    for key in unsupported_top_level:
        issues.append(f"unsupported deprecated config key '{key}' present")
    for key in unsupported_hparams:
        issues.append(f"unsupported deprecated hyperparameter name '{key}' present")
    for key in unsupported_parameter_space:
        issues.append(f"unsupported deprecated parameter_space entry '{key}' present")

    if hyper_parameters and not isinstance(parameter_space, dict):
        return ["hyper_parameters declared without parameter_space mapping"]

    def _has_runtime_default(param_name: str) -> Tuple[bool, Any]:
        if param_name in payload:
            return True, payload[param_name]
        if "." not in param_name:
            return False, None
        node: Any = payload
        for part in param_name.split("."):
            if not isinstance(node, dict) or part not in node:
                return False, None
            node = node[part]
        return True, node

    for param_name in hyper_parameters:
        has_default, default_value = _has_runtime_default(param_name)
        if not has_default:
            issues.append(f"missing default value for hyperparameter '{param_name}'")
            continue
        if param_name not in parameter_space:
            issues.append(f"missing parameter_space entry for '{param_name}'")
        if isinstance(default_value, list):
            issues.append(f"hyperparameter '{param_name}' still uses list-style defaults")

    return issues


def config_exists_for_model(model_name: str) -> bool:
    return (CONFIGS_ROOT / f"{model_name}.yaml").exists()


def _prepare_batch_for_device(batch: Any, device: Any) -> Any:
    import torch

    if isinstance(batch, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return [
            value.to(device) if isinstance(value, torch.Tensor) else value
            for value in batch
        ]
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def _get_first_training_batch(train_data: Any, config: Dict[str, Any]) -> Any:
    if config["is_federated"]:
        first_user = next(iter(train_data.user_set))
        client_loader = train_data.loaders[first_user]
        if hasattr(client_loader, "pretrain_setup"):
            client_loader.pretrain_setup()
        return next(iter(client_loader))

    if hasattr(train_data, "pretrain_setup"):
        train_data.pretrain_setup()
    return next(iter(train_data))


def _get_first_eval_batch(eval_data: Any, config: Dict[str, Any]) -> Any:
    if eval_data is None:
        return None

    if config["is_federated"]:
        first_user = next(iter(eval_data.user_set))
        client_loader = eval_data.loaders[first_user]
        return next(iter(client_loader))

    return next(iter(eval_data))


def run_smoke_test(model_name: str, dataset_name: str) -> Tuple[Optional[bool], Optional[str]]:
    import torch

    sys.path.insert(0, str(REPO_ROOT))

    from core.training import prepare_env
    from core.model_registry import get_model

    config_overrides = {
        "max_epochs": 1,
        "hyper_parameters": [],
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "save_model": False,
        "print_model_info": False,
        "eval_test_during_training": False,
        "topk": [10],
        "federated": {
            "local_epochs": 1,
            "clients_sample_ratio": 0.2,
            "clients_sample_strategy": "random",
        },
        "evaluation": {
            "skip_eval_during_training": False,
        },
    }

    config, train_data, valid_data, _ = prepare_env(
        model_name,
        dataset_name,
        config_overrides,
        setup_logging=False,
    )

    model_class = get_model(config["model"])
    model = model_class(config, train_data).to(config["device"])
    model.train()

    train_batch = _prepare_batch_for_device(
        _get_first_training_batch(train_data, config),
        config["device"],
    )
    loss = model.calculate_loss(train_batch)
    if isinstance(loss, tuple):
        loss = sum(loss)
    loss.backward()

    model.eval()
    eval_batch = _get_first_eval_batch(valid_data, config)
    if eval_batch is not None:
        eval_batch = _prepare_batch_for_device(eval_batch, config["device"])
        with torch.no_grad():
            _ = model.full_sort_predict(eval_batch)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True, None


def validate_models(model_names: List[str], dataset_name: str, run_smoke: bool) -> List[ModelValidationResult]:
    results: List[ModelValidationResult] = []

    for model_name in model_names:
        issues: List[str] = []
        src_path = find_model_source(model_name)
        source_ok = src_path is not None and src_path.exists()
        config_ok = config_exists_for_model(model_name)
        profile, missing_flags = load_model_profile_config(model_name)
        contract_issues = validate_model_config_contract(model_name) if config_ok else []
        class_ok = False

        paradigm = "UNKNOWN"
        if not missing_flags:
            paradigm = infer_paradigm(profile)

        dependency = "UNKNOWN"
        if src_path and src_path.exists():
            source = src_path.read_text(encoding="utf-8", errors="ignore")
            class_ok = bool(
                re.search(rf"^\s*class\s+{re.escape(model_name)}\b", source, re.MULTILINE)
            )
            dependency = infer_dependency(src_path)

        if not source_ok:
            issues.append("missing model source file")
        if not class_ok:
            issues.append("model class not declared in source")
        if not config_ok:
            issues.append("missing config file")
        if missing_flags:
            issues.append(
                "missing required model profile flags: " + ", ".join(missing_flags)
            )
        if contract_issues:
            issues.extend(contract_issues)

        static_ok = len(issues) == 0
        smoke_ok: Optional[bool] = None
        smoke_skip: Optional[str] = None

        if run_smoke and static_ok:
            smoke_ok, smoke_reason = run_smoke_test(model_name, dataset_name)
            if smoke_ok is None:
                smoke_skip = smoke_reason
            elif smoke_ok is False and smoke_reason:
                issues.append(smoke_reason)
        elif run_smoke and not static_ok:
            smoke_skip = "static validation failed"

        results.append(
            ModelValidationResult(
                model=model_name,
                paradigm=paradigm,
                dependency=dependency,
                source_ok=source_ok,
                class_ok=class_ok,
                profile_ok=not missing_flags,
                config_ok=config_ok,
                contract_ok=not contract_issues,
                static_ok=static_ok,
                smoke_ok=smoke_ok,
                smoke_skip=smoke_skip,
                issues=issues,
            )
        )

    return results


def format_result_line(result: ModelValidationResult) -> str:
    smoke_state = "SKIPPED"
    if result.smoke_ok is True:
        smoke_state = "PASS"
    elif result.smoke_ok is False:
        smoke_state = "FAIL"

    return (
        f"{result.model}: "
        f"paradigm={result.paradigm}, "
        f"dependency={result.dependency}, "
        f"static={'PASS' if result.static_ok else 'FAIL'}, "
        f"smoke={smoke_state}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate registered models.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific models to validate. Defaults to all configs/models/*.yaml entries.",
    )
    parser.add_argument(
        "--dataset",
        default="Beauty",
        help="Dataset name used for optional smoke tests.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run optional one-epoch smoke tests when runtime dependencies exist.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to save the validation report as JSON.",
    )
    args = parser.parse_args()

    model_names = args.models or list_registered_models()
    results = validate_models(model_names, args.dataset, args.smoke)

    print("Model validation summary")
    print("=" * 80)
    for result in results:
        print(format_result_line(result))
        if result.issues:
            print("  issues:", "; ".join(result.issues))
        elif result.smoke_skip:
            print("  smoke skipped:", result.smoke_skip)

    report = [asdict(result) for result in results]
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {output_path}")

    has_static_failure = any(not result.static_ok for result in results)
    has_smoke_failure = any(result.smoke_ok is False for result in results)
    return 1 if has_static_failure or has_smoke_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
