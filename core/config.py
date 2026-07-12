# coding: utf-8
 

"""Unified configuration management for NexusRec."""

import copy
import datetime
import os
import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import yaml

from .evaluation import export_contract


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_numeric_string(value: str) -> Any:
    """Convert a numeric-looking string to int/float, otherwise return it unchanged."""
    if not _NUMERIC_RE.match(value):
        return value

    number = float(value)
    has_decimal = "." in value
    has_exponent = "e" in value.lower()
    if number.is_integer() and not has_decimal and not has_exponent:
        return int(number)
    return number



def _coerce_numeric_strings(obj: Any) -> Any:
    """Recursively convert string values that look like numbers (e.g. '1e-5') to float/int.

    PyYAML safe_load does not recognise scientific notation like ``1e-5`` as
    floats — they come through as plain strings. This helper normalizes them
    in-place so downstream code never has to worry about the distinction.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _coerce_numeric_strings(value)
        return obj
    if isinstance(obj, list):
        return [_coerce_numeric_strings(item) for item in obj]
    if isinstance(obj, str):
        return _parse_numeric_string(obj)
    return obj


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Load a YAML file that must contain a mapping or be empty."""
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigValidationError(f"Config file must contain a mapping: {path}")
    return payload


def deep_merge_dict(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base_dict)
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Runtime config readers and validators
# ---------------------------------------------------------------------------

_FLATTEN_GROUPS = ["training", "evaluation", "output", "experiment", "sequential", "sampling", "resources"]

BAD_ROOT_KEYS = frozenset({"epochs", "budget", "hpo"})
BAD_KEYS = frozenset(
    {
        "objective_metric",
        "maximize_metric",
        "dropout",
        "search_space",
    }
)
BAD_MODEL_ALIAS_KEYS = frozenset(
    {
        "hidden_layers",
        "num_gcn_layers",
    }
)

# Keys injected by the CLI/runtime that legitimately appear in config_dict overrides
# without being declared in the merged YAML schema (model/dataset selectors and
# experiment/logging flags). Everything else in an override must match a real config key.
RUNTIME_OVERRIDE_KEYS = frozenset(
    {
        "model",
        "dataset",
        "gpu_id",
        "type",
        "comment",
        "verbose",
        "smart_hpo",
        "early_stopping",
        "hyper_parameters",
        "checkpoint_dir",
        "hpo_dir",
        "log_dir",
    }
)


def flatten_nested_groups(config: Dict[str, Any]) -> None:
    """Promote nested group fields to top-level.

    Model/CLI overrides already written to top-level via deep_merge take
    precedence because nested defaults only fill missing top-level keys.
    """
    for group_name in _FLATTEN_GROUPS:
        if group_name in config and isinstance(config[group_name], dict):
            group = config[group_name]
            for key, value in group.items():
                if key not in config:
                    config[key] = value


def coerce_runtime_scalar(value: Any) -> Any:
    """Normalize a runtime value to a plain Python scalar."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        raise ConfigValidationError(
            "Unexpected list-valued scalar config. "
            "Declare search ranges under parameter_space instead of runtime keys."
        )
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        parsed = _parse_numeric_string(value)
        return parsed
    return value


def extract_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the training parameters TrainerBase consumes from the config.

    Only the keys the trainer reads are returned; batch sizes / seed are read
    directly from config where needed, so they are not duplicated here.
    """
    return {
        "max_epochs": config["max_epochs"],
        "eval_step": config["eval_step"],
        "eval_enabled": config["eval_enabled"],
        "skip_eval_during_training": config["skip_eval_during_training"],
        "stopping_step": config["stopping_step"],
        "early_stopping": config["early_stopping"],
        "clip_grad_norm": config["clip_grad_norm"],
        "device": config["device"],
    }


def extract_federated_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract federated training parameters from the unified config."""
    federated = config["federated"]
    if not isinstance(federated, dict):
        raise KeyError("Config key 'federated' must be a dict.")
    required_keys = [
        "local_epochs",
        "clients_sample_ratio",
        "clients_sample_strategy",
        "aggregation_method",
    ]
    missing = [k for k in required_keys if k not in federated]
    if missing:
        raise KeyError(
            f"Missing required federated config key(s): {missing}. "
            "Add them under the 'federated:' section in your model YAML "
            "(e.g. configs/models/YourModel.yaml)."
        )
    return {k: federated[k] for k in required_keys}


def extract_evaluation_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract evaluation parameters from the unified config."""
    return {
        "valid_metric": str(config["valid_metric"]),
        "valid_metric_bigger": config["valid_metric_bigger"],
    }


# ---------------------------------------------------------------------------
# Runtime device/path assignment and validation
# ---------------------------------------------------------------------------

def assign_runtime_device(config: Dict[str, Any]) -> None:
    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_id = config["gpu_id"]

    # A negative gpu_id is the documented "run on CPU" sentinel — honor it even
    # when CUDA is available (otherwise torch.device("cuda:-1") would be invalid).
    if gpu_count > 0 and gpu_id >= 0:
        if gpu_id >= gpu_count:
            logging.getLogger("nexusrec").warning(
                f"gpu_id {gpu_id} >= available GPU count {gpu_count}, falling back to GPU 0"
            )
            gpu_id = 0

        config["device"] = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        config["device"] = torch.device("cpu")


def set_paths(config: Dict[str, Any]) -> None:
    """Set up output paths from the output: config section (flattened to top-level)."""
    model_name = config["model"]
    dataset_name = config["dataset"]
    run_type = config["type"]
    comment = config["comment"]

    fmt = {"model": model_name, "dataset": dataset_name, "type": run_type}

    # Read path templates from output: config (flattened by flatten_nested_groups)
    log_template = config["log_path"]
    checkpoint_template = config["checkpoint_path"]
    save_template = config["save_path"]

    hpo_template = config["hpo_path"]

    base_log_path = os.path.abspath(log_template.format(**fmt))
    base_checkpoint_path = os.path.abspath(checkpoint_template.format(**fmt))
    base_result_path = os.path.abspath(save_template.format(**fmt))
    base_hpo_path = os.path.abspath(hpo_template.format(**fmt))

    if "paths" not in config:
        config["paths"] = {}
    paths = config["paths"]
    if "log" not in paths:
        paths["log"] = base_log_path
    if "checkpoint" not in paths:
        paths["checkpoint"] = base_checkpoint_path
    if "save" not in paths:
        paths["save"] = base_result_path
    if "hpo" not in paths:
        paths["hpo"] = base_hpo_path

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    if "log_dir" not in config:
        config["log_dir"] = base_log_path
    if "checkpoint_dir" not in config:
        config["checkpoint_dir"] = base_checkpoint_path
    if "hpo_dir" not in config:
        config["hpo_dir"] = base_hpo_path

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    comment_str = f".{comment}" if comment else ""
    log_filename = f"[{model_name}]-[{dataset_name}]-[{run_type}{comment_str}]-[{timestamp}].txt"
    result_filename = f"[{model_name}]-[{dataset_name}]-[{run_type}{comment_str}].csv"
    model_filename = f"[{run_type}].pkl"

    config["log_file_name"] = os.path.join(base_log_path, log_filename)
    config["result_file_name"] = os.path.join(base_result_path, result_filename)
    config["model_dir"] = os.path.join(base_checkpoint_path, model_filename)


def validate_parameter_ranges(config: Dict[str, Any], error_cls: Type[Exception]) -> None:
    for key in ["dropout_rate", "attention_dropout_rate", "hidden_dropout_rate", "init_dropout"]:
        if key not in config:
            continue
        vals = config[key] if isinstance(config[key], (list, tuple)) else [config[key]]
        for v in vals:
            f = float(v)
            if not 0.0 <= f <= 1.0:
                raise error_cls(f"{key}={f} must be between 0.0 and 1.0")


def validate_training(config: Dict[str, Any], error_cls: Type[Exception]) -> None:
    missing = []
    for key in ["max_epochs", "learning_rate"]:
        if key not in config or config[key] is None:
            missing.append(key)
    if missing:
        raise error_cls(f"Missing required training config: {missing}")

    # The resume cadence is a modulo divisor (completed_epochs % cadence); a
    # value of 0 would fail at the first epoch boundary. Validate before training,
    # independent of whether resume is enabled.
    if "checkpoint_every_n_epochs" in config:
        cadence = config["checkpoint_every_n_epochs"]
        if not isinstance(cadence, int) or isinstance(cadence, bool) or cadence < 1:
            raise error_cls(
                f"checkpoint_every_n_epochs must be a positive integer, got {cadence!r}"
            )

    # HPO selects the winning trial by optimization.objective, while each trial
    # trains/tracks its best epoch by valid_metric_bigger. These two direction
    # knobs are independent; a mismatch would silently rank trials in the opposite
    # direction and report the WORST trial as best. Fail fast so the inconsistency
    # surfaces at config-load time (both default to maximize/true and agree).
    objective = config["optimization"]["objective"]
    if objective not in ("maximize", "minimize"):
        raise error_cls(
            f"optimization.objective must be 'maximize' or 'minimize', got {objective!r}"
        )
    if (objective == "maximize") != bool(config["valid_metric_bigger"]):
        raise error_cls(
            f"Inconsistent optimization direction: optimization.objective={objective!r} "
            f"but valid_metric_bigger={config['valid_metric_bigger']!r}. They must agree "
            f"(maximize<->true, minimize<->false) or HPO selects the worst trial as best."
        )


def validate_export(config: Dict[str, Any], error_cls: Type[Exception]) -> None:
    export_contract.validate_section(
        config,
        error_cls,
        legacy_conflict_message=(
            "output.export cannot be enabled together with legacy "
            "save_recommended_topk. Disable save_recommended_topk for "
            "recommendation-list export."
        ),
    )


def reject_bad_keys(
    config: Dict[str, Any],
    error_cls: Type[Exception],
) -> None:
    findings = []

    def _walk(node: Any, path: Tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            current_path = path + (key,)
            dotted = ".".join(current_path)
            if not path and key in BAD_ROOT_KEYS:
                findings.append(dotted)
            if key in BAD_KEYS and "parameter_space" not in path:
                findings.append(dotted)
            if key in BAD_MODEL_ALIAS_KEYS:
                findings.append(dotted)
            _walk(value, current_path)

    _walk(config, ())

    if findings:
        raise error_cls(
            "Unsupported deprecated config keys detected: "
            + ", ".join(findings)
            + ". Use only the canonical configuration contract."
        )


def _collect_key_names(node: Any, acc: set) -> None:
    """Collect dict key names from a (possibly nested) mapping for the override
    allow-set.

    parameter_space is treated specially: add its param-NAME children (legitimate
    config keys, e.g. learning_rate) but do NOT descend into their HPO spec dicts
    (type/values/low/high). Otherwise 'low'/'high' enter the allow-set and a stray
    top-level override like {'low': 0.5} passes the unknown-key guard as an
    unconsumed override; the override side avoids this by skipping parameter_space.
    hpo_lineage is runtime-injected provenance and may contain run-specific keys.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            acc.add(key)
            if key == "parameter_space" and isinstance(value, dict):
                acc.update(value.keys())
            elif key == "hpo_lineage":
                continue
            else:
                _collect_key_names(value, acc)


def reject_unknown_override_keys(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
    error_cls: Type[Exception],
) -> None:
    """Fail fast on override keys that match no key in the merged base schema.

    Unknown override keys — typos, renamed keys, or incompatible feature knobs —
    would otherwise merge and remain unread, so the config a user
    *thinks* they ran can differ from what actually trained. ``base_config`` is the merged
    overall+model+dataset config BEFORE user overrides are applied; any override key
    (flat or nested) must name a key that already exists in that schema, or be a known
    runtime/CLI-injected key.
    """
    known: set = set()
    _collect_key_names(base_config, known)
    known |= RUNTIME_OVERRIDE_KEYS
    # Explicitly rejected keys are handled by reject_bad_keys after the merge so
    # callers receive the most specific validation message.
    known |= BAD_ROOT_KEYS | BAD_KEYS | BAD_MODEL_ALIAS_KEYS

    # Collect override key names, but do NOT descend into parameter_space: its
    # sub-keys are model-defined HPO range specs (hyperparameter names + type/values/
    # low/high), not runtime config keys, and are validated elsewhere.
    seen: set = set()

    def _collect_override(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                seen.add(key)
                if key not in {"parameter_space", "hpo_lineage"}:
                    _collect_override(value)

    _collect_override(overrides)
    unknown = sorted(seen - known)
    if unknown:
        raise error_cls(
            "Unknown override key(s) not present in the merged config schema: "
            + ", ".join(unknown)
            + ". They would not be consumed — check for typos or incompatible "
            "feature-specific keys."
        )


# ---------------------------------------------------------------------------
# Hyperparameter normalization helpers
# ---------------------------------------------------------------------------

_LIST_FIELDS_TO_PRESERVE = {
    "metrics",
    "topk",
    "hyper_parameters",
    "augmentation_strategies",
    "precedence_order",
    "available",
}

_LIST_FIELDS_WITH_MODEL_SEMANTICS = {
    "weight_size",
    "hidden_dims",
    "dims_mlp",
    "mixture_weights",
    # Runtime list params read verbatim by their models; these are not treated as
    # HPO search choices unless they also appear under parameter_space.
    "hop_fusion_weights",        # CFDiff: CAM_AE_multihops 1/2/3-hop weights (cfdiff.py:32)
    "learning_rate_scheduler",   # StepLR [gamma, step_size] (factory.py:125; LATTICE/MGCN)
}


# The five parameter_space range types that mark a parameter as HPO-searchable.
# Single source of truth shared by resolve_default, normalize_hparams's HPO
# auto-detection, and interface._has_hpo_ranges.
SEARCHABLE_RANGE_TYPES = frozenset(
    {"uniform", "loguniform", "int", "logscale", "logint"}
)


def _nested_lookup(config: Dict[str, Any], dotted: str) -> Any:
    """Walk a dotted key (e.g. 'loss_weights.ssl_loss') through nested dicts."""
    node = config
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _param_current_value(config: Dict[str, Any], param_name: str) -> Any:
    """Current value of a (possibly dotted) hyperparameter from the config.

    A dotted name (e.g. 'loss_weights.ssl_loss') is resolved through the nested
    config the model actually reads, NOT treated as a missing flat key — otherwise
    its default would fall through to parameter_space values[0] and diverge from
    the trained value.
    """
    if param_name in config:
        return config[param_name]
    if "." in param_name:
        return _nested_lookup(config, param_name)
    return None


def resolve_default(param_config: Dict[str, Any], current_value: Any) -> Any:
    """Resolve the default runtime value for a searchable parameter."""
    if current_value is not None and not isinstance(current_value, list):
        return current_value

    param_type = param_config["type"] if "type" in param_config else "choice"

    if param_type == "choice":
        values = param_config["values"] if "values" in param_config else []
        return coerce_runtime_scalar(values[0]) if values else None

    if current_value is not None:
        return coerce_runtime_scalar(current_value)

    if "default" in param_config:
        return coerce_runtime_scalar(param_config["default"])

    if param_type in SEARCHABLE_RANGE_TYPES:
        # The low end of the range is the deterministic (HPO-off) default. Log-
        # space ranges require low > 0 (enforced by the samplers in
        # core/hpo/parameters.py and optuna suggest_float(log=True)); a malformed
        # low<=0 therefore fails consistently on both paths rather than being
        # implicitly transformed here.
        return coerce_runtime_scalar(param_config["low"] if "low" in param_config else None)

    return None


def split_configs(config: Dict[str, Any]) -> None:
    """Split runtime defaults from hyperparameter search metadata."""
    hyper_parameters = []
    if "hyper_parameters" in config and config["hyper_parameters"] is not None:
        hyper_parameters = config["hyper_parameters"]

    existing_parameter_space = {}
    if "parameter_space" in config and config["parameter_space"] is not None:
        existing_parameter_space = config["parameter_space"]
    normalized_parameter_space: Dict[str, Dict[str, Any]] = {}

    for param_name, param_config in existing_parameter_space.items():
        if isinstance(param_config, dict):
            normalized_parameter_space[param_name] = copy.deepcopy(param_config)

    default_parameters: Dict[str, Any] = {}

    for param_name in hyper_parameters:
        current_value = _param_current_value(config, param_name)
        if param_name in normalized_parameter_space:
            default_parameters[param_name] = resolve_default(
                normalized_parameter_space[param_name],
                current_value,
            )
        else:
            default_parameters[param_name] = copy.deepcopy(current_value)

    for param_name, param_config in normalized_parameter_space.items():
        if param_name not in default_parameters:
            current_value = _param_current_value(config, param_name)
            default_parameters[param_name] = resolve_default(param_config, current_value)

    config["parameter_space"] = normalized_parameter_space
    config["default_parameters"] = default_parameters


def normalize_hparams(config: Dict[str, Any]) -> None:
    """Collapse searchable lists into scalar runtime defaults outside HPO."""
    is_hpo_mode = config["smart_hpo"] if "smart_hpo" in config else False

    default_parameters = config["default_parameters"] if "default_parameters" in config else {}
    for param_name, default_value in default_parameters.items():
        # Dotted params are read through their nested path and expanded in the HPO
        # trial path by _expand_dotted_params; keep the top-level runtime config flat.
        if "." in param_name:
            continue
        current_value = config[param_name] if param_name in config else None
        if isinstance(current_value, list) or current_value is None:
            config[param_name] = default_value

    if not is_hpo_mode:
        hyper_parameters = []
        if "hyper_parameters" in config and config["hyper_parameters"] is not None:
            hyper_parameters = config["hyper_parameters"]

        parameter_space = {}
        if "parameter_space" in config and config["parameter_space"] is not None:
            parameter_space = config["parameter_space"]
        for param_name in hyper_parameters:
            if param_name in parameter_space:
                param_config = parameter_space[param_name]
                if (
                    isinstance(param_config, dict)
                    and "type" in param_config
                    and param_config["type"] == "choice"
                ):
                    values = param_config["values"] if "values" in param_config else []
                    if isinstance(values, list) and len(values) > 1:
                        is_hpo_mode = True
                        break
                elif (
                    isinstance(param_config, dict)
                    and "type" in param_config
                    and param_config["type"] in SEARCHABLE_RANGE_TYPES
                ):
                    is_hpo_mode = True
                    break

    if is_hpo_mode:
        return

    for key, value in list(config.items()):
        if not isinstance(value, list):
            continue
        if key in _LIST_FIELDS_TO_PRESERVE:
            continue
        if key.endswith("_list") or key in _LIST_FIELDS_WITH_MODEL_SEMANTICS:
            continue
        if not value:
            continue
        raise ConfigValidationError(
            f"Unexpected list-valued runtime config '{key}'. "
            "Move searchable ranges into parameter_space or keep runtime values scalar."
        )

class ConfigValidationError(ValueError):
    """Configuration validation error."""
    pass


class ConfigManager:
    """Unified configuration manager with a thin orchestration surface."""

    def __init__(
        self,
        model: str = None,
        dataset: str = None,
        config_dict: Dict[str, Any] = None,
        trial_info: str = None,
    ):
        self.model = model
        self.dataset = dataset
        self.config_dict = {} if config_dict is None else config_dict
        self.trial_info = trial_info
        self._config: Dict[str, Any] = {}
        self._load_configuration()

    def _load_configuration(self):
        self._load_default_config()
        if self.model:
            self._load_model_config()
        if self.dataset:
            self._load_dataset_config()
        self._apply_config_overrides()
        self._post_process_config()

    # Root of the repository: two levels above core/config.py (core/ → project root)
    _REPO_ROOT = Path(__file__).resolve().parent.parent

    def _load_default_config(self):
        default_config_path = self._REPO_ROOT / "configs" / "overall.yaml"
        if not default_config_path.exists():
            raise FileNotFoundError(
                f"Required config file not found: {default_config_path}. "
                "NexusRec requires configs/overall.yaml to be present."
            )
        self._config = _load_yaml_mapping(default_config_path)

    def _load_model_config(self):
        model_config_path = self._REPO_ROOT / "configs" / "models" / f"{self.model}.yaml"
        if not model_config_path.exists():
            if self.model is None:
                return
            raise FileNotFoundError(
                f"Required model config file not found: {model_config_path}. "
                "Every model must declare its paradigm flags and model-specific config in configs/models/{Model}.yaml."
            )

        model_config = _load_yaml_mapping(model_config_path)
        self._deep_merge_config(model_config)

    def _load_dataset_config(self):
        dataset_config_path = self._REPO_ROOT / "configs" / "datasets" / f"{self.dataset}.yaml"
        if not dataset_config_path.exists():
            return

        dataset_config = _load_yaml_mapping(dataset_config_path)
        model_overrides = dataset_config.pop("model_overrides", {})
        if model_overrides and not isinstance(model_overrides, dict):
            raise ConfigValidationError(
                f"Dataset config field 'model_overrides' must be a mapping: {dataset_config_path}"
            )

        self._deep_merge_config(dataset_config)
        if not self.model:
            return

        model_specific_config = model_overrides.get(self.model, {})
        if model_specific_config and not isinstance(model_specific_config, dict):
            raise ConfigValidationError(
                f"Dataset model override for {self.model!r} must be a mapping: "
                f"{dataset_config_path}"
            )
        if model_specific_config:
            self._deep_merge_config(model_specific_config)

    def _apply_config_overrides(self):
        if self.config_dict:
            # Fail fast on override keys that match no key in the merged base schema.
            reject_unknown_override_keys(
                self._config, self.config_dict, ConfigValidationError
            )
            self._deep_merge_config(self.config_dict)
            # Explicit user overrides must win over a model-YAML top-level default.
            # flatten_nested_groups only promotes a nested-group key when it is
            # ABSENT at top level, so a user override passed via a nested group
            # (e.g. config_dict={"training": {"learning_rate": 0.5}} or
            # --param_overrides '{"training": {...}}') would otherwise be skipped
            # for any model whose YAML sets that key top-level. Promote the
            # user's flatten-group override keys to top-level so deep_merge
            # precedence holds for every runtime key, not just the CLI flags.
            for group_name in _FLATTEN_GROUPS:
                if group_name in self.config_dict and isinstance(
                    self.config_dict[group_name], dict
                ):
                    for key, value in self.config_dict[group_name].items():
                        base_value = None
                        if (
                            group_name in self._config
                            and isinstance(self._config[group_name], dict)
                            and key in self._config[group_name]
                        ):
                            base_value = self._config[group_name][key]
                        if (
                            key in self._config
                            and isinstance(self._config[key], dict)
                            and isinstance(value, dict)
                        ):
                            self._config[key] = deep_merge_dict(self._config[key], value)
                        elif isinstance(base_value, dict) and isinstance(value, dict):
                            self._config[key] = deep_merge_dict(base_value, value)
                        else:
                            self._config[key] = value

    def _post_process_config(self):
        _coerce_numeric_strings(self._config)
        reject_bad_keys(self._config, ConfigValidationError)

        if self.model:
            self._config["model"] = self.model
        if self.dataset:
            self._config["dataset"] = self.dataset

        flatten_nested_groups(self._config)
        self._validate_training()
        split_configs(self._config)
        normalize_hparams(self._config)
        assign_runtime_device(self._config)

        if "hyper_parameters" not in self._config:
            self._config["hyper_parameters"] = []
        set_paths(self._config)
        validate_parameter_ranges(self._config, ConfigValidationError)
        validate_export(self._config, ConfigValidationError)

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value

    def __contains__(self, key):
        return key in self._config

    def keys(self):
        return self._config.keys()

    def items(self):
        return self._config.items()

    def values(self):
        return self._config.values()

    def copy(self):
        new_config_manager = ConfigManager.__new__(ConfigManager)
        new_config_manager.model = self.model
        new_config_manager.dataset = self.dataset
        new_config_manager.config_dict = copy.deepcopy(self.config_dict)
        new_config_manager.trial_info = self.trial_info
        new_config_manager._config = copy.deepcopy(self._config)
        return new_config_manager

    def __str__(self):
        lines = []
        categories = {
            "Model": ["model", "dataset", "is_federated", "is_multimodal_model", "is_sequential"],
            "Training": [
                "max_epochs",
                "learning_rate",
                "weight_decay",
                "train_batch_size",
                "eval_batch_size",
            ],
            "Evaluation": ["metrics", "topk", "valid_metric", "valid_metric_bigger"],
            "System": ["device", "seed", "gpu_id"],
        }

        if self.trial_info:
            lines.append("-" * 25 + f"{self.trial_info}" + "-" * 25)
        else:
            lines.append("-" * 35 + " Configuration " + "-" * 35)

        hyper_parameters = []
        if "hyper_parameters" in self._config and self._config["hyper_parameters"] is not None:
            hyper_parameters = self._config["hyper_parameters"]
        if hyper_parameters:
            param_items = []
            for param in hyper_parameters:
                if param not in self._config:
                    continue
                value = self._config[param]
                if isinstance(value, float):
                    if abs(value) < 1e-3 or abs(value) >= 1e4:
                        param_items.append(f"{param}={value:.6g}")
                    else:
                        param_items.append(f"{param}={value}")
                else:
                    param_items.append(f"{param}={value}")
            if param_items:
                lines.append(f"Parameters: {', '.join(param_items)}")

        for category, keys in categories.items():
            category_items = [(key, value) for key, value in self._config.items() if key in keys]
            if not category_items:
                continue
            param_strings = []
            for key, value in category_items:
                if isinstance(value, list) and len(value) > 3:
                    if key in {"metrics", "topk", "hyper_parameters"}:
                        value_str = str(value)
                    else:
                        value_str = f"[{value[0]}...{value[-1]}]({len(value)})"
                elif isinstance(value, str) and len(str(value)) > 40:
                    value_str = f"{str(value)[:37]}..."
                else:
                    value_str = str(value)
                param_strings.append(f"{key}={value_str}")
            lines.append(f"\n{category}: " + ", ".join(param_strings))

        return "\n".join(lines)

    def _deep_merge_config(self, override_config: Dict[str, Any]):
        self._config = deep_merge_dict(self._config, override_config)

    def _validate_training(self):
        validate_training(self._config, ConfigValidationError)


__all__ = [
    "ConfigManager",
    "ConfigValidationError",
    "BAD_KEYS",
    "BAD_MODEL_ALIAS_KEYS",
    "BAD_ROOT_KEYS",
    "extract_evaluation_params",
    "extract_federated_params",
    "extract_training_params",
]
