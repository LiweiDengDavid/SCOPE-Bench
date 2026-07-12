"""Model registry — profile validation and dynamic loading."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import yaml

_DEFAULT_MODEL = "VBPR"
_MODEL_HELP = "Model name (VBPR, FedAvg, BERT4Rec, LightGCN, etc.)"


# ---------------------------------------------------------------------------
# Profile — explicit model runtime profile validation
# ---------------------------------------------------------------------------

_PROFILE_KEYS = ("is_federated", "is_multimodal_model", "is_sequential")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "models"
_MODELS_SOURCE_ROOT = _REPO_ROOT / "models"


def load_model_profile(model_name: str) -> dict[str, bool]:
    """Load paradigm flags from the model YAML file."""
    config_path = _MODELS_CONFIG_ROOT / f"{model_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Model config not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # also catches None (empty YAML)
        raise ValueError(f"Model config must contain a mapping: {config_path}")
    missing = [key for key in _PROFILE_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"Model config {config_path} is missing required paradigm flag(s): {missing}"
        )
    return {key: bool(payload[key]) for key in _PROFILE_KEYS}


def infer_paradigm(profile: dict[str, bool]) -> str:
    """Infer the high-level paradigm label from boolean flags."""
    if profile["is_sequential"]:
        return "sequential/multimodal" if profile["is_multimodal_model"] else "sequential/id"
    if profile["is_federated"]:
        return "federated/multimodal" if profile["is_multimodal_model"] else "federated/id"
    return "centralized/multimodal" if profile["is_multimodal_model"] else "centralized/id"


def get_paradigm_root(model_name: str) -> Path:
    """Resolve the canonical paradigm root directory for a model."""
    profile = load_model_profile(model_name)
    return _MODELS_SOURCE_ROOT / Path(infer_paradigm(profile))


def get_model_module_path(model_name: str) -> str:
    """Resolve the canonical module import path for a model."""
    source_path = get_model_source_path(model_name)
    return ".".join(source_path.with_suffix("").parts)


def get_model_source_path(model_name: str) -> Path:
    """Resolve the canonical source file path for a model."""
    paradigm_root = get_paradigm_root(model_name)
    if not paradigm_root.exists():
        raise ImportError(f"Paradigm root does not exist for {model_name}: {paradigm_root}")

    expected_name = f"{model_name.lower()}.py"
    matches = sorted(paradigm_root.rglob(expected_name))
    if not matches:
        raise ImportError(
            f"Failed to find model source '{expected_name}' under paradigm root {paradigm_root}"
        )
    if len(matches) > 1:
        raise ImportError(
            f"Found multiple model sources for {model_name} under {paradigm_root}: {matches}"
        )
    return matches[0].relative_to(_REPO_ROOT)


# ---------------------------------------------------------------------------
# Loading — model and trainer dynamic loading helpers
# ---------------------------------------------------------------------------

def get_model(model_name):
    """Load a model class from its canonical package."""
    module_path = get_model_module_path(model_name)
    if importlib.util.find_spec(module_path) is None:
        raise ImportError(f"Failed to find model module '{module_path}' for {model_name}")

    model_module = importlib.import_module(module_path)
    if not hasattr(model_module, model_name):
        raise AttributeError(
            f"Module '{module_path}' does not define the expected class '{model_name}'"
        )
    return getattr(model_module, model_name)


def _get_federated_trainer(model_name: str):
    """Return the most-specific federated trainer for *model_name*."""
    import sys
    model_class = get_model(model_name)
    model_module = sys.modules[model_class.__module__]
    trainer_name = f"{model_name}Trainer"
    if hasattr(model_module, trainer_name):
        return getattr(model_module, trainer_name)
    from .federated.trainer import FederatedTrainer
    return FederatedTrainer


def get_trainer(model_name=None, is_federated=False, is_sequential=False):
    """Resolve trainer class for the requested model/paradigm.

    Resolution order:
    1. Sequential  -> SequentialTrainer (config flag or model-profile detection)
    2. Federated   -> model-specific ``{ModelName}Trainer`` if it exists, else FederatedTrainer
    3. Default     -> TrainerBase
    """
    if is_sequential:
        from .sequential.trainer import SequentialTrainer
        return SequentialTrainer

    if is_federated:
        if not model_name:
            raise ValueError("model_name is required for federated trainer resolution")
        return _get_federated_trainer(model_name)

    if model_name:
        profile = load_model_profile(model_name)
        if profile["is_sequential"]:
            from .sequential.trainer import SequentialTrainer
            return SequentialTrainer
        if profile["is_federated"]:
            return _get_federated_trainer(model_name)

    from .base.trainer import TrainerBase
    return TrainerBase
