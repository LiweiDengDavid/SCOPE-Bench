# coding: utf-8
"""
Training Environment — config creation, logging, data loading, HPO env setup.
"""

from __future__ import annotations

import os
import platform
import logging
from typing import Any, Dict, Tuple

import torch

from ..config import ConfigManager
from ..runtime.logger import init_logger
from ..utils.training import init_seed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def prepare_data(config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Build train/valid/test loaders for the resolved runtime config."""
    if config["is_sequential"]:
        return _prepare_sequential_data(config)

    return _prepare_standard_data(config)


def _prepare_sequential_data(config: Dict[str, Any]) -> Tuple[Any, Any, Any] | None:
    logger = logging.getLogger("nexusrec")
    logger.info("Preparing sequential recommendation data...")

    from ..sequential.integration import auto_setup as sequential_auto_setup
    from ..model_registry import get_model

    model_name = config["model"]
    model_class = get_model(model_name)

    from ..data.dataset import RecDataset
    dataset = RecDataset(config)
    components = sequential_auto_setup(
        model_class=model_class,
        config=config,
        inter_feat=dataset.df,
        user_num=dataset.user_num,
        item_num=dataset.item_num,
    )

    if "data" not in components:
        raise RuntimeError(
            f"Sequential integration for {model_name} returned no 'data' key. "
            f"Check sequential/integration.py auto_setup()."
        )

    data_components = components["data"]
    logger.info(f"[{config['dataset']} stats] Sequential dataset prepared")
    logger.info(f"Train sequences: {len(data_components['dataset'].train_seq)}")
    logger.info(f"Valid sequences: {len(data_components['dataset'].valid_seq)}")
    logger.info(f"Test sequences: {len(data_components['dataset'].test_seq)}")
    return (
        data_components["train_dataloader"],
        data_components["valid_dataloader"],
        data_components["test_dataloader"],
    )


def _prepare_standard_data(config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    logger = logging.getLogger("nexusrec")
    logger.info("Preparing standard recommendation data...")

    from ..data.dataset import RecDataset
    from ..data.pipeline import create_loaders

    dataset = RecDataset(config)

    logger.info(f"[{config['dataset']} stats] Overall: {dataset}")

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info(f"[{config['dataset']} stats] Train:   {train_dataset}")
    logger.info(f"[{config['dataset']} stats] Valid:   {valid_dataset}")
    logger.info(f"[{config['dataset']} stats] Test:    {test_dataset}")

    return create_loaders(config, train_dataset, valid_dataset, test_dataset)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def prepare_env(
    model: str,
    dataset: str,
    config_dict: Dict[str, Any],
    setup_logging: bool = True,
    trial_info: str = None,
) -> Tuple[Dict[str, Any], Any, Any, Any]:
    """Build config, logging, and loaders for a training run."""
    config = ConfigManager(model, dataset, config_dict, trial_info=trial_info)

    if setup_logging:
        init_logger(config)
        logger = logging.getLogger("nexusrec")
        if not getattr(prepare_env, "_dir_logged", False):
            logger.info(f"██ Directory: {os.getcwd()} on Server: {platform.node()} ██")
            prepare_env._dir_logged = True
        logger.info(config)

    init_seed(config["seed"], config["deterministic_algorithms"])
    train_data, valid_data, test_data = prepare_data(config)

    return config, train_data, valid_data, test_data


def setup_hpo_environment(config: Dict[str, Any]) -> None:
    """Apply runtime settings for HPO trials from the optimization: config section."""
    optimization = config["optimization"]
    for key in ("save_model", "print_model_info", "eval_final_test"):
        if key in optimization:
            config[key] = optimization[key]

    # Per-epoch training resume is incompatible with HPO: all trials share one
    # checkpoint_dir/resume_state.pth (the per-trial dir is used only for the
    # best-model save), so later trials could reuse another trial's weights and
    # start_epoch. HPO has its own --no-resume trial-search resumption; force
    # per-trial training resume off here.
    config["resume_training"] = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
