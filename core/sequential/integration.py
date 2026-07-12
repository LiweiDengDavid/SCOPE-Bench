# coding: utf-8
"""
Sequential recommendation integration helpers.

Provides detection and automatic setup of sequential recommendation
components for the main NexusRec framework.
"""

import logging
from typing import Any, Dict

from .dataset import SequentialDataset
from .dataloader import SequentialDataLoader
from .recommender import SequentialRecommender

logger = logging.getLogger("nexusrec")


def detect_sequential_model(model_class, config: Dict[str, Any]) -> bool:
    """Detect if a model requires sequential data processing."""
    if config["is_sequential"]:
        return True

    if isinstance(model_class, type) and issubclass(model_class, SequentialRecommender):
        return True

    return False


def auto_setup(
    model_class,
    config: Dict[str, Any],
    inter_feat,
    user_num: int,
    item_num: int,
) -> Dict[str, Any]:
    """Automatically set up sequential recommendation data components.

    Returns ``{"data": data_components}`` (the sole key the caller consumes),
    or an empty dict if the model is not sequential. The trainer builds its own
    evaluator via ``SequentialTrainer._create_evaluator``, so no evaluator is
    constructed here.
    """
    if not detect_sequential_model(model_class, config):
        return {}

    logger.info("Setting up sequential recommendation components...")

    seq_dataset = SequentialDataset(
        config=config,
        inter_feat=inter_feat,
        user_num=user_num,
        item_num=item_num,
    )

    data_components = {
        "dataset": seq_dataset,
        "train_dataloader": SequentialDataLoader(config=config, dataset=seq_dataset, mode="train"),
        "valid_dataloader": SequentialDataLoader(config=config, dataset=seq_dataset, mode="valid"),
        "test_dataloader": SequentialDataLoader(config=config, dataset=seq_dataset, mode="test"),
    }

    return {"data": data_components}
