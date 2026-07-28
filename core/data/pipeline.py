# coding: utf-8
"""
Loader Pipeline - Unified loader construction for NexusRec
==========================================================

Build train/valid/test loaders for a given paradigm without leaking
paradigm-specific conditionals upward.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .dataloader import EvalDataLoader, TrainDataLoader


def _build_combined_eval_dataset(train_dataset: Any, valid_dataset: Any) -> Any:
    import copy
    import pandas as pd

    # Only combined.df is consumed downstream (EvalDataLoader reads .df/.uid_field/
    # .iid_field for test-time history masking). copy.copy carries the full-catalog
    # item_num/user_num from train_dataset; recomputing them from train+valid max()
    # would be both dead and wrong (an item id present only in test is excluded).
    combined = copy.copy(train_dataset)
    combined.df = pd.concat([train_dataset.df, valid_dataset.df], ignore_index=True)
    return combined


def _resolve_test_additional_dataset(
    config: Dict[str, Any],
    train_dataset: Any,
    valid_dataset: Any,
) -> Any:
    mask_mode = config["test_history_mask"]
    if mask_mode == "train_only":
        return train_dataset
    if mask_mode == "train_valid":
        return _build_combined_eval_dataset(train_dataset, valid_dataset)
    raise ValueError(
        "test_history_mask must be 'train_only' or 'train_valid', "
        f"got '{mask_mode}'"
    )


def create_loaders(
    config: Dict[str, Any],
    train_dataset: Any,
    valid_dataset: Any,
    test_dataset: Any,
) -> Tuple[Any, Any, Any]:
    """Build (train_loader, valid_loader, test_loader) for the current paradigm."""
    if config["is_federated"]:
        from ..federated.dataloader import FederatedDataLoader

        train_loader = FederatedDataLoader(
            config, train_dataset,
            batch_size=config["train_batch_size"], shuffle=True,
        )
        valid_loader = FederatedDataLoader(
            config, valid_dataset,
            additional_dataset=train_dataset, stage="valid",
            batch_size=config["eval_batch_size"],
        )
        test_additional_dataset = _resolve_test_additional_dataset(
            config, train_dataset, valid_dataset
        )
        # train_dataset keeps the popularity base (Novelty/item buckets) on the
        # pure train split even when the masking history is train+valid.
        test_loader = FederatedDataLoader(
            config, test_dataset,
            additional_dataset=test_additional_dataset, stage="test",
            batch_size=config["eval_batch_size"],
            train_dataset=train_dataset,
        )
    else:
        train_loader = TrainDataLoader(
            config, train_dataset,
            batch_size=config["train_batch_size"], shuffle=True,
        )
        valid_loader = EvalDataLoader(
            config, valid_dataset,
            additional_dataset=train_dataset,
            batch_size=config["eval_batch_size"],
        )
        test_additional_dataset = _resolve_test_additional_dataset(
            config, train_dataset, valid_dataset
        )
        # train_dataset keeps the popularity base (Novelty/item buckets) on the
        # pure train split even when the masking history is train+valid.
        test_loader = EvalDataLoader(
            config, test_dataset,
            additional_dataset=test_additional_dataset,
            batch_size=config["eval_batch_size"],
            train_dataset=train_dataset,
        )

    return train_loader, valid_loader, test_loader
