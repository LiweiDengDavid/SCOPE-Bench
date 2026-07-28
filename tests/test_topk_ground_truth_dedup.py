import logging

import numpy as np
import pandas as pd
import pytest
import torch

from core.data.dataloader import EvalDataLoader
from core.data.dataset import RecDataset
from core.evaluation.evaluator import TopKEvaluator
from core.federated.dataloader import FederatedDataLoader


def _config():
    return {
        "dataset": "unit",
        "data_path": "./datasets/",
        "USER_ID_FIELD": "userID",
        "ITEM_ID_FIELD": "itemID",
        "inter_splitting_label": "split_label",
        "device": torch.device("cpu"),
        "eval_batch_size": 8,
        "metrics": ["Recall", "NDCG", "MAP", "Precision"],
        "topk": [2],
        "save_recommended_topk": False,
        "item_bucket_metrics": False,
    }


def _datasets(config):
    train_df = pd.DataFrame(
        {
            "userID": [0, 1],
            "itemID": [0, 1],
        }
    )
    eval_df = pd.DataFrame(
        {
            "userID": [0, 0, 0, 1, 1],
            "itemID": [2, 2, 3, 3, 3],
        }
    )
    train = RecDataset.from_dataframe(config, train_df, item_num=4, user_num=2)
    evaluation = RecDataset.from_dataframe(config, eval_df, item_num=4, user_num=2)
    return train, evaluation


def test_topk_loader_deduplicates_ground_truth_and_metric_denominators(caplog):
    config = _config()
    train, evaluation = _datasets(config)
    loader = EvalDataLoader(
        config,
        evaluation,
        additional_dataset=train,
        batch_size=2,
    )

    assert [items.tolist() for items in loader.get_eval_items()] == [[2, 3], [3]]
    assert loader.get_eval_len_list().tolist() == [2, 1]
    assert loader.eval_duplicate_count == 2

    recommendations = [torch.tensor([[2, 3], [3, 2]])]
    evaluator = TopKEvaluator(config)
    with caplog.at_level(logging.INFO, logger="nexusrec"):
        metrics = evaluator.evaluate(recommendations, loader)
        evaluator.evaluate(recommendations, loader)

    assert metrics["Recall@2"] == pytest.approx(1.0)
    assert metrics["NDCG@2"] == pytest.approx(1.0)
    assert metrics["MAP@2"] == pytest.approx(1.0)
    assert metrics["Precision@2"] == pytest.approx(0.75)
    dedup_logs = [
        record for record in caplog.records
        if "Top-K evaluation deduplicated 2 repeated" in record.getMessage()
    ]
    assert len(dedup_logs) == 1


def test_federated_topk_loader_uses_the_same_unique_ground_truth():
    config = _config()
    train, evaluation = _datasets(config)
    loader = FederatedDataLoader(
        config,
        evaluation,
        stage="test",
        additional_dataset=train,
        train_dataset=train,
        batch_size=2,
    )

    assert loader.get_eval_items() == [[2, 3], [3]]
    assert loader.get_eval_len_list() == [2, 1]
    assert loader.eval_duplicate_count == 2


def test_stable_dedup_preserves_first_seen_item_order():
    config = _config()
    train, _evaluation = _datasets(config)
    eval_df = pd.DataFrame(
        {
            "userID": [0, 0, 0, 0],
            "itemID": [3, 2, 3, 1],
        }
    )
    evaluation = RecDataset.from_dataframe(
        config, eval_df, item_num=4, user_num=2
    )
    loader = EvalDataLoader(
        config,
        evaluation,
        additional_dataset=train,
        batch_size=2,
    )

    np.testing.assert_array_equal(loader.get_eval_items()[0], np.array([3, 2, 1]))
