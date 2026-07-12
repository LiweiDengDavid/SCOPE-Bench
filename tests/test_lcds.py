import json
import math
from pathlib import Path

import numpy as np
import torch

from core.evaluation.evaluator import TopKEvaluator
from core.evaluation.lcds import (
    build_cds_gain_table,
    build_lcds_result_dict,
    configured_cds_gain_table,
    lcds_metric_arrays,
)
from core.utils.result import Result


def test_lcds_metric_arrays_use_arithmetic_and_exposure_weighting():
    topk_index = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    gains = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)

    arrays = lcds_metric_arrays(topk_index, gains)

    assert np.allclose(arrays["A-LCDS"], [0.5, 0.5, 0.5])
    discounts = np.asarray([1.0 / math.log2(rank + 1) for rank in [1, 2, 3]])
    row_one = np.cumsum(np.asarray([0.0, 0.5, 1.0]) * discounts) / np.cumsum(discounts)
    row_two = np.cumsum(np.asarray([1.0, 0.5, 0.0]) * discounts) / np.cumsum(discounts)
    assert np.allclose(arrays["E-LCDS"], (row_one + row_two) / 2)

    result = build_lcds_result_dict(topk_index, gains, [1, 3])
    assert result["A-LCDS@1"] == 0.5
    assert result["A-LCDS@3"] == 0.5


def test_cds_gain_table_maps_internal_items_through_source_pid(tmp_path: Path):
    dataset_dir = tmp_path / "ShortVideoFull"
    dataset_dir.mkdir()
    (dataset_dir / "id_mappings.json").write_text(
        json.dumps({"item_raw_to_new": {"200": 0, "100": 1}}),
        encoding="utf-8",
    )
    (dataset_dir / "items.json").write_text(
        json.dumps(
            [
                {"source_pid": 100, "video_id": 10},
                {"source_pid": 200, "video_id": 20},
            ]
        ),
        encoding="utf-8",
    )
    cds_jsonl = dataset_dir / "scores.jsonl"
    cds_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"video_id": 10, "score": 6}),
                json.dumps({"video_id": 20, "score": None}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    table = build_cds_gain_table(dataset_dir, cds_jsonl)

    assert table.gains.tolist() == [0.0, 1.0]
    assert table.stats["numeric_score_count"] == 1
    assert table.stats["null_score_count"] == 1
    assert table.stats["missing_score_count"] == 0


def test_topk_evaluator_outputs_ranking_and_lcds_at_the_same_cutoffs(tmp_path: Path):
    dataset_dir = tmp_path / "ShortVideoSampled"
    dataset_dir.mkdir()
    (dataset_dir / "id_mappings.json").write_text(
        json.dumps({"item_raw_to_new": {"100": 0, "200": 1}}),
        encoding="utf-8",
    )
    (dataset_dir / "items.json").write_text(
        json.dumps(
            [
                {"source_pid": 100, "video_id": 10},
                {"source_pid": 200, "video_id": 20},
            ]
        ),
        encoding="utf-8",
    )
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "\n".join(
            [
                json.dumps({"video_id": 10, "score": 0}),
                json.dumps({"video_id": 20, "score": 6}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "metrics": ["Recall", "NDCG", "Precision"],
        "topk": [1, 2],
        "save_recommended_topk": False,
        "item_bucket_metrics": False,
        "item_bucket_tail_quantile": 0.2,
        "item_bucket_head_quantile": 0.8,
        "data_path": str(tmp_path),
        "dataset": "ShortVideoSampled",
        "lcds": {
            "enabled": True,
            "dataset_dir": str(dataset_dir),
            "cds_jsonl": str(scores_path),
            "gain_divisor": 6.0,
        },
    }

    class Dataset:
        item_num = 2

    class EvalData:
        dataset = Dataset()

        @staticmethod
        def get_eval_items():
            return [np.asarray([1]), np.asarray([0])]

        @staticmethod
        def get_eval_len_list():
            return [1, 1]

    result = TopKEvaluator(config).evaluate(
        [torch.as_tensor([[1, 0], [0, 1]], dtype=torch.long)],
        EvalData(),
    )

    assert set(result) == {
        "Recall@1",
        "Recall@2",
        "NDCG@1",
        "NDCG@2",
        "Precision@1",
        "Precision@2",
        "A-LCDS@1",
        "A-LCDS@2",
        "E-LCDS@1",
        "E-LCDS@2",
    }
    assert result["A-LCDS@1"] == 0.5
    assert result["A-LCDS@2"] == 0.5
    assert result["E-LCDS@1"] == 0.5
    assert result["E-LCDS@2"] == 0.5

    result_path = tmp_path / "combined_metrics.csv"
    Result.write(result_path, result)
    saved = Result.load(result_path, required_columns=sorted(result))
    assert saved["Recall@1"] == result["Recall@1"]
    assert saved["A-LCDS@1"] == result["A-LCDS@1"]
    assert saved["E-LCDS@2"] == result["E-LCDS@2"]

    table = configured_cds_gain_table(config)
    shifted = build_lcds_result_dict(
        np.asarray([[1, 2]], dtype=np.int64),
        table.gains,
        [1, 2],
        item_id_offset=1,
    )
    assert shifted["A-LCDS@1"] == 0.0
    assert shifted["A-LCDS@2"] == 0.5

    legacy_config = dict(config)
    legacy_config.pop("lcds")
    legacy_config["lcpd"] = {
        "enabled": True,
        "dataset_dir": str(dataset_dir),
        "cpd_jsonl": str(scores_path),
        "gain_divisor": 6.0,
    }
    legacy_table = configured_cds_gain_table(legacy_config)
    assert legacy_table.gains.tolist() == table.gains.tolist()
