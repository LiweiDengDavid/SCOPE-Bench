import json
from pathlib import Path

from core.utils.recommendation import Recommendation


def test_recommendation_load_supports_grouped_json(tmp_path: Path):
    path = tmp_path / "recommendations.json"
    records = [
        {
            "user_id": 0,
            "items": [
                {"rank": 1, "item_id": 2, "score": 0.8},
                {"rank": 2, "item_id": 1, "score": 0.4},
            ],
        }
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    Recommendation.metadata_path(path).write_text(
        json.dumps(
            {
                "artifact_type": Recommendation.TYPE,
                "format": Recommendation.JSON,
                "id_space": Recommendation.ID_SPACE,
                "id_index_base": 0,
                "rank_base": 1,
                "model_item_id_offset": 0,
                "include_scores": True,
                "topk": 2,
                "row_count": 1,
                "exported_user_count": 1,
                "recommendation_count": 2,
                "user_count": 3,
                "item_count": 4,
                "row_grain": "user",
            }
        ),
        encoding="utf-8",
    )

    loaded, metadata = Recommendation.load(path)

    assert metadata["format"] == Recommendation.JSON
    assert loaded == records
