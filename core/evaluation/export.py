# coding: utf-8
"""Recommendation-list export for final evaluation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from ..utils.recommendation import Recommendation
from ..utils.result import Result
from ..utils.training import artifact_token
from .export_contract import (
    require_flag,
    require_topk,
    validate_section,
)


def is_enabled(config: Dict[str, Any]) -> bool:
    if "export" not in config:
        return False
    section = config["export"]
    if not isinstance(section, dict):
        raise ValueError("output.export must be a mapping.")
    require_flag(section["enabled"], "enabled", ValueError)
    return section["enabled"]


def include_scores(config: Dict[str, Any]) -> bool:
    section = config["export"]
    require_flag(section["include_scores"], "include_scores", ValueError)
    return section["include_scores"]


def validate_config(config: Dict[str, Any]) -> None:
    validate_section(
        config,
        ValueError,
        legacy_conflict_message=(
            "output.export is the recommendation-list export. "
            "Disable legacy save_recommended_topk to avoid producing two "
            "different top-k artifacts."
        ),
    )


def _validate_count(value: Any, label: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value < 1
    ):
        raise ValueError(
            f"Recommendation export {label} must be a positive integer, got {value!r}"
        )
    return int(value)


def _validate_item_id_offset(item_id_offset: Any) -> int:
    if (
        not isinstance(item_id_offset, (int, np.integer))
        or isinstance(item_id_offset, (bool, np.bool_))
        or item_id_offset < 0
    ):
        raise ValueError(
            "Recommendation export item_id_offset must be a non-negative "
            f"integer, got {item_id_offset!r}"
        )
    return int(item_id_offset)


def _validate_eval_idx(idx: Any) -> int:
    if (
        not isinstance(idx, (int, np.integer))
        or isinstance(idx, (bool, np.bool_))
        or idx < 0
    ):
        raise ValueError(
            "Recommendation export eval idx must be a non-negative "
            f"integer, got {idx!r}"
        )
    return int(idx)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _coerce_integer_array(value: Any, label: str) -> np.ndarray:
    array = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Recommendation export {label} must use integer dtype.")
    return array.astype(int)


def _coerce_topk_indices(topk_indices: Any) -> np.ndarray:
    items = _coerce_integer_array(topk_indices, "topk_indices")
    if items.ndim != 2:
        raise ValueError(
            "Recommendation export topk_indices must be a 2-D matrix, "
            f"got shape {items.shape}."
        )
    if items.shape[1] < 1:
        raise ValueError(
            "Recommendation export topk_indices must have at least one item column."
        )
    return items


def _coerce_score_matrix(value: Any) -> np.ndarray:
    scores = _to_numpy(value)
    if not (
        np.issubdtype(scores.dtype, np.integer)
        or np.issubdtype(scores.dtype, np.floating)
    ):
        raise ValueError("Recommendation export topk_scores must use numeric dtype.")
    return scores.astype(float)


def _resolve_topk(config: Dict[str, Any], available_topk: int) -> int:
    requested = config["export"]["topk"]
    if requested is None:
        return int(available_topk)
    require_topk(requested, ValueError)
    if requested > available_topk:
        raise ValueError(
            f"output.export.topk={requested} exceeds available evaluated top-k "
            f"width {available_topk}."
        )
    return requested


def _build_rows(
    config: Dict[str, Any],
    eval_users: Any,
    topk_indices: Any,
    topk_scores: Any,
    topk: int,
    user_count: int,
    item_count: int,
    item_id_offset: int,
) -> List[Dict[str, Any]]:
    users = _coerce_integer_array(eval_users, "eval_users").reshape(-1)
    items = _coerce_topk_indices(topk_indices)
    if users.shape[0] != items.shape[0]:
        raise ValueError(
            "Recommendation export requires eval users and top-k rows to match, "
            f"got {users.shape[0]} users and {items.shape[0]} rows."
        )

    with_scores = config["export"]["include_scores"]
    score_values = None
    if with_scores:
        if topk_scores is None:
            raise ValueError("output.export.include_scores=true requires top-k scores.")
        score_values = _coerce_score_matrix(topk_scores)
        if score_values.shape != items.shape:
            raise ValueError(
                "Recommendation export score matrix must match top-k index shape, "
                f"got scores={score_values.shape}, topk={items.shape}."
            )
        if not np.isfinite(score_values[:, :topk]).all():
            raise ValueError("Recommendation export scores contain NaN/Inf values.")

    rows: List[Dict[str, Any]] = []
    seen_users = set()
    for eval_index, user_id in enumerate(users.tolist()):
        user_id = int(user_id)
        if user_id < 0 or user_id >= user_count:
            raise ValueError(
                f"Recommendation export user_id is out of range: "
                f"user_id={user_id}, user_count={user_count}."
            )
        if user_id in seen_users:
            raise ValueError(
                f"Recommendation export has duplicate user_id={user_id}."
            )
        seen_users.add(user_id)
        seen_items = set()
        ranked_items: List[Dict[str, Any]] = []
        for local_rank, model_item_id in enumerate(items[eval_index, :topk].tolist(), 1):
            item_id = int(model_item_id) - item_id_offset
            if item_id < 0:
                raise ValueError(
                    f"Recommendation export item_id {model_item_id} maps below zero "
                    f"with offset {item_id_offset}."
                )
            if item_id >= item_count:
                raise ValueError(
                    f"Recommendation export item_id is out of range: "
                    f"item_id={item_id}, item_count={item_count}."
                )
            if item_id in seen_items:
                raise ValueError(
                    f"Recommendation export has duplicate item_id={item_id} "
                    f"for eval_index={eval_index}."
                )
            seen_items.add(item_id)
            item = {"rank": int(local_rank), "item_id": int(item_id)}
            if with_scores:
                item["score"] = float(score_values[eval_index, local_rank - 1])
            ranked_items.append(item)
        if len(ranked_items) != topk:
            raise ValueError(
                "Recommendation export item list length does not match topk: "
                f"user_id={user_id}, items={len(ranked_items)}, topk={topk}."
            )
        row = {"user_id": user_id, "items": ranked_items}
        rows.append(row)
    if not rows:
        raise ValueError("Recommendation export produced no rows.")
    return rows


def _base_path(config: Dict[str, Any], idx: int, topk: int) -> Path:
    export_path = str(config["export"]["path"])
    output_dir = Path(export_path) if export_path else Path(config["paths"]["save"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_tag = (
        f"{config['type']}.{config['comment']}.seed{config['seed']}."
        f"idx{idx}.top{topk}.recommendations"
    )
    return output_dir / (
        f"[{config['model']}]-[{config['dataset']}]-"
        f"[{artifact_token(raw_tag)}]"
    )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(rows, file_obj, ensure_ascii=False, allow_nan=False)
        file_obj.write("\n")


def _write_delimited(
    path: Path,
    rows: List[Dict[str, Any]],
    delimiter: str,
    with_scores: bool,
) -> None:
    fieldnames = Recommendation.columns(with_scores)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(Recommendation.to_rows(rows, with_scores))


def _normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        raise ValueError("Recommendation export metrics must be a mapping.")
    normalized: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"Recommendation export metric {key!r} must be numeric, got {value!r}"
            )
        metric_value = float(value)
        if not np.isfinite(metric_value):
            raise ValueError(f"Recommendation export metric {key!r} is NaN/Inf.")
        normalized[str(key)] = metric_value
    return normalized


def _normalize_metadata_scalar(key: str, value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"Recommendation export metadata field {key!r} is NaN/Inf.")
        return scalar
    raise ValueError(
        f"Recommendation export metadata field {key!r} is not JSON-serializable: {value!r}"
    )


def _normalize_metadata_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): _normalize_metadata_scalar(str(key), value)
        for key, value in record.items()
    }


def _write_metadata(
    path: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    provenance: Dict[str, Any],
    fmt: str,
    row_grain: str,
    row_count: int,
    exported_user_count: int,
    recommendation_count: int,
    topk: int,
    idx: int,
    user_count: int,
    item_count: int,
    item_id_offset: int,
) -> None:
    section = config["export"]
    metadata = {
        "artifact_type": Recommendation.TYPE,
        "format": fmt,
        "model": config["model"],
        "dataset": config["dataset"],
        "type": config["type"],
        "comment": config["comment"],
        "seed": config["seed"],
        "split": section["split"],
        "eval_idx": int(idx),
        "topk": int(topk),
        "include_scores": section["include_scores"],
        "id_space": Recommendation.ID_SPACE,
        "id_index_base": 0,
        "user_id_semantics": "NexusRec internal zero-based user index",
        "item_id_semantics": "NexusRec internal zero-based item index",
        "items_semantics": "ordered recommendation objects; rank=1 is top-1",
        "row_grain": row_grain,
        "list_order": "rank_ascending",
        "rank_base": 1,
        "sorted_by": "score_desc",
        "score_direction": "higher_is_better",
        "score_comparability": "same_user_same_run",
        "score_semantics": (
            "post-mask raw ranking scores; higher scores rank earlier within "
            "the same user/run"
        ),
        "metrics": metrics,
        "row_count": int(row_count),
        "exported_user_count": int(exported_user_count),
        "recommendation_count": int(recommendation_count),
        "user_count": int(user_count),
        "item_count": int(item_count),
        "user_id_field": config["USER_ID_FIELD"],
        "item_id_field": config["ITEM_ID_FIELD"],
        "model_item_id_offset": int(item_id_offset),
        **provenance,
    }
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            metadata,
            file_obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file_obj.write("\n")


def write_recommendations(
    config: Dict[str, Any],
    eval_users: Any,
    topk_indices: Any,
    topk_scores: Any,
    idx: int,
    metrics: Dict[str, Any],
    user_count: int,
    item_count: int,
    item_id_offset: int = 0,
) -> List[str]:
    validate_config(config)
    topk_array = _coerce_topk_indices(topk_indices)
    available_topk = int(topk_array.shape[1])
    topk = _resolve_topk(config, available_topk)
    item_id_offset = _validate_item_id_offset(item_id_offset)
    idx = _validate_eval_idx(idx)
    user_count = _validate_count(user_count, "user_count")
    item_count = _validate_count(item_count, "item_count")
    rows = _build_rows(
        config,
        eval_users,
        topk_array,
        topk_scores,
        topk,
        user_count,
        item_count,
        item_id_offset,
    )
    metrics = _normalize_metrics(metrics)
    provenance = _normalize_metadata_record(Result.provenance(config))

    base_path = _base_path(config, idx, topk)
    paths: List[str] = []
    exported_user_count = len(rows)
    recommendation_count = sum(len(row["items"]) for row in rows)
    for fmt in config["export"]["formats"]:
        file_path = base_path.with_suffix(f".{fmt}")
        if fmt == Recommendation.JSON:
            _write_json(file_path, rows)
        elif fmt == Recommendation.JSONL:
            _write_jsonl(file_path, rows)
        elif fmt in (Recommendation.CSV, Recommendation.TSV):
            _write_delimited(
                file_path,
                rows,
                Recommendation.delimiter(fmt),
                config["export"]["include_scores"],
            )
        else:
            raise ValueError(f"Unsupported output.export format: {fmt}")
        row_grain = Recommendation.row_grain(fmt)
        row_count = (
            exported_user_count
            if fmt in (Recommendation.JSON, Recommendation.JSONL)
            else recommendation_count
        )
        _write_metadata(
            Recommendation.metadata_path(file_path),
            config,
            metrics,
            provenance,
            fmt,
            row_grain,
            row_count,
            exported_user_count,
            recommendation_count,
            topk,
            idx,
            user_count,
            item_count,
            item_id_offset,
        )
        paths.append(str(file_path))
    return paths
