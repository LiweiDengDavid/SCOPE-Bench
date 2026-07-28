# coding: utf-8
"""Recommendation-list artifact helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


class Recommendation:
    """Read, validate, and reshape recommendation-list artifacts."""

    TYPE = "recommendation_list"
    ID_SPACE = "nexusrec_internal"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    FORMATS = frozenset({JSON, JSONL, CSV, TSV})
    USER_WEIGHT_KEY = "user_embedding.weight"
    ITEM_WEIGHT_KEY = "item_embedding.weight"

    @classmethod
    def columns(cls, include_scores: bool) -> List[str]:
        fields = ["user_id", "rank", "item_id"]
        if include_scores:
            fields.append("score")
        return fields

    @classmethod
    def row_grain(cls, fmt: str) -> str:
        if fmt in (cls.JSON, cls.JSONL):
            return "user"
        if fmt in (cls.CSV, cls.TSV):
            return "recommendation"
        raise ValueError(f"Unsupported recommendation artifact format: {fmt}")

    @classmethod
    def delimiter(cls, fmt: str) -> str:
        if fmt == cls.CSV:
            return ","
        if fmt == cls.TSV:
            return "\t"
        raise ValueError(f"Recommendation format is not delimited: {fmt}")

    @classmethod
    def metadata_path(cls, path: str | Path) -> Path:
        artifact_path = Path(path)
        return artifact_path.with_suffix(artifact_path.suffix + ".metadata.json")

    @classmethod
    def load_metadata(cls, path: str | Path) -> Dict[str, Any]:
        artifact_path = Path(path)
        metadata = json.loads(
            cls.metadata_path(artifact_path).read_text(encoding="utf-8")
        )
        cls._validate_metadata(artifact_path, metadata)
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        artifact_path = Path(path)
        metadata = cls.load_metadata(artifact_path)
        fmt = metadata["format"]
        if fmt == cls.JSON:
            records = cls._read_json(artifact_path)
        elif fmt == cls.JSONL:
            records = cls._read_jsonl(artifact_path)
        elif fmt in (cls.CSV, cls.TSV):
            records = cls._read_delimited(
                artifact_path,
                cls.delimiter(fmt),
                metadata["include_scores"],
            )
        else:
            raise ValueError(f"Unsupported recommendation artifact format: {fmt}")
        cls._validate_records(records, metadata)
        return records, metadata

    @classmethod
    def to_rows(
        cls,
        records: Iterable[Dict[str, Any]],
        include_scores: bool,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for record in records:
            for item in record["items"]:
                row = {
                    "user_id": record["user_id"],
                    "rank": item["rank"],
                    "item_id": item["item_id"],
                }
                if include_scores:
                    row["score"] = item["score"]
                rows.append(row)
        return rows

    @classmethod
    def load_embeddings(
        cls,
        path: str | Path,
        checkpoint: str | Path,
        user_key: str = USER_WEIGHT_KEY,
        item_key: str = ITEM_WEIGHT_KEY,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        records, metadata = cls.load(path)
        payload = torch.load(
            Path(checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        state = cls._checkpoint_state(payload)
        user_table = cls._embedding_table(state, user_key)
        item_table = cls._embedding_table(state, item_key)
        user_ids, item_ids = cls._id_tensors(records)
        model_item_ids = item_ids + metadata["model_item_id_offset"]

        cls._validate_indices(
            user_ids,
            user_table.shape[0],
            "user_id",
            user_key,
        )
        cls._validate_indices(
            model_item_ids,
            item_table.shape[0],
            "model_item_id",
            item_key,
        )
        embeddings = {
            "user_ids": user_ids,
            "item_ids": item_ids,
            "model_item_ids": model_item_ids,
            "user_embeddings": user_table[user_ids].clone(),
            "item_embeddings": item_table[model_item_ids].clone(),
        }
        return embeddings, metadata

    @classmethod
    def _validate_metadata(cls, path: Path, metadata: Dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            raise ValueError("Recommendation metadata must be a mapping.")
        if metadata["artifact_type"] != cls.TYPE:
            raise ValueError(
                "Recommendation metadata artifact_type must be "
                f"{cls.TYPE!r}."
            )
        fmt = cls._require_string(metadata["format"], "metadata.format")
        if fmt not in cls.FORMATS:
            raise ValueError(f"Unsupported recommendation artifact format: {fmt}")
        if path.suffix != f".{fmt}":
            raise ValueError(
                "Recommendation metadata format does not match artifact suffix: "
                f"format={fmt}, suffix={path.suffix}"
            )
        id_space = cls._require_string(metadata["id_space"], "metadata.id_space")
        if id_space != cls.ID_SPACE:
            raise ValueError(
                f"Recommendation metadata id_space must be {cls.ID_SPACE!r}."
            )
        cls._require_int(metadata["id_index_base"], "metadata.id_index_base")
        cls._require_int(metadata["rank_base"], "metadata.rank_base")
        if metadata["id_index_base"] != 0:
            raise ValueError("Recommendation metadata id_index_base must be 0.")
        if metadata["rank_base"] != 1:
            raise ValueError("Recommendation metadata rank_base must be 1.")
        cls._require_non_negative_int(
            metadata["model_item_id_offset"],
            "metadata.model_item_id_offset",
        )
        if not isinstance(metadata["include_scores"], bool):
            raise ValueError("Recommendation metadata include_scores must be boolean.")
        cls._require_positive_int(metadata["topk"], "metadata.topk")
        cls._require_positive_int(metadata["row_count"], "metadata.row_count")
        cls._require_positive_int(
            metadata["exported_user_count"],
            "metadata.exported_user_count",
        )
        cls._require_positive_int(
            metadata["recommendation_count"],
            "metadata.recommendation_count",
        )
        cls._require_positive_int(metadata["user_count"], "metadata.user_count")
        cls._require_positive_int(metadata["item_count"], "metadata.item_count")
        expected_grain = cls.row_grain(fmt)
        row_grain = cls._require_string(metadata["row_grain"], "metadata.row_grain")
        if row_grain != expected_grain:
            raise ValueError(
                "Recommendation metadata row_grain does not match format: "
                f"row_grain={row_grain}, format={fmt}"
            )

    @classmethod
    def _validate_records(
        cls,
        records: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        if len(records) != metadata["exported_user_count"]:
            raise ValueError(
                "Recommendation exported_user_count mismatch: "
                f"metadata={metadata['exported_user_count']}, actual={len(records)}"
            )
        if (
            metadata["format"] in (cls.JSON, cls.JSONL)
            and metadata["row_count"] != len(records)
        ):
            raise ValueError(
                "Recommendation row_count mismatch: "
                f"metadata={metadata['row_count']}, actual={len(records)}"
            )

        seen_users = set()
        recommendation_count = 0
        for record in records:
            user_id = cls._require_int(record["user_id"], "user_id")
            if user_id < 0 or user_id >= metadata["user_count"]:
                raise ValueError(
                    "Recommendation user_id is out of range: "
                    f"user_id={user_id}, user_count={metadata['user_count']}"
                )
            if user_id in seen_users:
                raise ValueError(f"Recommendation has duplicate user_id={user_id}.")
            seen_users.add(user_id)
            cls._validate_items(record["items"], metadata, user_id)
            recommendation_count += len(record["items"])

        if recommendation_count != metadata["recommendation_count"]:
            raise ValueError(
                "Recommendation recommendation_count mismatch: "
                f"metadata={metadata['recommendation_count']}, actual={recommendation_count}"
            )
        if (
            metadata["format"] in (cls.CSV, cls.TSV)
            and metadata["row_count"] != recommendation_count
        ):
            raise ValueError(
                "Recommendation row_count mismatch: "
                f"metadata={metadata['row_count']}, actual={recommendation_count}"
            )

    @classmethod
    def _read_json(cls, path: Path) -> List[Dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        if not isinstance(payload, list):
            raise ValueError("Recommendation JSON must contain a list of records.")
        return payload

    @classmethod
    def _read_jsonl(cls, path: Path) -> List[Dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    @classmethod
    def _read_delimited(
        cls,
        path: Path,
        delimiter: str,
        include_scores: bool,
    ) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj, delimiter=delimiter)
            columns = cls.columns(include_scores)
            if reader.fieldnames != columns:
                raise ValueError(
                    "Recommendation table header mismatch: "
                    f"expected={columns}, actual={reader.fieldnames}"
                )
            items_by_user: Dict[int, List[Dict[str, Any]]] = {}
            for raw_row in reader:
                user_id = int(raw_row["user_id"])
                item = {
                    "rank": int(raw_row["rank"]),
                    "item_id": int(raw_row["item_id"]),
                }
                if include_scores:
                    item["score"] = float(raw_row["score"])
                if user_id not in items_by_user:
                    items_by_user[user_id] = []
                items_by_user[user_id].append(item)
        return [
            {"user_id": user_id, "items": sorted(items, key=lambda item: item["rank"])}
            for user_id, items in items_by_user.items()
        ]

    @classmethod
    def _id_tensors(
        cls,
        records: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_ids = torch.tensor(
            [record["user_id"] for record in records],
            dtype=torch.long,
        )
        item_ids = torch.tensor(
            [
                [item["item_id"] for item in record["items"]]
                for record in records
            ],
            dtype=torch.long,
        )
        return user_ids, item_ids

    @staticmethod
    def _checkpoint_state(checkpoint: Any) -> Dict[str, Any]:
        if not isinstance(checkpoint, dict):
            raise ValueError("Recommendation checkpoint must be a mapping.")
        if "model_state_dict" not in checkpoint:
            raise ValueError("Recommendation checkpoint is missing model_state_dict.")
        state = checkpoint["model_state_dict"]
        if not isinstance(state, dict):
            raise ValueError(
                "Recommendation checkpoint model_state_dict must be a mapping."
            )
        return state

    @classmethod
    def _embedding_table(
        cls,
        state: Dict[str, Any],
        key: str,
    ) -> torch.Tensor:
        if key not in state:
            raise ValueError(
                f"Recommendation checkpoint is missing embedding table: {key}"
            )
        value = state[key]
        if not torch.is_tensor(value):
            raise ValueError(
                f"Recommendation checkpoint {key} must be a tensor."
            )
        if value.ndim != 2:
            raise ValueError(
                f"Recommendation checkpoint {key} must be a 2-D embedding table."
            )
        return value.detach().cpu()

    @staticmethod
    def _validate_indices(
        ids: torch.Tensor,
        table_rows: int,
        id_label: str,
        table_label: str,
    ) -> None:
        if ids.numel() == 0:
            raise ValueError(f"Recommendation {id_label} tensor is empty.")
        max_id = int(ids.max().item())
        min_id = int(ids.min().item())
        if min_id < 0 or max_id >= table_rows:
            raise ValueError(
                "Recommendation checkpoint embedding table is too small: "
                f"{id_label} range=[{min_id}, {max_id}], "
                f"{table_label} rows={table_rows}"
            )

    @classmethod
    def _validate_items(
        cls,
        items: Any,
        metadata: Dict[str, Any],
        user_id: int,
    ) -> None:
        if not isinstance(items, list):
            raise ValueError(f"Recommendation items must be a list: user_id={user_id}")
        if len(items) != metadata["topk"]:
            raise ValueError(
                "Recommendation item list length does not match topk: "
                f"user_id={user_id}, items={len(items)}, topk={metadata['topk']}"
            )
        seen_ranks = set()
        seen_items = set()
        for item in items:
            rank = cls._require_int(item["rank"], "rank")
            item_id = cls._require_int(item["item_id"], "item_id")
            if rank < 1 or rank > metadata["topk"]:
                raise ValueError(
                    f"Recommendation rank is out of range: rank={rank}, topk={metadata['topk']}"
                )
            if rank in seen_ranks:
                raise ValueError(
                    f"Recommendation has duplicate rank={rank} for user_id={user_id}."
                )
            seen_ranks.add(rank)
            if item_id < 0 or item_id >= metadata["item_count"]:
                raise ValueError(
                    "Recommendation item_id is out of range: "
                    f"item_id={item_id}, item_count={metadata['item_count']}"
                )
            if item_id in seen_items:
                raise ValueError(
                    f"Recommendation has duplicate item_id={item_id} for user_id={user_id}."
                )
            seen_items.add(item_id)
            if metadata["include_scores"]:
                cls._require_score(item["score"], "score")
            elif "score" in item:
                raise ValueError(
                    f"Recommendation score is present but include_scores=false: user_id={user_id}"
                )
        expected_ranks = set(range(1, metadata["topk"] + 1))
        if seen_ranks != expected_ranks:
            raise ValueError(
                "Recommendation ranks must be contiguous from 1 to topk: "
                f"user_id={user_id}"
            )

    @staticmethod
    def _require_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Recommendation {label} must be an integer, got {value!r}")
        return value

    @staticmethod
    def _require_string(value: Any, label: str) -> str:
        if not isinstance(value, str) or value == "":
            raise ValueError(f"Recommendation {label} must be a non-empty string.")
        return value

    @classmethod
    def _require_positive_int(cls, value: Any, label: str) -> int:
        value = cls._require_int(value, label)
        if value < 1:
            raise ValueError(f"Recommendation {label} must be positive, got {value!r}")
        return value

    @classmethod
    def _require_non_negative_int(cls, value: Any, label: str) -> int:
        value = cls._require_int(value, label)
        if value < 0:
            raise ValueError(
                f"Recommendation {label} must be non-negative, got {value!r}"
            )
        return value

    @staticmethod
    def _require_score(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Recommendation {label} must be numeric, got {value!r}")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"Recommendation {label} is NaN/Inf.")
        return score
