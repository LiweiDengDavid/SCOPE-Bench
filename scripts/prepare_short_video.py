#!/usr/bin/env python3
# coding: utf-8
"""Prepare sampled or full WWW2025 short-video data for SCOPE-Bench.

This script converts interaction logs to NexusRec's required `inter.csv` format
and aligns per-video image features to contiguous item ids.

Default positive rule:
    click=True and hate=False

Supported positive rules:
    - click
    - watch3
    - watch50

Default split/filter contract:
    - min_user_interactions=4
    - temporal_ratio=0.8/0.1/0.1
    - sentence_transformer text features
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


LOGGER = logging.getLogger("prepare_short_video")

TEXT_TITLE_FIELDS = (
    "source_match_title_cn",
    "source_match_title",
    "source_title_cn",
    "source_title",
    "title",
    "caption",
)
TEXT_BRAND_FIELDS = ("brand",)
TEXT_CATEGORY_FIELDS = (
    "category",
    "category_cn",
    "first_level_category",
    "second_level_category",
    "third_level_category",
    "first_level_category_cn",
    "second_level_category_cn",
    "third_level_category_cn",
)
TEXT_DESCRIPTION_FIELDS = (
    "description",
    "asr_text",
    "asr_text_cn",
)


def raise_csv_field_limit() -> None:
    """Allow very long title/tag/asr-like CSV fields in the full dataset."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


@dataclass(frozen=True)
class AggregatedEvent:
    user_id: int
    pid: int
    exposed_time: int
    watch_time: float
    duration: float
    click: bool
    cvm_like: bool
    comment: bool
    follow: bool
    collect: bool
    forward: bool
    hate: bool


def parse_args() -> argparse.Namespace:
    bundle_parser = argparse.ArgumentParser(add_help=False)
    bundle_parser.add_argument(
        "--bundle",
        choices=["sampled", "full"],
        default="sampled",
    )
    bundle_args, _ = bundle_parser.parse_known_args()

    default_root = os.environ.get(
        "SHORT_VIDEO_DATA_ROOT",
        "./raw/Short-Video-dataset-WWW2025",
    )
    is_full = bundle_args.bundle == "full"
    default_input = os.path.join(
        default_root,
        "interaction.csv" if is_full else "interaction_sampled.csv",
    )
    default_output = (
        "./datasets/ShortVideoFull" if is_full else "./datasets/ShortVideoSampled"
    )
    default_feature_dir = os.path.join(default_root, "visual_feature_fixed")
    default_items_mapping_json = os.environ.get(
        "SHORT_VIDEO_ITEMS_JSON",
        os.path.join(default_root, "fix_ShortVideo", "items_final_fixed.json"),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Convert sampled or full WWW2025 short-video interactions into "
            "SCOPE-Bench dataset files."
        )
    )
    parser.add_argument(
        "--bundle",
        choices=["sampled", "full"],
        default=bundle_args.bundle,
        help="Select default input/output paths (default: sampled).",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=default_input,
        help="Path to the sampled or full interaction CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output,
        help="Output directory for NexusRec-ready files",
    )
    parser.add_argument(
        "--positive_rule",
        type=str,
        default="click",
        choices=["click", "watch3", "watch50"],
        help="Positive interaction rule",
    )
    parser.add_argument(
        "--min_user_interactions",
        type=int,
        default=4,
        help="Minimum positive interactions per user after filtering",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="temporal_ratio",
        choices=["leave_one_out", "temporal_ratio"],
        help=(
            "How to split each user's chronological positive sequence. "
            "temporal_ratio uses --train_ratio/--valid_ratio and leaves the rest for test."
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Per-user train ratio used when --split_method temporal_ratio.",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Per-user valid ratio used when --split_method temporal_ratio.",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=default_feature_dir,
        help=(
            "Directory of per-video visual features with format {pid}.npy. "
            "If missing and --allow_zero_image_features is not set, script fails."
        ),
    )
    parser.add_argument(
        "--items_mapping_json",
        type=str,
        default=default_items_mapping_json,
        help=(
            "Optional items.json path for pid -> feature-id mapping. "
            "When provided, source_pid and video_id are both supported."
        ),
    )
    parser.add_argument(
        "--allow_zero_image_features",
        action="store_true",
        help=(
            "Allow generating all-zero image_features.npy when feature files "
            "cannot be found."
        ),
    )
    parser.add_argument(
        "--image_pooling",
        type=str,
        default="cover",
        choices=["cover", "mean", "flatten"],
        help=(
            "How to reduce multi-dimensional image feature arrays to one vector "
            "per item. 'cover' uses row 0, which is the cover image feature."
        ),
    )
    parser.add_argument(
        "--feature_workers",
        type=int,
        default=8,
        help="Number of worker threads for loading per-video image feature files.",
    )
    parser.add_argument(
        "--fallback_image_dim",
        type=int,
        default=512,
        help="Image dim used only when --allow_zero_image_features is set",
    )
    parser.add_argument(
        "--text_feature_mode",
        type=str,
        default="sentence_transformer",
        choices=["sentence_transformer", "title_hash", "zeros"],
        help=(
            "How to build text_features.npy. "
            "'sentence_transformer' follows MMRec-style sentence embeddings; "
            "'title_hash' hashes raw title text into dense vectors."
        ),
    )
    parser.add_argument(
        "--sentence_transformer_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name used for MMRec-style text features.",
    )
    parser.add_argument(
        "--sentence_transformer_batch_size",
        type=int,
        default=256,
        help="Batch size for SentenceTransformer.encode().",
    )
    parser.add_argument(
        "--sentence_transformer_device",
        type=str,
        default="",
        help="Optional device string for SentenceTransformer, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--text_metadata_json",
        type=str,
        default="",
        help=(
            "Optional item metadata JSON for sentence text. Empty means using "
            "--items_mapping_json when it contains item metadata."
        ),
    )
    parser.add_argument(
        "--title_source_csv",
        type=str,
        default="",
        help=(
            "CSV used to read raw title text. Empty means using --input_csv."
        ),
    )
    parser.add_argument(
        "--title_column",
        type=str,
        default="source_match_title_cn",
        help=(
            "Title field used for text features. It may be a column in "
            "--title_source_csv or a field in --text_metadata_json."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV file encoding",
    )
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _as_int(value: str) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _as_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def aggregate_events(
    input_csv: Path,
    encoding: str,
) -> Dict[Tuple[int, int, int], AggregatedEvent]:
    """Aggregate duplicate expanded rows by (user_id, pid, exposed_time)."""
    raise_csv_field_limit()
    raw: Dict[Tuple[int, int, int], Dict[str, object]] = {}

    with input_csv.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                _as_int(row["user_id"]),
                _as_int(row["pid"]),
                _as_int(row["exposed_time"]),
            )
            if key not in raw:
                raw[key] = {
                    "watch_time": _as_float(row.get("watch_time", "0")),
                    "duration": _as_float(row.get("duration", "0")),
                    "click": _as_bool(row.get("click", "False")),
                    "cvm_like": _as_bool(row.get("cvm_like", "False")),
                    "comment": _as_bool(row.get("comment", "False")),
                    "follow": _as_bool(row.get("follow", "False")),
                    "collect": _as_bool(row.get("collect", "False")),
                    "forward": _as_bool(row.get("forward", "False")),
                    "hate": _as_bool(row.get("hate", "False")),
                }
                continue

            # Merge expanded duplicates: OR booleans, keep numeric from first row.
            current = raw[key]
            current["click"] = bool(current["click"]) or _as_bool(row.get("click", "False"))
            current["cvm_like"] = bool(current["cvm_like"]) or _as_bool(row.get("cvm_like", "False"))
            current["comment"] = bool(current["comment"]) or _as_bool(row.get("comment", "False"))
            current["follow"] = bool(current["follow"]) or _as_bool(row.get("follow", "False"))
            current["collect"] = bool(current["collect"]) or _as_bool(row.get("collect", "False"))
            current["forward"] = bool(current["forward"]) or _as_bool(row.get("forward", "False"))
            current["hate"] = bool(current["hate"]) or _as_bool(row.get("hate", "False"))

    aggregated: Dict[Tuple[int, int, int], AggregatedEvent] = {}
    for (user_id, pid, exposed_time), item in raw.items():
        aggregated[(user_id, pid, exposed_time)] = AggregatedEvent(
            user_id=user_id,
            pid=pid,
            exposed_time=exposed_time,
            watch_time=float(item["watch_time"]),
            duration=float(item["duration"]),
            click=bool(item["click"]),
            cvm_like=bool(item["cvm_like"]),
            comment=bool(item["comment"]),
            follow=bool(item["follow"]),
            collect=bool(item["collect"]),
            forward=bool(item["forward"]),
            hate=bool(item["hate"]),
        )
    return aggregated


def is_positive(event: AggregatedEvent, rule: str) -> bool:
    strong = (
        event.cvm_like
        or event.comment
        or event.follow
        or event.collect
        or event.forward
    )
    if event.hate:
        return False

    if rule == "click":
        return event.click
    if rule == "watch3":
        return (event.watch_time >= 3.0) or strong
    if rule == "watch50":
        denom = event.duration if event.duration > 0 else 1e-9
        return (event.watch_time / denom >= 0.5) or strong
    raise ValueError(f"Unsupported positive rule: {rule}")


def _split_sequence_labels(
    n_events: int,
    split_method: str,
    train_ratio: float,
    valid_ratio: float,
) -> List[int]:
    if n_events < 3:
        raise ValueError("At least 3 interactions are required for train/valid/test splits.")

    if split_method == "leave_one_out":
        return [
            2 if idx == n_events - 1 else 1 if idx == n_events - 2 else 0
            for idx in range(n_events)
        ]

    if split_method == "temporal_ratio":
        if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
            raise ValueError(
                "For temporal_ratio, require train_ratio > 0, valid_ratio > 0, "
                "and train_ratio + valid_ratio < 1."
            )
        train_end = int(n_events * train_ratio)
        valid_end = int(n_events * (train_ratio + valid_ratio))
        train_end = min(max(train_end, 1), n_events - 2)
        valid_end = min(max(valid_end, train_end + 1), n_events - 1)
        labels = []
        for idx in range(n_events):
            if idx < train_end:
                labels.append(0)
            elif idx < valid_end:
                labels.append(1)
            else:
                labels.append(2)
        return labels

    raise ValueError(f"Unsupported split_method: {split_method}")


def build_split_rows(
    events: Iterable[AggregatedEvent],
    positive_rule: str,
    min_user_interactions: int,
    split_method: str = "leave_one_out",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[int, int], Dict[int, int], Dict[str, int]]:
    """Create NexusRec rows with per-user temporal split and contiguous id remapping."""
    by_user: Dict[int, List[AggregatedEvent]] = defaultdict(list)
    for event in events:
        if is_positive(event, positive_rule):
            by_user[event.user_id].append(event)

    # Keep users with enough positives for LOO.
    kept_users = [u for u, seq in by_user.items() if len(seq) >= min_user_interactions]
    kept_users.sort()
    user_map = {raw_uid: idx for idx, raw_uid in enumerate(kept_users)}

    # Item mapping is built from retained users only.
    used_raw_items = set()
    for raw_uid in kept_users:
        for e in by_user[raw_uid]:
            used_raw_items.add(e.pid)
    sorted_items = sorted(used_raw_items)
    item_map = {raw_iid: idx for idx, raw_iid in enumerate(sorted_items)}

    inter_rows: List[Tuple[int, int, int, int]] = []
    split_counter = Counter()
    for raw_uid in kept_users:
        # Stable ordering: time asc, then pid for tie break.
        seq = sorted(by_user[raw_uid], key=lambda e: (e.exposed_time, e.pid))
        split_labels = _split_sequence_labels(
            len(seq),
            split_method=split_method,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
        uid = user_map[raw_uid]
        for event, split in zip(seq, split_labels):
            iid = item_map[event.pid]
            inter_rows.append((uid, iid, split, event.exposed_time))
            split_counter[split] += 1

    stats = {
        "split_method": split_method,
        "train_ratio": train_ratio if split_method == "temporal_ratio" else None,
        "valid_ratio": valid_ratio if split_method == "temporal_ratio" else None,
        "users_before_min_filter": len(by_user),
        "users_after_min_filter": len(kept_users),
        "items_after_user_filter": len(item_map),
        "interactions_after_filter": len(inter_rows),
        "train_rows": split_counter[0],
        "valid_rows": split_counter[1],
        "test_rows": split_counter[2],
    }
    return inter_rows, user_map, item_map, stats


def build_loo_rows(
    events: Iterable[AggregatedEvent],
    positive_rule: str,
    min_user_interactions: int,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[int, int], Dict[int, int], Dict[str, int]]:
    """Backward-compatible wrapper for leave-one-out preprocessing."""
    return build_split_rows(
        events=events,
        positive_rule=positive_rule,
        min_user_interactions=min_user_interactions,
        split_method="leave_one_out",
    )


def write_inter_csv(path: Path, rows: List[Tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["userID", "itemID", "split_label", "timestamp"])
        writer.writerows(rows)


def load_pid_to_feature_id_map(items_mapping_json: Path) -> Dict[int, int]:
    """Load raw pid -> feature-file id mapping from items.json-like payload.

    Supported payloads:
      - List[dict] with keys including `video_id` and optionally `source_pid`
      - Dict[str/int, str/int] raw->feature mapping
    """
    with items_mapping_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping: Dict[int, int] = {}

    if isinstance(payload, list):
        for idx, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            video_id = row.get("video_id")
            if video_id is None:
                # Fallback: assume 1-based row index if video_id missing.
                video_id = idx + 1
            try:
                video_id = int(video_id)
            except Exception:
                continue

            # Always support video_id itself as key.
            mapping[video_id] = video_id

            source_pid = row.get("source_pid")
            if source_pid is not None:
                try:
                    mapping[int(source_pid)] = video_id
                except Exception:
                    pass
        return mapping

    if isinstance(payload, dict):
        for k, v in payload.items():
            try:
                mapping[int(k)] = int(v)
            except Exception:
                continue
        return mapping

    raise ValueError(
        f"Unsupported mapping payload type in {items_mapping_json}: {type(payload).__name__}"
    )


def _to_image_vector(arr: np.ndarray, image_pooling: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and image_pooling == "cover":
        if arr.shape[0] == 0:
            return arr.reshape(-1)
        return arr[0]
    if arr.ndim == 2 and image_pooling == "mean":
        return arr.mean(axis=0)
    if arr.ndim == 2 and image_pooling == "flatten":
        return arr.reshape(-1)
    # Fallback for >2 dims: flatten to keep full information.
    return arr.reshape(-1)


def _discover_feature_dim(
    feature_dir: Path,
    candidate_raw_items: Iterable[int],
    image_pooling: str,
    raw_pid_to_feature_id: Optional[Dict[int, int]] = None,
) -> int:
    for raw_pid in candidate_raw_items:
        feature_id = (
            raw_pid_to_feature_id.get(raw_pid, raw_pid)
            if raw_pid_to_feature_id is not None
            else raw_pid
        )
        file_path = feature_dir / f"{feature_id}.npy"
        if not file_path.exists():
            continue
        arr = np.load(file_path, allow_pickle=False)
        vec = _to_image_vector(arr, image_pooling=image_pooling)
        if vec.size == 0:
            continue
        return int(vec.shape[0])
    raise FileNotFoundError(
        f"No usable feature file found in {feature_dir} for sampled item ids."
    )


def build_image_feature_matrix(
    feature_dir: Path,
    item_map: Dict[int, int],
    allow_zero_image_features: bool,
    fallback_image_dim: int,
    image_pooling: str,
    raw_pid_to_feature_id: Optional[Dict[int, int]] = None,
    feature_workers: int = 1,
) -> Tuple[np.ndarray, List[int], int, int]:
    """Build [n_items, d] image feature matrix aligned to remapped item ids."""
    n_items = len(item_map)
    raw_items_sorted = sorted(item_map.keys())

    if feature_dir.exists() and feature_dir.is_dir():
        try:
            dim = _discover_feature_dim(
                feature_dir,
                raw_items_sorted,
                image_pooling=image_pooling,
                raw_pid_to_feature_id=raw_pid_to_feature_id,
            )
        except FileNotFoundError:
            if not allow_zero_image_features:
                raise
            dim = fallback_image_dim
            LOGGER.warning(
                "No feature files found in %s; using all-zero image features with dim=%d",
                feature_dir,
                dim,
            )
    else:
        if not allow_zero_image_features:
            raise FileNotFoundError(
                f"Feature directory does not exist: {feature_dir}. "
                "Pass a valid --feature_dir or set --allow_zero_image_features."
            )
        dim = fallback_image_dim
        LOGGER.warning(
            "Feature directory missing; using all-zero image features with dim=%d",
            dim,
        )

    matrix = np.zeros((n_items, dim), dtype=np.float32)
    missing_raw_pids: List[int] = []
    loaded = 0
    remapped_hit = 0

    def load_one(item: Tuple[int, int]) -> Tuple[int, int, int, Optional[np.ndarray]]:
        raw_pid, new_iid = item
        feature_id = (
            raw_pid_to_feature_id.get(raw_pid, raw_pid)
            if raw_pid_to_feature_id is not None
            else raw_pid
        )
        file_path = feature_dir / f"{feature_id}.npy"
        if not file_path.exists():
            return raw_pid, new_iid, feature_id, None

        arr = np.load(file_path, allow_pickle=False)
        vec = _to_image_vector(arr, image_pooling=image_pooling)
        if vec.shape[0] != dim:
            raise ValueError(
                f"Feature dim mismatch at {file_path}: got {vec.shape[0]}, expected {dim}"
            )
        return raw_pid, new_iid, feature_id, vec

    if feature_dir.exists() and feature_dir.is_dir():
        items = list(item_map.items())
        workers = max(1, int(feature_workers))
        LOGGER.info(
            "Loading image features for %d items with %d worker(s)",
            len(items),
            workers,
        )

        def consume(iterator: Iterable[Tuple[int, int, int, Optional[np.ndarray]]]) -> None:
            nonlocal loaded, remapped_hit
            for seen, (raw_pid, new_iid, feature_id, vec) in enumerate(iterator, start=1):
                if vec is None:
                    missing_raw_pids.append(raw_pid)
                else:
                    if feature_id != raw_pid:
                        remapped_hit += 1
                    matrix[new_iid] = vec
                    loaded += 1
                if seen % 10000 == 0:
                    LOGGER.info(
                        "Image feature progress: %d/%d loaded=%d missing=%d",
                        seen,
                        n_items,
                        loaded,
                        len(missing_raw_pids),
                    )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                consume(executor.map(load_one, items))
        else:
            consume(map(load_one, items))
    else:
        missing_raw_pids.extend(raw_items_sorted)

    LOGGER.info(
        "Image feature matrix built: shape=%s loaded=%d missing=%d remapped_hit=%d",
        matrix.shape,
        loaded,
        len(missing_raw_pids),
        remapped_hit,
    )
    return matrix, missing_raw_pids, dim, remapped_hit


def write_text_feature_zeros(path: Path, n_items: int, dim: int) -> np.ndarray:
    text = np.zeros((n_items, dim), dtype=np.float32)
    np.save(path, text)
    return text


def _flatten_text_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            parts.extend(_flatten_text_value(item))
        return parts
    if isinstance(value, dict):
        parts = []
        for item in value.values():
            parts.extend(_flatten_text_value(item))
        return parts

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    # MMRec Amazon metadata stores categories as a Python-list-like string.
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    ):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [text]
        return _flatten_text_value(parsed)

    return [text]


def _append_metadata_fields(parts: List[str], metadata: Dict[str, Any], fields: Tuple[str, ...]) -> None:
    seen = set(parts)
    for field in fields:
        for value in _flatten_text_value(metadata.get(field)):
            value = value.replace("\n", " ").strip()
            if value and value not in seen:
                parts.append(value)
                seen.add(value)


def build_mmrec_sentence(metadata: Dict[str, Any]) -> str:
    """Build MMRec-style text: title + brand + categories + description."""
    parts: List[str] = []
    _append_metadata_fields(parts, metadata, TEXT_TITLE_FIELDS)
    _append_metadata_fields(parts, metadata, TEXT_BRAND_FIELDS)
    _append_metadata_fields(parts, metadata, TEXT_CATEGORY_FIELDS)
    _append_metadata_fields(parts, metadata, TEXT_DESCRIPTION_FIELDS)
    return " ".join(parts).replace("\n", " ").strip()


def load_item_metadata_by_pid(items_metadata_json: Path) -> Dict[int, Dict[str, Any]]:
    """Load source_pid/video_id -> item metadata from an items.json-like list."""
    with items_metadata_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        return {}

    metadata_by_pid: Dict[int, Dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        for key in ("source_pid", "video_id"):
            value = row.get(key)
            if value is None:
                continue
            try:
                metadata_by_pid[int(value)] = row
            except Exception:
                continue
    return metadata_by_pid


def _fill_sentence_fallback_from_csv(
    sentences: List[str],
    item_map: Dict[int, int],
    source_csv: Optional[Path],
    encoding: str,
    title_column: str,
) -> int:
    if source_csv is None or not source_csv.exists():
        return 0

    raise_csv_field_limit()
    filled = 0
    with source_csv.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if title_column not in row:
                continue
            raw_pid = _as_int(row.get("pid", "0"))
            mapped = item_map.get(raw_pid)
            if mapped is None or sentences[mapped]:
                continue
            title = str(row.get(title_column, "")).replace("\n", " ").strip()
            if not title:
                continue
            sentences[mapped] = title
            filled += 1
    return filled


def build_sentence_transformer_text_features(
    item_map: Dict[int, int],
    items_metadata_json: Optional[Path],
    source_csv: Optional[Path],
    encoding: str,
    title_column: str,
    model_name: str,
    batch_size: int,
    device: str = "",
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode MMRec-style item sentences with SentenceTransformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for --text_feature_mode "
            "sentence_transformer. Install it with `pip install sentence-transformers`."
        ) from exc

    metadata_by_pid: Dict[int, Dict[str, Any]] = {}
    if items_metadata_json is not None and items_metadata_json.exists():
        metadata_by_pid = load_item_metadata_by_pid(items_metadata_json)

    sentences: List[str] = [""] * len(item_map)
    metadata_hits = 0
    for raw_pid, new_iid in item_map.items():
        metadata = metadata_by_pid.get(raw_pid)
        if metadata is None:
            continue
        sentence = build_mmrec_sentence(metadata)
        if sentence:
            sentences[new_iid] = sentence
            metadata_hits += 1

    csv_fallback_filled = _fill_sentence_fallback_from_csv(
        sentences=sentences,
        item_map=item_map,
        source_csv=source_csv,
        encoding=encoding,
        title_column=title_column,
    )
    empty_sentence_count = sum(1 for sentence in sentences if not sentence)
    if empty_sentence_count:
        LOGGER.warning(
            "SentenceTransformer text has %d empty item sentences; encoding them as blank strings.",
            empty_sentence_count,
        )

    LOGGER.info(
        "Encoding %d item sentences with SentenceTransformer(%s), metadata_hits=%d csv_fallback=%d",
        len(sentences),
        model_name,
        metadata_hits,
        csv_fallback_filled,
    )
    model_kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **model_kwargs)
    embeddings = model.encode(
        sentences,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    stats = {
        "metadata_hit_count": metadata_hits,
        "csv_fallback_filled_count": csv_fallback_filled,
        "empty_sentence_count": empty_sentence_count,
        "sentence_transformer_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    return embeddings, stats


def build_title_hash_text_features(
    source_csv: Path,
    item_map: Dict[int, int],
    dim: int,
    encoding: str,
    title_column: str,
    items_metadata_json: Optional[Path] = None,
) -> Tuple[np.ndarray, int]:
    """Build text features from raw title strings using hashing vectorizer."""
    raise_csv_field_limit()
    titles: List[str] = [""] * len(item_map)
    hit = 0

    metadata_by_pid: Dict[int, Dict[str, Any]] = {}
    if items_metadata_json is not None and items_metadata_json.exists():
        metadata_by_pid = load_item_metadata_by_pid(items_metadata_json)
        for raw_pid, new_iid in item_map.items():
            metadata = metadata_by_pid.get(raw_pid)
            if metadata is None:
                continue
            title = str(metadata.get(title_column, "")).replace("\n", " ").strip()
            if not title:
                continue
            titles[new_iid] = title
            hit += 1

    missing_before_csv = len(item_map) - hit
    if missing_before_csv and source_csv is not None and source_csv.exists():
        with source_csv.open("r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            if title_column not in (reader.fieldnames or []):
                LOGGER.warning(
                    "Title column '%s' not found in %s; using metadata titles only.",
                    title_column,
                    source_csv,
                )
            else:
                for row in reader:
                    raw_pid = _as_int(row.get("pid", "0"))
                    mapped = item_map.get(raw_pid)
                    if mapped is None:
                        continue
                    # Keep first non-empty title for each item.
                    if titles[mapped]:
                        continue
                    title = str(row.get(title_column, "")).strip()
                    if not title:
                        continue
                    titles[mapped] = title
                    hit += 1

    missing_title_count = len(item_map) - hit
    vectorizer = HashingVectorizer(
        n_features=dim,
        alternate_sign=False,
        norm="l2",
        analyzer="char",
        ngram_range=(2, 4),
    )
    sparse = vectorizer.transform(titles)
    dense = sparse.astype(np.float32).toarray()
    return dense, missing_title_count


def run_contract_checks(inter_rows: List[Tuple[int, ...]], image: np.ndarray, text: np.ndarray) -> Dict[str, object]:
    if not inter_rows:
        raise ValueError("inter_rows is empty after preprocessing.")

    splits = {row[2] for row in inter_rows}
    if not splits.issubset({0, 1, 2}):
        raise ValueError(f"Invalid split labels found: {sorted(splits)}")

    user_splits: Dict[int, set] = defaultdict(set)
    users = set()
    items = set()
    has_timestamp = False
    timestamp_min = None
    timestamp_max = None
    for row in inter_rows:
        uid, iid, split = row[:3]
        if uid < 0 or iid < 0:
            raise ValueError("Negative userID/itemID found.")
        users.add(uid)
        items.add(iid)
        user_splits[uid].add(split)
        if len(row) >= 4:
            has_timestamp = True
            timestamp = int(row[3])
            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)

    for uid, owned in user_splits.items():
        if not {0, 1, 2}.issubset(owned):
            raise ValueError(
                f"User {uid} does not have all train/valid/test splits: {sorted(owned)}"
            )

    max_uid = max(users)
    max_iid = max(items)
    if max_uid + 1 != len(users):
        raise ValueError("userID is not contiguous from 0..n-1.")
    if max_iid + 1 != len(items):
        raise ValueError("itemID is not contiguous from 0..n-1.")

    expected_rows = max_iid + 1
    if image.shape[0] != expected_rows:
        raise ValueError(
            f"image_features rows mismatch: got {image.shape[0]} expected {expected_rows}"
        )
    if text.shape[0] != expected_rows:
        raise ValueError(
            f"text_features rows mismatch: got {text.shape[0]} expected {expected_rows}"
        )
    if not np.isfinite(image).all():
        raise ValueError("image_features.npy contains NaN/Inf.")
    if not np.isfinite(text).all():
        raise ValueError("text_features.npy contains NaN/Inf.")

    train_rows = sum(1 for row in inter_rows if row[2] == 0)
    valid_rows = sum(1 for row in inter_rows if row[2] == 1)
    test_rows = sum(1 for row in inter_rows if row[2] == 2)

    return {
        "num_users": len(users),
        "num_items": len(items),
        "num_interactions": len(inter_rows),
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "test_rows": test_rows,
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "text_shape": [int(text.shape[0]), int(text.shape[1])],
        "has_timestamp": has_timestamp,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
    }


def prepare_dataset(args: argparse.Namespace) -> Path:
    """Prepare a WWW2025 short-video split from parsed CLI arguments."""
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    items_mapping_json = (
        Path(args.items_mapping_json).expanduser().resolve()
        if args.items_mapping_json
        else None
    )
    title_source_csv = (
        Path(args.title_source_csv).expanduser().resolve()
        if args.title_source_csv
        else input_csv
    )
    text_metadata_json = (
        Path(args.text_metadata_json).expanduser().resolve()
        if getattr(args, "text_metadata_json", "")
        else items_mapping_json
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if args.text_feature_mode == "title_hash" and not title_source_csv.exists():
        raise FileNotFoundError(f"Title source CSV not found: {title_source_csv}")
    if items_mapping_json is not None and not items_mapping_json.exists():
        raise FileNotFoundError(f"items_mapping_json not found: {items_mapping_json}")
    if getattr(args, "text_metadata_json", "") and text_metadata_json is not None and not text_metadata_json.exists():
        raise FileNotFoundError(f"text_metadata_json not found: {text_metadata_json}")

    LOGGER.info("Reading and aggregating: %s", input_csv)
    aggregated = aggregate_events(input_csv, encoding=args.encoding)
    LOGGER.info("Aggregated unique exposure events: %d", len(aggregated))

    inter_rows, user_map, item_map, stats = build_split_rows(
        events=aggregated.values(),
        positive_rule=args.positive_rule,
        min_user_interactions=args.min_user_interactions,
        split_method=getattr(args, "split_method", "leave_one_out"),
        train_ratio=getattr(args, "train_ratio", 0.8),
        valid_ratio=getattr(args, "valid_ratio", 0.1),
    )
    LOGGER.info("Post-filter stats: %s", stats)

    output_dir.mkdir(parents=True, exist_ok=True)
    inter_csv_path = output_dir / "inter.csv"
    write_inter_csv(inter_csv_path, inter_rows)
    LOGGER.info("Wrote inter.csv: %s", inter_csv_path)

    raw_pid_to_feature_id = None
    if items_mapping_json is not None:
        raw_pid_to_feature_id = load_pid_to_feature_id_map(items_mapping_json)
        LOGGER.info(
            "Loaded pid->feature-id mapping: %d entries from %s",
            len(raw_pid_to_feature_id),
            items_mapping_json,
        )
        output_items_json = output_dir / "items.json"
        if items_mapping_json != output_items_json:
            shutil.copy2(items_mapping_json, output_items_json)
            LOGGER.info("Copied item metadata/mapping: %s", output_items_json)
        output_named_items_json = output_dir / items_mapping_json.name
        if output_named_items_json != output_items_json and items_mapping_json != output_named_items_json:
            shutil.copy2(items_mapping_json, output_named_items_json)
            LOGGER.info("Copied fixed item metadata/mapping: %s", output_named_items_json)

    image, missing_raw_pids, image_dim, remapped_hit = build_image_feature_matrix(
        feature_dir=feature_dir,
        item_map=item_map,
        allow_zero_image_features=args.allow_zero_image_features,
        fallback_image_dim=args.fallback_image_dim,
        image_pooling=args.image_pooling,
        raw_pid_to_feature_id=raw_pid_to_feature_id,
        feature_workers=getattr(args, "feature_workers", 1),
    )
    image_path = output_dir / "image_features.npy"
    np.save(image_path, image)
    LOGGER.info("Wrote image features: %s", image_path)

    text_path = output_dir / "text_features.npy"
    missing_title_count = None
    text_feature_stats: Dict[str, int] = {}
    if args.text_feature_mode == "zeros":
        text = write_text_feature_zeros(text_path, n_items=image.shape[0], dim=image_dim)
        LOGGER.info("Wrote zero text features: %s", text_path)
    elif args.text_feature_mode == "sentence_transformer":
        text, text_feature_stats = build_sentence_transformer_text_features(
            item_map=item_map,
            items_metadata_json=text_metadata_json,
            source_csv=title_source_csv,
            encoding=args.encoding,
            title_column=args.title_column,
            model_name=getattr(args, "sentence_transformer_model", "all-MiniLM-L6-v2"),
            batch_size=getattr(args, "sentence_transformer_batch_size", 256),
            device=getattr(args, "sentence_transformer_device", ""),
        )
        missing_title_count = text_feature_stats.get("empty_sentence_count", 0)
        np.save(text_path, text)
        LOGGER.info(
            "Wrote SentenceTransformer text features: %s shape=%s",
            text_path,
            text.shape,
        )
    elif args.text_feature_mode == "title_hash":
        text, missing_title_count = build_title_hash_text_features(
            source_csv=title_source_csv,
            item_map=item_map,
            dim=image_dim,
            encoding=args.encoding,
            title_column=args.title_column,
            items_metadata_json=text_metadata_json,
        )
        np.save(text_path, text)
        LOGGER.info(
            "Wrote title-hash text features: %s (missing titles=%d)",
            text_path,
            missing_title_count,
        )
    else:
        raise ValueError(f"Unsupported text_feature_mode: {args.text_feature_mode}")

    check = run_contract_checks(inter_rows, image, text)
    LOGGER.info("Contract checks passed: %s", check)

    metadata = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "items_mapping_json": str(items_mapping_json) if items_mapping_json else "",
        "source_items_file": items_mapping_json.name if items_mapping_json else "",
        "mapping_entry_count": len(raw_pid_to_feature_id) if raw_pid_to_feature_id is not None else 0,
        "mapping_remapped_hit_count": remapped_hit,
        "image_pooling": args.image_pooling,
        "feature_workers": getattr(args, "feature_workers", 1),
        "timestamp_field": "timestamp",
        "timestamp_source_field": "exposed_time",
        "text_feature_mode": args.text_feature_mode,
        "text_metadata_json": str(text_metadata_json) if text_metadata_json else "",
        "sentence_transformer_model": getattr(args, "sentence_transformer_model", ""),
        "sentence_transformer_batch_size": getattr(args, "sentence_transformer_batch_size", ""),
        "sentence_transformer_device": getattr(args, "sentence_transformer_device", ""),
        "text_feature_stats": text_feature_stats,
        "title_source_csv": str(title_source_csv),
        "title_column": args.title_column,
        "positive_rule": args.positive_rule,
        "min_user_interactions": args.min_user_interactions,
        "split_method": getattr(args, "split_method", "leave_one_out"),
        "train_ratio": getattr(args, "train_ratio", 0.8),
        "valid_ratio": getattr(args, "valid_ratio", 0.1),
        "allow_zero_image_features": args.allow_zero_image_features,
        "fallback_image_dim": args.fallback_image_dim,
        "stats": stats,
        "contract_check": check,
        "user_count": len(user_map),
        "item_count": len(item_map),
        "missing_image_feature_count": len(missing_raw_pids),
        "missing_image_feature_raw_pids_file": "missing_image_feature_raw_pids.txt",
        "missing_title_count": missing_title_count,
        "id_mapping_file": "id_mappings.json",
        "bundle_ready": True,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output_dir / "missing_image_feature_raw_pids.txt").write_text(
        "\n".join(str(pid) for pid in sorted(missing_raw_pids)),
        encoding="utf-8",
    )

    id_mappings = {
        "user_raw_to_new": {str(raw): new for raw, new in user_map.items()},
        "item_raw_to_new": {str(raw): new for raw, new in item_map.items()},
    }
    (output_dir / "id_mappings.json").write_text(
        json.dumps(id_mappings, ensure_ascii=False),
        encoding="utf-8",
    )

    LOGGER.info("Done. Dataset is ready at: %s", output_dir)
    return output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
