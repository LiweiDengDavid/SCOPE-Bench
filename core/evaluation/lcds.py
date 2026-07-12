# coding: utf-8
"""List-wise Cognitive Depth Score metrics for recommendation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CdsGainTable:
    """CDS gains indexed by NexusRec internal zero-based item id."""

    gains: np.ndarray
    stats: Dict[str, Any]


def lcds_metric_arrays(
    topk_index: np.ndarray,
    gains: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute A-LCDS and E-LCDS prefix scores for a top-k matrix.

    Args:
        topk_index: Integer matrix shaped ``[n_users, max_k]``.
        gains: Per-item gain vector, indexed by internal item id.

    Returns:
        Mapping from metric name to an array of length ``max_k``. Element
        ``k-1`` is the mean user-level score at cutoff ``k``.
    """
    topk_index = np.asarray(topk_index)
    gains = np.asarray(gains, dtype=np.float64)
    if topk_index.ndim != 2 or topk_index.shape[1] < 1:
        raise ValueError(
            f"topk_index must be a non-empty 2-D matrix, got shape {topk_index.shape}."
        )
    if not np.issubdtype(topk_index.dtype, np.integer):
        raise ValueError("topk_index must contain integer item ids.")
    min_item = int(topk_index.min())
    max_item = int(topk_index.max())
    if min_item < 0 or max_item >= gains.shape[0]:
        raise ValueError(
            "topk_index contains item ids outside the CDS gain table: "
            f"range=[{min_item}, {max_item}], gains={gains.shape[0]}."
        )

    item_gains = gains[topk_index]
    max_k = topk_index.shape[1]
    ks = np.arange(1, max_k + 1, dtype=np.float64)
    a_lcds = (np.cumsum(item_gains, axis=1) / ks).mean(axis=0)

    discounts = 1.0 / np.log2(np.arange(1, max_k + 1, dtype=np.float64) + 1.0)
    weighted = np.cumsum(item_gains * discounts, axis=1)
    denom = np.cumsum(discounts)
    e_lcds = (weighted / denom).mean(axis=0)
    return {"A-LCDS": a_lcds, "E-LCDS": e_lcds}


def build_lcds_result_dict(
    topk_index: np.ndarray,
    gains: np.ndarray,
    topk: Sequence[int],
    *,
    item_id_offset: int = 0,
) -> Dict[str, float]:
    """Return rounded LCDS metrics at the requested cutoffs."""
    adjusted_topk = np.asarray(topk_index) - int(item_id_offset)
    arrays = lcds_metric_arrays(adjusted_topk, gains)
    result: Dict[str, float] = {}
    for metric, values in arrays.items():
        for k in topk:
            if k < 1 or k > len(values):
                raise ValueError(
                    f"Requested {metric}@{k}, but recommendation width is {len(values)}."
                )
            result[f"{metric}@{k}"] = round(float(values[k - 1]), 6)
    return result


def build_internal_item_video_map(dataset_dir: str | Path) -> np.ndarray:
    """Map internal item ids to CDS ``video_id`` values.

    ShortVideoFull stores model-facing item ids after filtering/reindexing. CDS
    rows use the original ``video_id`` from ``items.json``; ``id_mappings.json``
    bridges the internal id to the source pid.
    """
    dataset_path = Path(dataset_dir)
    mapping_path = dataset_path / "id_mappings.json"
    items_path = dataset_path / "items.json"

    mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_to_internal = mappings["item_raw_to_new"]
    internal_to_source = {
        int(internal_id): int(source_pid)
        for source_pid, internal_id in raw_to_internal.items()
    }
    if not internal_to_source:
        raise ValueError(f"No item mappings found in {mapping_path}.")

    items = json.loads(items_path.read_text(encoding="utf-8"))
    source_to_video: Dict[int, int] = {}
    for item in items:
        if "video_id" not in item:
            continue
        video_id = int(item["video_id"])
        for key in ("source_pid", "raw_video_id", "video_id"):
            if key in item and item[key] is not None:
                source_to_video[int(item[key])] = video_id

    item_count = max(internal_to_source) + 1
    internal_to_video = np.full(item_count, -1, dtype=np.int64)
    missing_sources: List[int] = []
    for internal_id, source_pid in internal_to_source.items():
        if source_pid not in source_to_video:
            missing_sources.append(source_pid)
            continue
        internal_to_video[internal_id] = source_to_video[source_pid]
    if missing_sources:
        sample = ", ".join(map(str, missing_sources[:5]))
        raise ValueError(
            f"{len(missing_sources)} mapped source_pid values are missing from "
            f"{items_path}; sample: {sample}"
        )
    return internal_to_video


def load_cds_scores(cds_jsonl: str | Path) -> Dict[int, int | None]:
    """Load CDS scores keyed by ``video_id``.

    ``None`` is preserved for insufficient-information labels so callers can
    count the conservative zero-gain fallback separately.
    """
    scores: Dict[int, int | None] = {}
    with Path(cds_jsonl).open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "video_id" not in row:
                raise ValueError(f"CDS row {line_number} is missing video_id.")
            video_id = int(row["video_id"])
            score = row.get("score")
            if score is None:
                scores[video_id] = None
            elif isinstance(score, int) and 0 <= score <= 6:
                scores[video_id] = int(score)
            else:
                raise ValueError(
                    f"CDS row {line_number} has invalid score={score!r}; "
                    "expected integer 0..6 or null."
                )
    if not scores:
        raise ValueError(f"No CDS scores loaded from {cds_jsonl}.")
    return scores


def build_cds_gain_table(
    dataset_dir: str | Path,
    cds_jsonl: str | Path,
    gain_divisor: float = 6.0,
) -> CdsGainTable:
    """Build per-item CDS gains for LCDS.

    Numeric CDS labels map to ``score / gain_divisor``. ``null`` labels and
    missing CDS rows use gain 0, matching the conservative policy in the paper
    definition.
    """
    if gain_divisor <= 0:
        raise ValueError(f"gain_divisor must be positive, got {gain_divisor!r}.")
    internal_to_video = build_internal_item_video_map(dataset_dir)
    scores = load_cds_scores(cds_jsonl)
    gains = np.zeros(internal_to_video.shape[0], dtype=np.float64)

    score_hist: Dict[int, int] = {score: 0 for score in range(7)}
    null_count = 0
    missing_count = 0
    for internal_id, video_id in enumerate(internal_to_video.tolist()):
        score = scores.get(video_id, "missing")
        if score == "missing":
            missing_count += 1
            continue
        if score is None:
            null_count += 1
            continue
        score_hist[int(score)] += 1
        gains[internal_id] = float(score) / gain_divisor

    stats = {
        "item_count": int(internal_to_video.shape[0]),
        "numeric_score_count": int(sum(score_hist.values())),
        "null_score_count": int(null_count),
        "missing_score_count": int(missing_count),
        "zero_gain_count": int(score_hist[0] + null_count + missing_count),
        "score_histogram": score_hist,
        "gain_divisor": float(gain_divisor),
    }
    return CdsGainTable(gains=gains, stats=stats)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


@lru_cache(maxsize=8)
def _build_cached_cds_gain_table(
    dataset_dir: str,
    cds_jsonl: str,
    gain_divisor: float,
    mapping_mtime_ns: int,
    items_mtime_ns: int,
    scores_mtime_ns: int,
) -> CdsGainTable:
    """Cache immutable gain tables across evaluators/HPO trials in one process."""
    del mapping_mtime_ns, items_mtime_ns, scores_mtime_ns
    return build_cds_gain_table(dataset_dir, cds_jsonl, gain_divisor)


def configured_cds_gain_table(config: Any) -> CdsGainTable | None:
    """Load the configured CDS gain table, or return ``None`` when disabled.

    Relative paths are resolved from the repository root. File mtimes are part
    of the cache key, so replacing a dataset mapping or score file invalidates
    the in-process cache automatically.
    """
    if "lcds" in config:
        settings = config["lcds"]
        settings_key = "lcds"
    elif "lcpd" in config:
        # Backward-compatible read path for experiment configs created before
        # the paper standardized the metric name as LCDS.
        settings = config["lcpd"]
        settings_key = "lcpd"
    else:
        return None
    if not isinstance(settings, dict):
        raise TypeError(f"Config key {settings_key!r} must be a mapping.")
    if not bool(settings.get("enabled", False)):
        return None

    dataset_dir_value = settings.get("dataset_dir")
    if not dataset_dir_value:
        dataset_dir_value = Path(config["data_path"]) / str(config["dataset"])
    cds_jsonl_value = settings.get("cds_jsonl", settings.get("cpd_jsonl"))
    if not cds_jsonl_value:
        raise ValueError("lcds.enabled=true requires lcds.cds_jsonl.")

    dataset_dir = _resolve_repo_path(dataset_dir_value)
    cds_jsonl = _resolve_repo_path(cds_jsonl_value)
    mapping_path = dataset_dir / "id_mappings.json"
    items_path = dataset_dir / "items.json"
    required = (mapping_path, items_path, cds_jsonl)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "LCDS is enabled but required artifact(s) are missing: "
            + ", ".join(missing)
        )

    gain_divisor = float(settings.get("gain_divisor", 6.0))
    return _build_cached_cds_gain_table(
        str(dataset_dir),
        str(cds_jsonl),
        gain_divisor,
        mapping_path.stat().st_mtime_ns,
        items_path.stat().st_mtime_ns,
        cds_jsonl.stat().st_mtime_ns,
    )


def recommendation_records_to_matrix(records: Iterable[Dict[str, Any]]) -> np.ndarray:
    """Convert grouped recommendation records into a user x rank item matrix."""
    rows: List[List[int]] = []
    width = None
    for record in records:
        items = sorted(record["items"], key=lambda item: item["rank"])
        row = [int(item["item_id"]) for item in items]
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("Recommendation records have inconsistent top-k widths.")
        rows.append(row)
    if not rows:
        raise ValueError("Recommendation records are empty.")
    return np.asarray(rows, dtype=np.int64)


def positive_items_for_users(
    interaction_csv: str | Path,
    users: Sequence[int],
    *,
    user_field: str = "userID",
    item_field: str = "itemID",
    split_field: str = "split_label",
    test_split_value: int = 2,
) -> List[np.ndarray]:
    """Return test positives ordered to match exported recommendation users."""
    import pandas as pd

    frame = pd.read_csv(
        interaction_csv,
        usecols=[user_field, item_field, split_field],
    )
    test_frame = frame[frame[split_field] == test_split_value]
    grouped = test_frame.groupby(user_field)[item_field]
    positive_items: List[np.ndarray] = []
    for user_id in users:
        if user_id in grouped.groups:
            positive_items.append(grouped.get_group(user_id).to_numpy(dtype=np.int64))
        else:
            positive_items.append(np.asarray([], dtype=np.int64))
    return positive_items


def recommendation_users(records: Iterable[Dict[str, Any]]) -> List[int]:
    """Return exported users in artifact order."""
    return [int(record["user_id"]) for record in records]
