#!/usr/bin/env python
# coding: utf-8
"""Compute Recall/NDCG/Precision and LCDS from saved recommendation lists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluation.lcds import (
    build_cds_gain_table,
    build_lcds_result_dict,
    positive_items_for_users,
    recommendation_records_to_matrix,
    recommendation_users,
)
from core.evaluation.topk_kernel import (
    build_bool_rec_matrix,
    build_topk_result_dict,
    compute_metric_arrays,
)
from core.utils.recommendation import Recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read recommendation-list artifacts and compute behavioral ranking "
            "metrics plus A-LCDS/E-LCDS."
        )
    )
    parser.add_argument(
        "--recommendations",
        nargs="+",
        required=True,
        help="Recommendation artifact path(s): json, jsonl, csv, or tsv with metadata.",
    )
    parser.add_argument("--dataset-dir", default="datasets/ShortVideoFull")
    parser.add_argument(
        "--cds-jsonl",
        "--cpd-jsonl",
        dest="cds_jsonl",
        default="scoring/results/Qwen3_7_Max_full_t0p3_seed42_scores.jsonl",
        help=(
            "CDS annotation JSONL. --cpd-jsonl is accepted as a deprecated alias."
        ),
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["Recall", "NDCG", "Precision"],
        help="Behavioral ranking metrics to recompute from the recommendation list.",
    )
    parser.add_argument(
        "--output",
        default="outputs/lcds/ShortVideoFull/lcds_summary.csv",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --output if it already exists.",
    )
    return parser.parse_args()


def validate_cutoffs(topk: List[int], width: int) -> None:
    for k in topk:
        if k < 1 or k > width:
            raise ValueError(
                f"Requested cutoff {k}, but recommendation width is {width}."
            )


def behavioral_metrics(
    topk_matrix: np.ndarray,
    positive_items: List[np.ndarray],
    metrics: List[str],
    topk: List[int],
) -> Dict[str, float]:
    bool_matrix = build_bool_rec_matrix(topk_matrix, positive_items)
    pos_len = np.asarray([len(items) for items in positive_items], dtype=np.int64)
    arrays = compute_metric_arrays(
        metrics,
        bool_matrix,
        pos_len,
        topk_matrix,
        n_items=None,
        item_pop_freq=None,
    )
    return build_topk_result_dict(arrays, topk)


def compute_one(
    path: str,
    gain_table,
    args: argparse.Namespace,
) -> Dict[str, object]:
    records, metadata = Recommendation.load(path)
    topk_matrix = recommendation_records_to_matrix(records)
    validate_cutoffs(args.topk, topk_matrix.shape[1])
    users = recommendation_users(records)
    positives = positive_items_for_users(
        Path(args.dataset_dir) / "inter.csv",
        users,
        user_field=metadata["user_id_field"],
        item_field=metadata["item_id_field"],
    )

    row: Dict[str, object] = {
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "type": metadata["type"],
        "comment": metadata["comment"],
        "recommendations": str(path),
        "export_topk": int(metadata["topk"]),
        "exported_user_count": int(metadata["exported_user_count"]),
        "recommendation_count": int(metadata["recommendation_count"]),
        "cds_item_count": int(gain_table.stats["item_count"]),
        "cds_numeric_score_count": int(gain_table.stats["numeric_score_count"]),
        "cds_null_score_count": int(gain_table.stats["null_score_count"]),
        "cds_missing_score_count": int(gain_table.stats["missing_score_count"]),
        "cds_zero_gain_count": int(gain_table.stats["zero_gain_count"]),
    }
    row.update(behavioral_metrics(topk_matrix, positives, args.metrics, args.topk))
    row.update(build_lcds_result_dict(topk_matrix, gain_table.gains, args.topk))
    return row


def write_rows(rows: List[Dict[str, object]], output: str, append: bool) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if append and output_path.exists():
        existing = pd.read_csv(output_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    cds_path = Path(args.cds_jsonl)
    if not cds_path.is_file():
        raise FileNotFoundError(
            f"CDS JSONL not found: {cds_path}. Recommendation files can still be "
            "reused; stage the CDS file at this path or pass --cds-jsonl, then "
            "rerun this offline script without rerunning model test."
        )
    gain_table = build_cds_gain_table(args.dataset_dir, args.cds_jsonl)
    recommendation_paths = [
        path for path in args.recommendations if not str(path).endswith(".metadata.json")
    ]
    if not recommendation_paths:
        raise ValueError("No recommendation data files found after skipping metadata JSON files.")
    rows = [compute_one(path, gain_table, args) for path in recommendation_paths]
    write_rows(rows, args.output, args.append)
    print(f"Wrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
