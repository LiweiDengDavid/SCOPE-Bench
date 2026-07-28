#!/usr/bin/env python3
# coding: utf-8
"""Rebuild both ShortVideo bundles from the fixed WWW2025 item/features files."""

from __future__ import annotations

import argparse
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

try:
    from prepare_short_video import prepare_dataset
except ModuleNotFoundError:
    from scripts.prepare_short_video import prepare_dataset


LOGGER = logging.getLogger("rebuild_short_video_fixed")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = REPO_ROOT / "raw" / "Short-Video-dataset-WWW2025"


def parse_args() -> argparse.Namespace:
    source_root_default = Path(
        os.environ.get("SHORT_VIDEO_DATA_ROOT", str(DEFAULT_SOURCE_ROOT))
    )

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild ShortVideoSampled and ShortVideoFull with "
            "items_final_fixed.json, visual_feature_fixed, and "
            "source_match_title_cn text."
        )
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=source_root_default,
        help="Root directory of Short-Video-dataset-WWW2025.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["sampled", "full"],
        default=["sampled", "full"],
        help="Dataset bundles to rebuild.",
    )
    parser.add_argument(
        "--feature_workers",
        type=int,
        default=8,
        help="Number of worker threads for loading visual features.",
    )
    parser.add_argument(
        "--sentence_transformer_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for ShortVideoFull text features.",
    )
    parser.add_argument(
        "--sentence_transformer_batch_size",
        type=int,
        default=256,
        help="SentenceTransformer batch size for ShortVideoFull.",
    )
    parser.add_argument(
        "--sentence_transformer_device",
        type=str,
        default="",
        help="Optional SentenceTransformer device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--title_column",
        type=str,
        default="source_match_title_cn",
        help="Fixed item-metadata field to use as the primary title.",
    )
    return parser.parse_args()


def fixed_paths(source_root: Path) -> dict[str, Path]:
    source_root = source_root.expanduser().resolve()
    return {
        "source_root": source_root,
        "items": source_root / "fix_ShortVideo" / "items_final_fixed.json",
        "features": source_root / "visual_feature_fixed",
        "sampled_inter": source_root / "interaction_sampled.csv",
        "full_inter": source_root / "interaction.csv",
    }


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required fixed source paths: " + ", ".join(missing))


def dataset_args(args: argparse.Namespace, paths: dict[str, Path], name: str) -> Namespace:
    common = {
        "positive_rule": "click",
        "feature_dir": str(paths["features"]),
        "items_mapping_json": str(paths["items"]),
        "allow_zero_image_features": False,
        "image_pooling": "cover",
        "feature_workers": args.feature_workers,
        "fallback_image_dim": 512,
        "text_metadata_json": str(paths["items"]),
        "title_source_csv": "",
        "title_column": args.title_column,
        "encoding": "utf-8-sig",
        "sentence_transformer_model": args.sentence_transformer_model,
        "sentence_transformer_batch_size": args.sentence_transformer_batch_size,
        "sentence_transformer_device": args.sentence_transformer_device,
    }

    if name == "sampled":
        return Namespace(
            input_csv=str(paths["sampled_inter"]),
            output_dir=str(REPO_ROOT / "datasets" / "ShortVideoSampled"),
            min_user_interactions=4,
            split_method="temporal_ratio",
            train_ratio=0.8,
            valid_ratio=0.1,
            text_feature_mode="sentence_transformer",
            **common,
        )

    if name == "full":
        return Namespace(
            input_csv=str(paths["full_inter"]),
            output_dir=str(REPO_ROOT / "datasets" / "ShortVideoFull"),
            min_user_interactions=4,
            split_method="temporal_ratio",
            train_ratio=0.8,
            valid_ratio=0.1,
            text_feature_mode="sentence_transformer",
            **common,
        )

    raise ValueError(f"Unsupported dataset name: {name}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    paths = fixed_paths(args.source_root)
    require_paths(
        [
            paths["items"],
            paths["features"],
            paths["sampled_inter"],
            paths["full_inter"],
        ]
    )

    for name in args.datasets:
        LOGGER.info("Rebuilding ShortVideo%s", "Sampled" if name == "sampled" else "Full")
        prepare_dataset(dataset_args(args, paths, name))


if __name__ == "__main__":
    main()
