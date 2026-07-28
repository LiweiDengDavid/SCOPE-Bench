#!/usr/bin/env python3
# coding: utf-8
"""Rebuild ShortVideo text_features.npy with MMRec-style SentenceTransformer embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from prepare_short_video import (
    build_sentence_transformer_text_features,
    run_contract_checks,
)


LOGGER = logging.getLogger("build_short_video_text_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MMRec-style text features for an already prepared ShortVideo "
            "dataset directory."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./datasets/ShortVideoFull",
        help="Prepared NexusRec dataset directory.",
    )
    parser.add_argument(
        "--items_json",
        type=str,
        default="",
        help="items.json metadata path. Empty means {dataset_dir}/items.json.",
    )
    parser.add_argument(
        "--id_mappings_json",
        type=str,
        default="",
        help="id_mappings.json path. Empty means {dataset_dir}/id_mappings.json.",
    )
    parser.add_argument(
        "--raw_interaction_csv",
        type=str,
        default="",
        help="Optional raw interaction CSV used as title fallback.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Output .npy path. Empty means {dataset_dir}/text_features.npy.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for SentenceTransformer.encode().",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Optional SentenceTransformer device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--title_column",
        type=str,
        default="source_match_title_cn",
        help=(
            "Title field used for text features. It may be a column in "
            "--raw_interaction_csv or a field in --items_json."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding for --raw_interaction_csv.",
    )
    parser.add_argument(
        "--skip_contract_check",
        action="store_true",
        help="Skip row/finite checks against inter.csv and image_features.npy.",
    )
    return parser.parse_args()


def load_item_map(id_mappings_json: Path) -> Dict[int, int]:
    with id_mappings_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_to_new = payload["item_raw_to_new"]
    return {int(raw): int(new) for raw, new in raw_to_new.items()}


def load_inter_rows(inter_csv: Path) -> List[Tuple[int, int, int]]:
    import csv

    rows = []
    with inter_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["userID"]), int(row["itemID"]), int(row["split_label"])))
    return rows


def update_metadata(
    metadata_path: Path,
    args: argparse.Namespace,
    items_json: Path,
    output_file: Path,
    stats: dict,
    check: dict,
) -> None:
    if not metadata_path.exists():
        return
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata.update(
        {
            "text_feature_mode": "sentence_transformer",
            "text_feature_file": output_file.name,
            "text_metadata_json": str(items_json),
            "sentence_transformer_model": args.model_name,
            "sentence_transformer_batch_size": args.batch_size,
            "sentence_transformer_device": args.device,
            "text_feature_stats": stats,
            "missing_title_count": stats.get("empty_sentence_count", 0),
        }
    )
    if check:
        metadata["contract_check"] = check
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    LOGGER.info("Updated metadata: %s", metadata_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    items_json = (
        Path(args.items_json).expanduser().resolve()
        if args.items_json
        else dataset_dir / "items.json"
    )
    id_mappings_json = (
        Path(args.id_mappings_json).expanduser().resolve()
        if args.id_mappings_json
        else dataset_dir / "id_mappings.json"
    )
    output_file = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else dataset_dir / "text_features.npy"
    )
    raw_interaction_csv = (
        Path(args.raw_interaction_csv).expanduser().resolve()
        if args.raw_interaction_csv
        else None
    )

    if not id_mappings_json.exists():
        raise FileNotFoundError(f"id_mappings_json not found: {id_mappings_json}")
    if not items_json.exists():
        raise FileNotFoundError(f"items_json not found: {items_json}")
    if raw_interaction_csv is not None and not raw_interaction_csv.exists():
        raise FileNotFoundError(f"raw_interaction_csv not found: {raw_interaction_csv}")

    item_map = load_item_map(id_mappings_json)
    text, stats = build_sentence_transformer_text_features(
        item_map=item_map,
        items_metadata_json=items_json,
        source_csv=raw_interaction_csv,
        encoding=args.encoding,
        title_column=args.title_column,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, text)
    LOGGER.info("Wrote text features: %s shape=%s stats=%s", output_file, text.shape, stats)

    check = {}
    if not args.skip_contract_check:
        image = np.load(dataset_dir / "image_features.npy", allow_pickle=False)
        rows = load_inter_rows(dataset_dir / "inter.csv")
        check = run_contract_checks(rows, image, text)
        LOGGER.info("Contract checks passed: %s", check)

    update_metadata(
        metadata_path=dataset_dir / "metadata.json",
        args=args,
        items_json=items_json,
        output_file=output_file,
        stats=stats,
        check=check,
    )


if __name__ == "__main__":
    main()
