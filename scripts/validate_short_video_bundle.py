#!/usr/bin/env python3
"""Validate the bundled ShortVideoSampled and ShortVideoFull artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("ShortVideoSampled", "ShortVideoFull")
REQUIRED_FILES = (
    "inter.csv",
    "image_features.npy",
    "text_features.npy",
    "id_mappings.json",
    "items.json",
    "items_final_fixed.json",
    "metadata.json",
    "missing_image_feature_raw_pids.txt",
)


def count_interactions(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row_count = sum(1 for _ in reader)
    return row_count, header


def validate_dataset(dataset_name: str) -> None:
    dataset_dir = REPO_ROOT / "datasets" / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    missing = [name for name in REQUIRED_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_name} is missing required files: {', '.join(missing)}"
        )

    with (dataset_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    contract = metadata["contract_check"]

    row_count, header = count_interactions(dataset_dir / "inter.csv")
    expected_header = ["userID", "itemID", "split_label", "timestamp"]
    if header != expected_header:
        raise ValueError(
            f"{dataset_name}/inter.csv header {header} != {expected_header}"
        )
    if row_count != contract["num_interactions"]:
        raise ValueError(
            f"{dataset_name} interaction rows {row_count} != "
            f"{contract['num_interactions']}"
        )

    image = np.load(dataset_dir / "image_features.npy", mmap_mode="r")
    text = np.load(dataset_dir / "text_features.npy", mmap_mode="r")
    if list(image.shape) != contract["image_shape"]:
        raise ValueError(
            f"{dataset_name} image shape {list(image.shape)} != "
            f"{contract['image_shape']}"
        )
    if list(text.shape) != contract["text_shape"]:
        raise ValueError(
            f"{dataset_name} text shape {list(text.shape)} != "
            f"{contract['text_shape']}"
        )
    if image.shape[0] != contract["num_items"] or text.shape[0] != contract["num_items"]:
        raise ValueError(f"{dataset_name} feature rows do not match num_items")
    if not metadata.get("bundle_ready"):
        raise ValueError(f"{dataset_name} metadata is not marked bundle_ready")

    print(
        f"{dataset_name}: OK "
        f"users={contract['num_users']} items={contract['num_items']} "
        f"interactions={row_count} image={tuple(image.shape)} text={tuple(text.shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Bundles to validate (default: both).",
    )
    args = parser.parse_args()
    for dataset_name in args.datasets:
        validate_dataset(dataset_name)


if __name__ == "__main__":
    main()
