#!/usr/bin/env python3
"""Build items_final_fixed.json from the final Step1/Step2 alignment outputs.

The final file keeps the same schema and canonical item fields as items.json.
Rows in the final match partition keep their matched ASR fields. Rows in the
final no_match partition are kept as interaction-side items, but their ASR
fields are set to null. Alignment diagnostics and raw_file fields are ignored.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ASR_FIELDS = ("asr_text", "asr_text_cn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge final Step1/Step2 match and no_match outputs into items_final_fixed.json."
    )
    parser.add_argument("--items", default="items.json", help="canonical input items JSON")
    parser.add_argument(
        "--match",
        default="final_hymt_step1_step2_alignment/items_match_hymt_step1_step2.json",
        help="final matched item JSON",
    )
    parser.add_argument(
        "--no-match",
        default="final_hymt_step1_step2_alignment/items_no_match_hymt_step1_step2.json",
        help="final no-match item JSON",
    )
    parser.add_argument(
        "--output",
        default="items_final_fixed.json",
        help="merged final item JSON output",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="write compact JSON instead of the default pretty-printed JSON",
    )
    parser.add_argument(
        "--include-raw-file",
        action="store_true",
        help="append raw_video_id and raw_file_mp4 fields from the final alignment partitions",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} must contain only JSON objects")
    return data


def validate_base_items(items: list[dict[str, Any]], path: Path) -> list[str]:
    if not items:
        raise ValueError(f"{path} is empty")
    fields = list(items[0].keys())
    missing_asr = [field for field in ASR_FIELDS if field not in fields]
    if missing_asr:
        raise ValueError(f"{path} is missing ASR fields: {missing_asr}")
    expected_ids = list(range(1, len(items) + 1))
    actual_ids = [row.get("video_id") for row in items]
    if actual_ids != expected_ids:
        raise ValueError(f"{path} video_id values are not continuous 1..N")
    return fields


def index_partition(
    rows: list[dict[str, Any]], path: Path, valid_ids: set[int]
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, int):
            raise ValueError(f"{path} contains non-integer video_id: {video_id!r}")
        if video_id not in valid_ids:
            raise ValueError(f"{path} contains out-of-range video_id: {video_id}")
        if video_id in indexed:
            raise ValueError(f"{path} contains duplicate video_id: {video_id}")
        indexed[video_id] = row
    return indexed


def validate_partition_canonical_fields(
    items: list[dict[str, Any]],
    base_fields: list[str],
    match_by_id: dict[int, dict[str, Any]],
    no_match_by_id: dict[int, dict[str, Any]],
) -> None:
    canonical_fields = [
        field for field in base_fields if field not in ASR_FIELDS and field != "video_id"
    ]
    for partition_name, partition in (
        ("match", match_by_id),
        ("no_match", no_match_by_id),
    ):
        for video_id, row in partition.items():
            base_row = items[video_id - 1]
            mismatched = [
                field
                for field in canonical_fields
                if row.get(field) != base_row.get(field)
            ]
            if mismatched:
                raise ValueError(
                    f"{partition_name} row video_id={video_id} has canonical field "
                    f"mismatches against items.json: {mismatched}"
                )


def build_final_items(
    items: list[dict[str, Any]],
    base_fields: list[str],
    match_by_id: dict[int, dict[str, Any]],
    no_match_by_id: dict[int, dict[str, Any]],
    include_raw_file: bool,
) -> list[dict[str, Any]]:
    overlap = set(match_by_id).intersection(no_match_by_id)
    if overlap:
        raise ValueError(f"match/no_match partitions overlap, sample={sorted(overlap)[:10]}")

    expected_ids = set(range(1, len(items) + 1))
    covered_ids = set(match_by_id).union(no_match_by_id)
    if covered_ids != expected_ids:
        missing = sorted(expected_ids - covered_ids)
        extra = sorted(covered_ids - expected_ids)
        raise ValueError(
            "match/no_match partitions do not cover items.json exactly: "
            f"missing_sample={missing[:10]}, extra_sample={extra[:10]}"
        )

    final_items: list[dict[str, Any]] = []
    for video_id, base_row in enumerate(items, 1):
        new_row = {field: base_row.get(field) for field in base_fields}
        if video_id in match_by_id:
            match_row = match_by_id[video_id]
            for field in ASR_FIELDS:
                new_row[field] = match_row.get(field)
            raw_row = match_row
        else:
            raw_row = no_match_by_id[video_id]
            for field in ASR_FIELDS:
                new_row[field] = None
        if include_raw_file:
            new_row["raw_video_id"] = raw_row.get("raw_video_id")
            new_row["raw_file_mp4"] = raw_row.get("raw_file_mp4")
        final_items.append(new_row)
    return final_items


def write_json(path: Path, data: list[dict[str, Any]], compact: bool) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    items_path = Path(args.items)
    match_path = Path(args.match)
    no_match_path = Path(args.no_match)
    output_path = Path(args.output)

    items = load_json_list(items_path)
    base_fields = validate_base_items(items, items_path)
    valid_ids = set(range(1, len(items) + 1))

    match_rows = load_json_list(match_path)
    no_match_rows = load_json_list(no_match_path)
    match_by_id = index_partition(match_rows, match_path, valid_ids)
    no_match_by_id = index_partition(no_match_rows, no_match_path, valid_ids)

    validate_partition_canonical_fields(items, base_fields, match_by_id, no_match_by_id)
    final_items = build_final_items(
        items,
        base_fields,
        match_by_id,
        no_match_by_id,
        args.include_raw_file,
    )
    write_json(output_path, final_items, args.compact)

    no_match_asr_null_rows = sum(
        all(row[field] is None for field in ASR_FIELDS)
        for row in final_items
        if row["video_id"] in no_match_by_id
    )
    match_asr_non_null_rows = sum(
        any(row[field] is not None for field in ASR_FIELDS)
        for row in final_items
        if row["video_id"] in match_by_id
    )
    ignored_match_fields = sorted(set(match_rows[0]) - set(base_fields)) if match_rows else []
    ignored_no_match_fields = sorted(set(no_match_rows[0]) - set(base_fields)) if no_match_rows else []

    print(f"wrote {output_path}")
    print(f"items: {len(final_items)}")
    print(f"match_rows: {len(match_by_id)}")
    print(f"no_match_rows: {len(no_match_by_id)}")
    print(f"match_rows_with_non_null_asr: {match_asr_non_null_rows}")
    print(f"no_match_rows_with_null_asr: {no_match_asr_null_rows}")
    print(f"schema_fields: {','.join(base_fields)}")
    print(f"include_raw_file: {args.include_raw_file}")
    print(f"ignored_match_fields: {','.join(ignored_match_fields)}")
    print(f"ignored_no_match_fields: {','.join(ignored_no_match_fields)}")


if __name__ == "__main__":
    main()
