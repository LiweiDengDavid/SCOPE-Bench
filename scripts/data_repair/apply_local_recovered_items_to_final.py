#!/usr/bin/env python3
"""Apply local recovered no-match items to items_final_fixed.json.

The recovered item file may contain local-pass diagnostic fields. This script
updates only the final item schema fields: asr_text, asr_text_cn, raw_video_id,
and raw_file_mp4. All canonical item/source/category fields remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


UPDATED_FIELDS = ("asr_text", "asr_text_cn", "raw_video_id", "raw_file_mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge local recovered no-match rows into items_final_fixed.json."
    )
    parser.add_argument("--items-final", default="items_final_fixed.json")
    parser.add_argument(
        "--recovered",
        default=(
            "llm_rescore_stats/"
            "local_no_match_candidate_pass_pm5_dur0p3_recovered_items.json"
        ),
    )
    parser.add_argument("--output", default="items_final_fixed.json")
    parser.add_argument(
        "--sync-copy",
        default="items_final_fixed_with_raw_file.json",
        help="optional duplicate output to keep in sync if the file exists",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} contains non-object rows")
    return data


def empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    items_path = Path(args.items_final)
    recovered_path = Path(args.recovered)
    output_path = Path(args.output)

    items = load_json_list(items_path)
    recovered = load_json_list(recovered_path)
    if not items:
        raise ValueError(f"{items_path} is empty")

    base_fields = list(items[0].keys())
    missing = [field for field in UPDATED_FIELDS if field not in base_fields]
    if missing:
        raise ValueError(f"{items_path} is missing fields: {missing}")

    item_by_id = {}
    for row in items:
        video_id = row.get("video_id")
        if not isinstance(video_id, int):
            raise ValueError(f"{items_path} has non-integer video_id: {video_id!r}")
        if video_id in item_by_id:
            raise ValueError(f"{items_path} has duplicate video_id: {video_id}")
        item_by_id[video_id] = row

    recovered_by_id = {}
    for row in recovered:
        video_id = row.get("video_id")
        if not isinstance(video_id, int):
            raise ValueError(
                f"{recovered_path} has non-integer video_id: {video_id!r}"
            )
        if video_id not in item_by_id:
            raise ValueError(f"{recovered_path} has out-of-range video_id: {video_id}")
        if video_id in recovered_by_id:
            raise ValueError(f"{recovered_path} has duplicate video_id: {video_id}")
        recovered_by_id[video_id] = row

    updated_rows = 0
    changed_raw_rows = 0
    changed_asr_rows = 0
    for video_id, recovered_row in recovered_by_id.items():
        row = item_by_id[video_id]
        before_raw = (row.get("raw_video_id"), row.get("raw_file_mp4"))
        before_asr = (row.get("asr_text"), row.get("asr_text_cn"))
        for field in UPDATED_FIELDS:
            row[field] = recovered_row.get(field)
        after_raw = (row.get("raw_video_id"), row.get("raw_file_mp4"))
        after_asr = (row.get("asr_text"), row.get("asr_text_cn"))
        updated_rows += 1
        changed_raw_rows += before_raw != after_raw
        changed_asr_rows += before_asr != after_asr

    # Preserve schema and field order exactly.
    final_items = [{field: row.get(field) for field in base_fields} for row in items]
    write_json(output_path, final_items)

    sync_path = Path(args.sync_copy) if args.sync_copy else None
    if sync_path and sync_path.exists() and not sync_path.is_symlink():
        write_json(sync_path, final_items)

    asr_both_empty = sum(
        empty(row.get("asr_text")) and empty(row.get("asr_text_cn"))
        for row in final_items
    )
    recovered_both_empty = sum(
        empty(row.get("asr_text")) and empty(row.get("asr_text_cn"))
        for row in recovered
    )

    print(f"wrote {output_path}")
    if sync_path and sync_path.exists() and not sync_path.is_symlink():
        print(f"synced {sync_path}")
    print(f"items: {len(final_items)}")
    print(f"updated_rows: {updated_rows}")
    print(f"changed_raw_rows: {changed_raw_rows}")
    print(f"changed_asr_rows: {changed_asr_rows}")
    print(f"recovered_rows_with_nonempty_asr: {updated_rows - recovered_both_empty}")
    print(f"final_rows_with_null_asr: {asr_both_empty}")
    print(f"schema_fields: {','.join(base_fields)}")


if __name__ == "__main__":
    main()
