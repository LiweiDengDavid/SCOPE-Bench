#!/usr/bin/env python3
"""Regenerate item_fixed.json from items.json and VIDEO_ID_ALIGNMENT_NOTE.md.

The markdown note is the source of truth for the piecewise raw/asr alignment
map. Canonical item fields stay tied to the original video_id; only ASR-side
fields are copied from the mapped raw/asr row, or set to null for canonical
null-ASR rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


CANONICAL_FIELDS = {
    "caption",
    "category",
    "first_level_category",
    "second_level_category",
    "third_level_category",
    "category_cn",
    "first_level_category_cn",
    "second_level_category_cn",
    "third_level_category_cn",
    "source_pid",
    "source_title_cn",
    "source_match_title_cn",
}

SEGMENT_RE = re.compile(
    r"video_id\s+(\d+)(?:\s+-\s+(\d+))?\s+"
    r"(?:offset\s+([+-]?\d+)|ASR\s*=\s*null)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate item_fixed.json using the alignment table in VIDEO_ID_ALIGNMENT_NOTE.md."
    )
    parser.add_argument("--items", default="items.json", help="input items JSON file")
    parser.add_argument(
        "--note",
        default="VIDEO_ID_ALIGNMENT_NOTE.md",
        help="markdown note containing the alignment segment table",
    )
    parser.add_argument("--output", default="item_fixed.json", help="output fixed JSON file")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="write compact JSON instead of the default pretty-printed JSON",
    )
    return parser.parse_args()


def parse_note(note_path: Path) -> list[tuple[int, int, int | None]]:
    note = note_path.read_text(encoding="utf-8")
    block_match = re.search(r"```text\n(.*?)\n```", note, flags=re.S)
    if not block_match:
        raise ValueError(f"No ```text segment table found in {note_path}")

    segments: list[tuple[int, int, int | None]] = []
    for line_no, raw_line in enumerate(block_match.group(1).splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        match = SEGMENT_RE.fullmatch(line)
        if not match:
            raise ValueError(f"Cannot parse segment table line {line_no}: {line!r}")
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        offset = int(match.group(3)) if match.group(3) is not None else None
        if end < start:
            raise ValueError(f"Invalid descending segment on line {line_no}: {line!r}")
        segments.append((start, end, offset))

    if not segments:
        raise ValueError(f"Empty segment table in {note_path}")

    validate_explicit_null_list(note, segments)
    return segments


def validate_explicit_null_list(
    note: str, segments: list[tuple[int, int, int | None]]
) -> None:
    status_null_match = re.search(
        r"explicit canonical-side null-ASR list is applied\s*"
        r"\(currently:\s*(.*?)\)\.",
        note,
        flags=re.S,
    )
    if not status_null_match:
        raise ValueError("Cannot find explicit canonical-side null-ASR list in the note")

    explicit_null_ids = sorted(
        int(video_id) for video_id in re.findall(r"video_id\s+(\d+)", status_null_match.group(1))
    )
    table_null_ids = sorted(
        video_id
        for start, end, offset in segments
        if offset is None
        for video_id in range(start, end + 1)
    )

    if explicit_null_ids != table_null_ids:
        only_status = sorted(set(explicit_null_ids) - set(table_null_ids))
        only_table = sorted(set(table_null_ids) - set(explicit_null_ids))
        raise ValueError(
            "Canonical null-ASR list disagrees with segment table: "
            f"only_status={only_status}, only_table={only_table}"
        )


def load_items(items_path: Path) -> list[dict[str, Any]]:
    with items_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{items_path} is not a JSON list")
    if not all(isinstance(row, dict) for row in items):
        raise ValueError(f"{items_path} must contain only JSON objects")

    expected_ids = list(range(1, len(items) + 1))
    actual_ids = [row.get("video_id") for row in items]
    if actual_ids != expected_ids:
        raise ValueError(f"{items_path} video_id values are not continuous 1..N")
    return items


def validate_segments(
    segments: list[tuple[int, int, int | None]], item_count: int
) -> None:
    if segments[0][0] != 1 or segments[-1][1] != item_count:
        raise ValueError(
            f"Segment table covers {segments[0][0]}..{segments[-1][1]}, "
            f"but items cover 1..{item_count}"
        )

    previous_end = 0
    for start, end, offset in segments:
        if start != previous_end + 1:
            raise ValueError(f"Segment table gap/overlap before {start}-{end}")
        if offset is not None:
            raw_start = start + offset
            raw_end = end + offset
            if raw_start < 1 or raw_end > item_count:
                raise ValueError(
                    f"Mapped raw_id out of range for segment {start}-{end} offset {offset:+d}"
                )
        previous_end = end


def infer_raw_fields(items: list[dict[str, Any]]) -> list[str]:
    field_order = list(items[0].keys())
    raw_fields = [
        field
        for field in field_order
        if field != "video_id" and field not in CANONICAL_FIELDS
    ]
    if raw_fields != ["asr_text", "asr_text_cn"]:
        raise ValueError(f"Unexpected raw/asr-side fields inferred: {raw_fields!r}")
    return raw_fields


def build_fixed_items(
    items: list[dict[str, Any]],
    segments: list[tuple[int, int, int | None]],
    raw_fields: list[str],
) -> tuple[list[dict[str, Any]], int, int]:
    fixed: list[dict[str, Any]] = []
    segment_index = 0
    null_rows = 0
    changed_asr_rows = 0

    for canonical_id, row in enumerate(items, 1):
        while not (segments[segment_index][0] <= canonical_id <= segments[segment_index][1]):
            segment_index += 1

        _, _, offset = segments[segment_index]
        new_row = dict(row)

        if offset is None:
            null_rows += 1
            for field in raw_fields:
                new_row[field] = None
        else:
            raw_row = items[canonical_id + offset - 1]
            for field in raw_fields:
                new_row[field] = raw_row[field]

        if any(row[field] != new_row[field] for field in raw_fields):
            changed_asr_rows += 1
        fixed.append(new_row)

    return fixed, null_rows, changed_asr_rows


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
    note_path = Path(args.note)
    output_path = Path(args.output)

    segments = parse_note(note_path)
    items = load_items(items_path)
    validate_segments(segments, len(items))
    raw_fields = infer_raw_fields(items)
    fixed, null_rows, changed_asr_rows = build_fixed_items(items, segments, raw_fields)
    write_json(output_path, fixed, args.compact)

    offset_counts: dict[str, int] = {}
    for start, end, offset in segments:
        key = "null" if offset is None else f"{offset:+d}"
        offset_counts[key] = offset_counts.get(key, 0) + (end - start + 1)

    print(f"wrote {output_path}")
    print(f"items: {len(fixed)}")
    print(f"segments: {len(segments)}")
    print(f"raw_side_fields: {','.join(raw_fields)}")
    print(f"canonical_null_asr_count: {null_rows}")
    print(f"asr_changed_rows: {changed_asr_rows}")
    print(f"offset_counts: {json.dumps(offset_counts, sort_keys=True)}")


if __name__ == "__main__":
    main()
