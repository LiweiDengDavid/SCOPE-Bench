"""Convert cognitive-depth score JSONL into CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluate_videos import parse_score_payload

csv.field_size_limit(sys.maxsize)

CSV_COLUMNS = [
    "video_id",
    "caption",
    "category",
    "asr_text",
    "score",
    "level_name",
    "reason",
    "evidence",
    "confidence",
    "parse_error",
    "error",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
    "model_call_count",
    "request_id",
    "response_model",
    "raw_response",
]


def _sort_key(record: Dict[str, Any]) -> tuple:
    vid = record.get("video_id")
    if isinstance(vid, (int, float)):
        return (0, float(vid))
    if isinstance(vid, str):
        try:
            return (0, float(vid))
        except ValueError:
            return (1, vid)
    return (2, "")


def _normalize_score(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if math.isnan(value) or not value.is_integer():
            return None
        value = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.upper() == "NAN":
            return None
        try:
            as_float = float(text)
        except ValueError:
            return None
        if math.isnan(as_float) or not as_float.is_integer():
            return None
        value = int(as_float)
    elif not isinstance(value, int):
        return None

    if 0 <= value <= 6:
        return int(value)
    return None


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] line {lineno}: malformed JSON skipped ({exc})",
                    file=sys.stderr,
                )
    return records


def load_items_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        items = json.loads(text)
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        vid = item.get("video_id")
        if vid is None:
            continue
        out[str(vid)] = item
    return out


def _row_for(record: Dict[str, Any], items_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {col: record.get(col, "") for col in CSV_COLUMNS}
    usage = record.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            if usage.get(key) is not None:
                row[key] = usage[key]
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens") is not None:
            row["cached_tokens"] = prompt_details["cached_tokens"]
    vid_key = str(record.get("video_id")) if record.get("video_id") is not None else None
    item = items_map.get(vid_key) if vid_key else None
    if item:
        for key in ("caption", "category", "asr_text"):
            current = row.get(key)
            if current is None or (isinstance(current, str) and not current.strip()):
                fallback = item.get(key)
                if fallback is not None:
                    row[key] = fallback
    score = _normalize_score(record.get("score"))

    # Repair one class of historical parse failures from saved raw_response
    # (e.g. malformed JSON with unescaped quotes).
    if score is None and isinstance(record.get("raw_response"), str):
        repaired = parse_score_payload(record["raw_response"])
        repaired_score = _normalize_score(repaired.get("score"))
        if repaired_score is not None:
            score = repaired_score
            row["score"] = repaired_score
            row["level_name"] = repaired.get("level_name", row.get("level_name", ""))
            row["reason"] = repaired.get("reason", row.get("reason", ""))
            row["confidence"] = repaired.get("confidence", row.get("confidence", ""))
            row["parse_error"] = repaired.get("parse_error", row.get("parse_error", ""))
            row["evidence"] = repaired.get("evidence", row.get("evidence", ""))

    evidence = row.get("evidence")
    if isinstance(evidence, list):
        row["evidence"] = " | ".join(str(x).strip() for x in evidence if str(x).strip())
    row["score"] = score if score is not None else "NAN"
    return row


def write_csv(
    records: List[Dict[str, Any]],
    out_path: Path,
    items_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_written: List[Dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = _row_for(rec, items_map)
            writer.writerow(row)
            rows_written.append(row)
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/scores.jsonl"),
        help="Input JSONL from evaluate_videos.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scores.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no_sort",
        action="store_true",
        help="Keep original order instead of sorting by video_id.",
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("data/items.json"),
        help="Optional items JSON/JSONL for backfilling caption/category/asr_text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    items_map = load_items_map(args.items)
    if not args.no_sort:
        records.sort(key=_sort_key)
    rows = write_csv(records, args.output, items_map)

    n_total = len(records)
    n_error = sum(1 for r in rows if r.get("error"))
    n_parse_error = sum(1 for r in rows if r.get("parse_error"))
    n_valid_score = sum(1 for r in rows if _normalize_score(r.get("score")) is not None)
    print(f"Read   {n_total} records from {args.input}")
    print(f"API errors: {n_error}")
    print(f"Parse errors: {n_parse_error}")
    print(f"Rows with valid 0~6 score: {n_valid_score}")
    print(f"Wrote  -> {args.output}")


if __name__ == "__main__":
    main()
