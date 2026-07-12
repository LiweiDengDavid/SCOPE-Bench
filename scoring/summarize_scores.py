#!/usr/bin/env python3
"""Create a compact, publishable summary from a full Qwen score JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sortable_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda value: (value is None, str(value)))
    }


def main() -> None:
    args = parse_args()
    scores: Counter[Any] = Counter()
    confidence: Counter[Any] = Counter()
    total = errors = parse_errors = valid = null_scores = 0

    with args.input.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if record.get("error"):
                errors += 1
            if record.get("parse_error"):
                parse_errors += 1
            score = record.get("score")
            if score is None:
                null_scores += 1
            elif isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 6:
                valid += 1
                scores[score] += 1
            confidence[record.get("confidence") or "missing"] += 1

    payload = {
        "source_file": args.input.name,
        "total_records": total,
        "valid_integer_scores": valid,
        "null_scores": null_scores,
        "api_error_records": errors,
        "parse_error_records": parse_errors,
        "score_distribution": sortable_counter(scores),
        "confidence_distribution": sortable_counter(confidence),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

