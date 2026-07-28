#!/usr/bin/env python3
"""Export the final no-match item list.

Final no_match means both ASR fields are JSON null in item_final_fixed.json.
Empty strings are not treated as null.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export item_final_no_match.json.")
    parser.add_argument("--items-final", default="item_final_fixed.json")
    parser.add_argument("--output", default="item_final_no_match.json")
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} contains non-object rows")
    return data


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    items = load_json_list(Path(args.items_final))

    final_no_match = [
        row
        for row in items
        if row.get("asr_text") is None and row.get("asr_text_cn") is None
    ]
    write_json(Path(args.output), final_no_match)

    print(f"wrote {args.output}")
    print(f"items_total: {len(items)}")
    print("definition: asr_text is null and asr_text_cn is null")
    print(f"final_no_match_rows: {len(final_no_match)}")


if __name__ == "__main__":
    main()
