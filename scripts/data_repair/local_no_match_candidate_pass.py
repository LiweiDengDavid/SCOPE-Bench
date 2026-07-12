#!/usr/bin/env python3
"""Local raw-file candidate pass for high-value no-match items.

This pass targets rows kept in items_final_fixed.json with JSON null ASR fields.
For each row, it searches raw files near the current raw_video_id and keeps
local candidates whose raw mp4 duration matches the interaction-side duration.
Manual raw/source confirmations from Video_ID_Alignment_Manual_log.md are
parsed and marked in the output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ASR_FIELDS = ("asr_text", "asr_text_cn")
MANUAL_RE = re.compile(
    r"raw_file/(\d+)\.mp4\s+->\s+source_pid\s+(\d+)"
    r"(?:,\s+video_id\s+(\d+))?",
)
CN_RE = re.compile(r"[\u4e00-\u9fff]+")
EN_RE = re.compile(r"[A-Za-z0-9]{3,}")
PUNCT_RE = re.compile(
    r"[\s\u3000，。！？、；：：“”‘’（）()《》\[\]【】{}<>"
    r"「」『』,.!?;:\"'`~@#$%^&*_+=|\\/\-]+"
)
STOP_TERMS = {
    "一个",
    "这个",
    "那个",
    "什么",
    "怎么",
    "可以",
    "不能",
    "不是",
    "没有",
    "就是",
    "还是",
    "起来",
    "出来",
    "进去",
    "大家",
    "今天",
    "视频",
    "看看",
    "一下",
    "真的",
    "这么",
    "那么",
    "时候",
    "因为",
    "如果",
    "然后",
    "但是",
    "所以",
    "我们",
    "你们",
    "他们",
    "它们",
    "自己",
    "现在",
    "最后",
    "第一",
    "第二",
    "直接",
    "不会",
    "不要",
    "原来",
    "结果",
    "竟然",
    "开始",
    "发现",
    "知道",
    "这是",
    "这也",
    "为了",
    "成为",
    "孩子",
    "女子",
    "男子",
    "女孩",
    "男孩",
    "女人",
    "男人",
    "的人",
    "的是",
    "你的",
    "我的",
    "他的",
    "她的",
    "它的",
}
STOP_EN = {"the", "and", "for", "with", "this", "that", "you", "your", "are", "was"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search current_raw_id +/- radius for no-match local candidates."
    )
    parser.add_argument("--items-final", default="items_final_fixed.json")
    parser.add_argument("--items-raw", default="items.json")
    parser.add_argument("--durations", default="items_raw_mp4_durations.json")
    parser.add_argument("--interactions", default="interaction.csv")
    parser.add_argument("--manual-log", default="Video_ID_Alignment_Manual_log.md")
    parser.add_argument("--output-dir", default="llm_rescore_stats")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--duration-threshold", type=float, default=0.3)
    return parser.parse_args()


def empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def asr_null(row: dict[str, Any]) -> bool:
    return all(row.get(field) is None for field in ASR_FIELDS)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def parse_manual_log(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    by_source_pid: dict[int, int] = {}
    by_video_id: dict[int, int] = {}
    if not path.exists():
        return by_source_pid, by_video_id

    text = path.read_text(encoding="utf-8")
    for raw_id_s, source_pid_s, video_id_s in MANUAL_RE.findall(text):
        raw_id = int(raw_id_s)
        source_pid = int(source_pid_s)
        by_source_pid[source_pid] = raw_id
        if video_id_s:
            by_video_id[int(video_id_s)] = raw_id
    return by_source_pid, by_video_id


def parse_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_interaction_durations(
    interactions_path: Path, source_pids: set[int]
) -> dict[int, dict[str, Any]]:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    pid_strings = {str(pid) for pid in source_pids}
    counts = {pid: Counter() for pid in pid_strings}
    row_counts = Counter()

    with interactions_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pid"]
            if pid not in counts:
                continue
            duration = parse_float(row.get("duration"))
            if duration is None:
                continue
            counts[pid][duration] += 1
            row_counts[pid] += 1

    result: dict[int, dict[str, Any]] = {}
    for pid_s, counter in counts.items():
        if not counter:
            continue
        duration, count = counter.most_common(1)[0]
        result[int(pid_s)] = {
            "interaction_duration": duration,
            "interaction_duration_count": count,
            "interaction_rows_for_item": row_counts[pid_s],
        }
    return result


def normalize(text: Any) -> str:
    return PUNCT_RE.sub("", str(text or "")).lower()


def title_terms(text: str) -> dict[str, float]:
    terms: dict[str, float] = {}
    for span in CN_RE.findall(text):
        span_len = len(span)
        if span_len < 2:
            continue
        lengths = [2] if span_len == 2 else range(3, min(6, span_len) + 1)
        for length in lengths:
            weight = 1.0 + max(0, length - 2) * 0.55
            if length == span_len and span_len >= 3:
                weight += 0.7
            for idx in range(0, span_len - length + 1):
                gram = span[idx : idx + length]
                if gram in STOP_TERMS or gram.isdigit():
                    continue
                if len(gram) <= 3 and (gram.startswith("的") or gram.endswith("的")):
                    continue
                terms[gram] = max(terms.get(gram, 0.0), weight)

    for token in EN_RE.findall(text.lower()):
        if token not in STOP_EN and not token.isdigit():
            terms[token] = max(terms.get(token, 0.0), 1.7)
    return terms


def overlap_score(terms: dict[str, float], raw_text: Any) -> tuple[float, list[str]]:
    raw_norm = normalize(raw_text)
    if not terms or not raw_norm:
        return 0.0, []
    total_weight = sum(terms.values())
    matched = [term for term in terms if term in raw_norm]
    matched.sort(key=lambda term: (len(term), terms[term]), reverse=True)
    score = sum(terms[term] for term in matched) / total_weight if total_weight else 0.0
    return score, matched[:20]


def confidence_for_candidate(
    manual_confirmed: bool, duration_diff: float, overlap: float
) -> str:
    if manual_confirmed:
        return "manual"
    if duration_diff <= 0.15 and overlap >= 0.05:
        return "high"
    if duration_diff <= 0.15:
        return "duration_high"
    if overlap >= 0.05:
        return "medium"
    return "duration_only"


def build_recovered_item(
    item: dict[str, Any],
    candidate: dict[str, Any],
    raw_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    raw_id = int(candidate["candidate_raw_video_id"])
    raw_row = raw_rows.get(raw_id, {})
    recovered = dict(item)
    recovered["asr_text"] = raw_row.get("asr_text")
    recovered["asr_text_cn"] = raw_row.get("asr_text_cn")
    recovered["raw_video_id"] = raw_id
    recovered["raw_file_mp4"] = candidate.get("candidate_raw_file_mp4")
    recovered["local_pass_status"] = "local_pm5_duration_recovered"
    recovered["local_pass_candidate_confidence"] = candidate.get(
        "candidate_confidence"
    )
    recovered["local_pass_duration_diff"] = candidate.get("candidate_duration_diff")
    recovered["local_pass_title_asr_overlap"] = candidate.get(
        "candidate_title_asr_overlap"
    )
    recovered["local_pass_manual_confirmed"] = candidate.get("manual_confirmed")
    recovered["local_pass_previous_raw_video_id"] = candidate.get(
        "current_raw_video_id"
    )
    recovered["local_pass_previous_raw_file_mp4"] = candidate.get(
        "current_raw_file_mp4"
    )
    return recovered


def main() -> None:
    args = parse_args()
    items_final = load_json_list(Path(args.items_final))
    items_raw = load_json_list(Path(args.items_raw))
    durations = load_json_list(Path(args.durations))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = {int(row["video_id"]): row for row in items_raw}
    raw_info = {int(row["raw_video_id"]): row for row in durations}
    item_count = len(items_final)

    manual_by_pid, manual_by_video = parse_manual_log(Path(args.manual_log))

    no_match_items = [
        row
        for row in items_final
        if asr_null(row)
    ]
    source_pids = {
        int(row["source_pid"])
        for row in no_match_items
        if not empty(row.get("source_pid"))
    }
    interaction_duration_by_pid = collect_interaction_durations(
        Path(args.interactions), source_pids
    )

    all_candidates: list[dict[str, Any]] = []
    best_matches: list[dict[str, Any]] = []
    recovered_items: list[dict[str, Any]] = []
    skipped_no_duration = 0
    skipped_no_center = 0

    for item in no_match_items:
        video_id = int(item["video_id"])
        source_pid = int(item["source_pid"])
        duration_info = interaction_duration_by_pid.get(source_pid)
        if duration_info is None:
            skipped_no_duration += 1
            continue

        current_raw = item.get("raw_video_id")
        if empty(current_raw):
            skipped_no_center += 1
            continue
        current_raw_id = int(current_raw)

        title = " ".join(
            str(item.get(field) or "")
            for field in ("source_title_cn", "source_match_title_cn")
        )
        terms = title_terms(title)
        interaction_duration = float(duration_info["interaction_duration"])
        candidates: list[dict[str, Any]] = []

        for raw_id in range(
            max(1, current_raw_id - args.radius),
            min(item_count, current_raw_id + args.radius) + 1,
        ):
            info = raw_info.get(raw_id)
            if not info or not info.get("raw_file_exists", False):
                continue
            raw_duration = parse_float(info.get("raw_duration"))
            if raw_duration is None:
                continue
            duration_diff = abs(raw_duration - interaction_duration)
            if duration_diff >= args.duration_threshold:
                continue

            raw_row = raw_rows.get(raw_id, {})
            overlap, matched_terms = overlap_score(terms, raw_row.get("asr_text_cn"))
            manual_confirmed = (
                manual_by_pid.get(source_pid) == raw_id
                or manual_by_video.get(video_id) == raw_id
            )
            candidate = {
                "video_id": video_id,
                "source_pid": source_pid,
                "source_title_cn": item.get("source_title_cn"),
                "source_match_title_cn": item.get("source_match_title_cn"),
                "caption": item.get("caption"),
                "category": item.get("category"),
                "category_cn": item.get("category_cn"),
                "current_raw_video_id": current_raw_id,
                "current_raw_file_mp4": item.get("raw_file_mp4"),
                "candidate_raw_video_id": raw_id,
                "candidate_raw_file_mp4": info.get("raw_file_mp4"),
                "candidate_distance_from_current_raw": raw_id - current_raw_id,
                "interaction_duration": interaction_duration,
                "interaction_duration_count": duration_info[
                    "interaction_duration_count"
                ],
                "interaction_rows_for_item": duration_info["interaction_rows_for_item"],
                "candidate_raw_duration": raw_duration,
                "candidate_duration_diff": round(duration_diff, 6),
                "candidate_title_asr_overlap": round(overlap, 6),
                "candidate_matched_terms": matched_terms,
                "candidate_asr_text_cn_preview": str(
                    raw_row.get("asr_text_cn") or ""
                )[:180],
                "manual_confirmed": manual_confirmed,
                "candidate_confidence": confidence_for_candidate(
                    manual_confirmed, duration_diff, overlap
                ),
            }
            candidates.append(candidate)

        candidates.sort(
            key=lambda row: (
                row["manual_confirmed"],
                -row["candidate_duration_diff"],
                row["candidate_title_asr_overlap"],
                -abs(row["candidate_distance_from_current_raw"]),
            ),
            reverse=True,
        )
        for rank, candidate in enumerate(candidates, 1):
            candidate["candidate_rank"] = rank
            candidate["candidate_count_for_item"] = len(candidates)
            all_candidates.append(candidate)

        if candidates:
            best_match = candidates[0]
            best_matches.append(best_match)
            recovered_items.append(build_recovered_item(item, best_match, raw_rows))

    confidence_counts = Counter(row["candidate_confidence"] for row in best_matches)
    distance_counts = Counter(
        str(row["candidate_distance_from_current_raw"]) for row in best_matches
    )
    candidate_count_distribution = Counter(
        str(row["candidate_count_for_item"]) for row in best_matches
    )

    prefix = (
        f"local_no_match_candidate_pass_pm{args.radius}_"
        f"dur{str(args.duration_threshold).replace('.', 'p')}"
    )
    all_path = output_dir / f"{prefix}_all_candidates.json"
    best_path = output_dir / f"{prefix}_best_matches.json"
    manual_path = output_dir / f"{prefix}_manual_confirmed_matches.json"
    recovered_path = output_dir / f"{prefix}_recovered_items.json"
    manual_recovered_path = (
        output_dir / f"{prefix}_manual_confirmed_recovered_items.json"
    )
    summary_path = output_dir / f"{prefix}_summary.json"

    summary = {
        "no_match_definition": "asr_text is null and asr_text_cn is null in items_final_fixed.json",
        "search_rule": (
            f"candidate_raw_id in current_raw_video_id +/- {args.radius} "
            f"and abs(interaction_duration - raw_duration) < {args.duration_threshold}"
        ),
        "interaction_duration_rule": "mode(duration) in interaction.csv for each source_pid",
        "manual_confirmations_parsed": len(manual_by_pid),
        "no_match_items": len(no_match_items),
        "no_match_items_with_interaction_duration": len(interaction_duration_by_pid),
        "skipped_no_duration": skipped_no_duration,
        "skipped_no_center_raw": skipped_no_center,
        "candidate_rows": len(all_candidates),
        "items_with_local_candidates": len(best_matches),
        "confidence_counts_for_best": dict(confidence_counts),
        "distance_counts_for_best": dict(sorted(distance_counts.items(), key=lambda kv: int(kv[0]))),
        "candidate_count_distribution_for_best": dict(
            sorted(candidate_count_distribution.items(), key=lambda kv: int(kv[0]))
        ),
        "outputs": {
            "all_candidates": str(all_path),
            "best_matches": str(best_path),
            "manual_confirmed_matches": str(manual_path),
            "recovered_items": str(recovered_path),
            "manual_confirmed_recovered_items": str(manual_recovered_path),
            "summary": str(summary_path),
        },
        "top_best_matches": best_matches[:50],
    }

    write_json(all_path, all_candidates)
    write_json(best_path, best_matches)
    write_json(
        manual_path,
        [row for row in best_matches if row.get("manual_confirmed")],
    )
    write_json(recovered_path, recovered_items)
    write_json(
        manual_recovered_path,
        [
            row
            for row in recovered_items
            if row.get("local_pass_manual_confirmed")
        ],
    )
    write_json(summary_path, summary)

    print(f"wrote {all_path}")
    print(f"wrote {best_path}")
    print(f"wrote {manual_path}")
    print(f"wrote {recovered_path}")
    print(f"wrote {manual_recovered_path}")
    print(f"wrote {summary_path}")
    print(f"no_match_items: {len(no_match_items)}")
    print(f"items_with_local_candidates: {len(best_matches)}")
    print(f"candidate_rows: {len(all_candidates)}")
    print(f"confidence_counts_for_best: {json.dumps(dict(confidence_counts), ensure_ascii=False, sort_keys=True)}")
    print(f"distance_counts_for_best: {json.dumps(dict(sorted(distance_counts.items(), key=lambda kv: int(kv[0]))), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
