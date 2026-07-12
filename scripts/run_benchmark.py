#!/usr/bin/env python3
# coding: utf-8
"""Plan or execute queue-style benchmark manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.benchmark import (
    build_benchmark_plan,
    build_benchmark_summary,
    execute_benchmark_plan,
    write_benchmark_ledger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark planner and executor for NexusRec.")
    parser.add_argument(
        "--spec",
        required=True,
        help="Queue-style benchmark spec containing an experiments list.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks",
        help="Directory where dry-run ledger artifacts are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand runs and write the ledger without executing training jobs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run completed benchmark jobs instead of skipping them.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate benchmark summary artifacts after execution completes.",
    )
    parser.add_argument(
        "--significance-baseline",
        type=str,
        help="Optional baseline model for paired significance reporting inside the summary.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric columns to include in benchmark summary and optional significance output.",
    )
    parser.add_argument(
        "--test",
        default=None,
        choices=["wilcoxon", "paired_t"],
        help="Optional override for the paired test used in benchmark significance reporting.",
    )
    parser.add_argument(
        "--significance-pair-field",
        default=None,
        choices=["seed", "comment"],
        help="Optional override for the pairing key used in benchmark significance reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_benchmark_plan(args.spec)
    if args.dry_run and args.summarize:
        raise ValueError("--summarize is a post-run summary step and cannot be combined with --dry-run.")
    if args.dry_run:
        outputs = write_benchmark_ledger(plan, args.output_dir)
        print(f"manifest_name: {plan['manifest_name']}")
        print(f"manifest_hash: {plan['manifest_hash']}")
        print(f"runs: {len(plan['runs'])}")
        print(f"output_dir: {outputs['output_dir']}")
        print(f"ledger_jsonl: {outputs['ledger_jsonl']}")
        print(f"ledger_csv: {outputs['ledger_csv']}")
        print(f"plan_json: {outputs['plan_json']}")
        return

    summary = execute_benchmark_plan(
        plan,
        args.output_dir,
        resume_enabled=not args.no_resume,
    )
    print(f"manifest_name: {summary['manifest_name']}")
    print(f"manifest_hash: {summary['manifest_hash']}")
    print(f"planned_runs: {summary['planned_runs']}")
    print(f"executed_runs: {summary['executed_runs']}")
    print(f"skipped_runs: {summary['skipped_runs']}")
    print(f"output_dir: {summary['output_dir']}")
    print(f"ledger_jsonl: {summary['ledger_jsonl']}")
    print(f"ledger_csv: {summary['ledger_csv']}")
    print(f"plan_json: {summary['plan_json']}")
    if args.summarize:
        report = build_benchmark_summary(
            plan,
            args.output_dir,
            significance_baseline=args.significance_baseline,
            significance_metrics=args.metrics,
            significance_test=args.test,
            significance_pair_field=args.significance_pair_field,
        )
        print(f"summary_runs_csv: {report['summary_runs_csv']}")
        print(f"summary_groups_csv: {report['summary_groups_csv']}")
        print(f"summary_markdown: {report['summary_markdown']}")
        if report["summary_significance_csv"] is not None:
            print(f"summary_significance_csv: {report['summary_significance_csv']}")


if __name__ == "__main__":
    main()
