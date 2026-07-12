"""CLI for strict paired significance testing on repeated experiment summaries."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.utils.significance import compare_paired_results, format_summary_table, load_result_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict paired significance tests on repeated NexusRec result CSVs. "
            "Each input CSV must contain exactly one summary row."
        )
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="Baseline result inputs (files, directories, or glob patterns).",
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        help="Candidate result inputs (files, directories, or glob patterns).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Metric columns to test. Defaults to metric columns inferred from baseline "
            "records; candidate records must contain the same columns."
        ),
    )
    parser.add_argument(
        "--pair-field",
        default="comment",
        choices=["comment", "stem"],
        help="Strict pairing key. 'comment' is recommended for repeated seed runs.",
    )
    parser.add_argument(
        "--test",
        default="wilcoxon",
        choices=["wilcoxon", "paired_t"],
        help="Paired statistical test to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_records = load_result_records(args.baseline)
    candidate_records = load_result_records(args.candidate)
    metadata, summaries = compare_paired_results(
        baseline_records=baseline_records,
        candidate_records=candidate_records,
        metrics=args.metrics,
        pair_field=args.pair_field,
        test_name=args.test,
    )
    print(format_summary_table(metadata, summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
