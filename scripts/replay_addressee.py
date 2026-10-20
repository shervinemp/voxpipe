"""Review shadow-mode attention decisions and sweep thresholds.

Usage:
    uv run python scripts/replay_addressee.py
    uv run python scripts/replay_addressee.py --decisions data/attention_decisions.jsonl
"""

import argparse
from typing import Optional

from voxpipe.pipeline.attention.replay import (
    best_under_false_positive_rate,
    load_records,
    paired_labels,
    summarize,
    sweep,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze addressee gate decisions and corrections.")
    parser.add_argument(
        "--decisions",
        default="data/attention_decisions.jsonl",
        help="JSONL file written by attention.decision_log_path.",
    )
    parser.add_argument(
        "--max-fp-rate",
        type=float,
        default=0.01,
        help="Target false-positive rate for the suggested threshold (default: 0.01).",
    )
    parser.add_argument("--top", type=int, default=8, help="Rows to print in the sweep table.")
    args = parser.parse_args(argv)

    records = load_records(args.decisions)
    summary = summarize(records)
    print(f"Decisions: {summary['decisions']}  answers: {summary['answers']}  corrections: {summary['corrections']}")
    if summary["reasons"]:
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(summary["reasons"].items()))
        print(f"Reasons:   {reasons}")

    labels = paired_labels(records)
    print(f"Paired labeled decisions: {len(labels)}")
    if not labels:
        print("No corrections paired with decisions yet; keep shadow mode running and use 'not you' / 'that was me'.")
        return 0

    print(f"\n{'thr':>5} {'acc':>6} {'prec':>6} {'recall':>6} {'fp-rate':>7} {'fn-rate':>7}  tp/fp/tn/fn")
    rows = [m for m in sweep(labels) if m.false_positive_rate <= args.max_fp_rate] or sweep(labels)
    for metrics in rows[: max(1, args.top)]:
        print(
            f"{metrics.threshold:5.2f} {metrics.accuracy:6.2f} {metrics.precision:6.2f} "
            f"{metrics.recall:6.2f} {metrics.false_positive_rate:7.3f} {metrics.false_negative_rate:7.3f}  "
            f"{metrics.true_positive}/{metrics.false_positive}/{metrics.true_negative}/{metrics.false_negative}"
        )

    best = best_under_false_positive_rate(labels, args.max_fp_rate)
    if best is None:
        print(f"\nNo threshold meets an FP rate <= {args.max_fp_rate:.3f}.")
    else:
        print(
            f"\nSuggested threshold: {best.threshold:.2f} "
            f"(accuracy {best.accuracy:.2f}, recall {best.recall:.2f}, "
            f"FP rate {best.false_positive_rate:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
