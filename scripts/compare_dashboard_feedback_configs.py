#!/usr/bin/env -S uv run
"""Compare two configs against stored dashboard feedback labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from api.config import AppConfig
from api.feedback import load_feedback
from api.learned_ranker import (
    build_labels_from_feedback,
    compare_dashboard_feedback_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two hn_rerank TOML configs on temporal dashboard feedback."
    )
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument(
        "--feedback-path",
        type=Path,
        default=None,
        help="Optional path to dashboard feedback JSON (defaults to main store).",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--min-holdout-count", type=int, default=20)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--score-tolerance", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _print_metric_set(label: str, metrics: object, score: float) -> None:
    metric_dict = asdict(metrics)
    print(f"\n{label}:")
    print(f"  weighted_score: {score:.4f}")
    print(f"  pairwise_accuracy: {metric_dict['pairwise_accuracy']:.4f}")
    print(f"  precision@10: {metric_dict['precision_at_10']:.4f}")
    print(f"  precision@5: {metric_dict['precision_at_5']:.4f}")
    print(f"  neutral_rate@10: {metric_dict['neutral_rate_at_10']:.4f}")
    print(f"  downvote_rate@10: {metric_dict['downvote_rate_at_10']:.4f}")
    roc_auc = metric_dict["roc_auc"]
    print(f"  roc_auc: {'n/a' if roc_auc is None else f'{roc_auc:.4f}'}")
    print(f"  top_sources: {json.dumps(metric_dict['top_sources'], sort_keys=True)}")


def main() -> None:
    args = parse_args()
    baseline = AppConfig.load(args.baseline_config)
    candidate = AppConfig.load(args.candidate_config)
    records = load_feedback(args.feedback_path) if args.feedback_path else load_feedback()
    labels = build_labels_from_feedback(records)
    result = compare_dashboard_feedback_configs(
        labels,
        baseline,
        candidate,
        holdout_fraction=args.holdout_fraction,
        min_holdout_count=args.min_holdout_count,
        min_class_count=args.min_class_count,
        score_tolerance=args.score_tolerance,
    )

    print("Dashboard feedback temporal holdout:")
    print(
        f"  labels={result.summary.label_count}, "
        f"usable={result.summary.usable_label_count}, "
        f"skipped_missing_story_metadata={result.summary.skipped_missing_story_metadata}"
    )
    print(
        f"  train={result.summary.train_label_count} "
        f"(up={result.summary.train_positive_labels}, "
        f"neutral={result.summary.train_neutral_labels}, "
        f"down={result.summary.train_negative_labels})"
    )
    print(
        f"  holdout={result.summary.holdout_label_count} "
        f"(up={result.summary.holdout_positive_labels}, "
        f"neutral={result.summary.holdout_neutral_labels}, "
        f"down={result.summary.holdout_negative_labels})"
    )
    print(f"  holdout_start_timestamp={result.summary.holdout_start_timestamp:.0f}")

    _print_metric_set("Baseline", result.incumbent, result.incumbent_score)
    _print_metric_set("Candidate", result.candidate, result.candidate_score)
    print(
        "\nDecision: "
        f"{'PASS' if result.passed else 'FAIL'} "
        f"(score_delta={result.score_delta:.4f})"
    )
    if result.primary_failures:
        print(f"  primary regressions: {', '.join(result.primary_failures)}")
    if result.guard_failures:
        print(f"  guard regressions: {', '.join(result.guard_failures)}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
        print(f"\nWrote comparison JSON: {args.json_out}")

    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
