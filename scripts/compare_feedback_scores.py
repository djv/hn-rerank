#!/usr/bin/env -S uv run
# ruff: noqa: E402
"""Compare offline score sources against stored dashboard feedback labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import LearnedRankerConfig
from api.feedback import FEEDBACK_STORE_PATH, load_feedback
from api.learned_ranker import (
    _predict_ordinal_outputs,
    build_features,
    build_labels_from_feedback,
    evaluate_labeled_score_sources,
    train_model,
)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "n/a"
    return f"{value - baseline:+.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare learned-ranker, hybrid, and semantic scores on stored dashboard feedback."
        )
    )
    parser.add_argument(
        "--path",
        default=FEEDBACK_STORE_PATH,
        type=Path,
        help="Feedback store path",
    )
    parser.add_argument("--max-folds", type=int, default=5, help="Cross-validation fold cap")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    records = load_feedback(path=args.path)
    labels = build_labels_from_feedback(records)
    if not labels:
        raise SystemExit("No labeled dashboard feedback with rank diagnostics found.")

    config = LearnedRankerConfig(
        min_positive_labels=2,
        min_negative_labels=2,
    )

    positive_labels = sum(item.label == 2 for item in labels)
    negative_labels = sum(item.label == 0 for item in labels)
    fold_count = min(args.max_folds, positive_labels, negative_labels)
    if fold_count < 2:
        raise SystemExit(
            "Need at least 2 upvote and 2 downvote labels to compare score sources."
        )

    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    y_binary = [item.legacy_binary_label for item in labels]
    learned_scores = [0.0 for _ in labels]
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=0)

    label_array = np.asarray(y_binary, dtype=np.int64)
    for train_indices, test_indices in splitter.split(np.zeros(len(y_binary)), label_array):
        fold_labels = [labels[int(i)] for i in train_indices]
        model = train_model(fold_labels, config)
        rows = [
            build_features(
                labels[int(i)].story,
                labels[int(i)].rank_result,
                now=labels[int(i)].feedback_updated_at or None,
                source_feature_weight=config.source_feature_weight,
            )
            for i in test_indices
        ]
        x_score = np.asarray(rows, dtype=np.float32)
        utility, _, _, _ = _predict_ordinal_outputs(model, x_score)
        for offset, label_index in enumerate(test_indices):
            learned_scores[int(label_index)] = float(utility[offset])

    comparisons = evaluate_labeled_score_sources(
        labels,
        {
            "learned_score": learned_scores,
            "hybrid_score": [float(item.rank_result.hybrid_score) for item in labels],
            "semantic_score": [float(item.rank_result.semantic_score) for item in labels],
        },
    )

    baseline = next(item.metrics for item in comparisons if item.label == "learned_score")

    print("Feedback score comparison")
    print(
        f"Labels: {len(labels)} total | up {positive_labels} | neutral "
        f"{sum(item.label == 1 for item in labels)} | down {negative_labels} | folds {fold_count}"
    )
    print("")
    print(
        f"{'score':<16} {'pairwise':<10} {'roc_auc':<10} {'p@5':<8} {'p@10':<8} "
        f"{'neutral@10':<12} {'down@10':<10}"
    )
    print("-" * 84)
    for item in comparisons:
        metrics = item.metrics
        print(
            f"{item.label:<16} "
            f"{_format_metric(metrics.pairwise_accuracy):<10} "
            f"{_format_metric(metrics.roc_auc):<10} "
            f"{_format_metric(metrics.precision_at_5):<8} "
            f"{_format_metric(metrics.precision_at_10):<8} "
            f"{_format_metric(metrics.neutral_rate_at_10):<12} "
            f"{_format_metric(metrics.downvote_rate_at_10):<10}"
        )

    print("\nDelta vs learned_score")
    print(
        f"{'score':<16} {'pairwise':<10} {'roc_auc':<10} {'p@5':<8} {'p@10':<8} "
        f"{'neutral@10':<12} {'down@10':<10}"
    )
    print("-" * 84)
    for item in comparisons:
        metrics = item.metrics
        print(
            f"{item.label:<16} "
            f"{_format_delta(metrics.pairwise_accuracy, baseline.pairwise_accuracy):<10} "
            f"{_format_delta(metrics.roc_auc, baseline.roc_auc):<10} "
            f"{_format_delta(metrics.precision_at_5, baseline.precision_at_5):<8} "
            f"{_format_delta(metrics.precision_at_10, baseline.precision_at_10):<8} "
            f"{_format_delta(metrics.neutral_rate_at_10, baseline.neutral_rate_at_10):<12} "
            f"{_format_delta(metrics.downvote_rate_at_10, baseline.downvote_rate_at_10):<10}"
        )

    semantic_metrics = next(item.metrics for item in comparisons if item.label == "semantic_score")
    semantic_delta = semantic_metrics.pairwise_accuracy - baseline.pairwise_accuracy
    if semantic_delta >= -0.01:
        summary = "effectively unchanged"
    elif semantic_delta >= -0.05:
        summary = "modestly worse"
    else:
        summary = "clearly worse"
    print(f"\nFirst-stage-only semantic_score verdict: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
