#!/usr/bin/env -S uv run
# ruff: noqa: E402
"""Compare current runtime ranking against a single feedback-trained CE-free model."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import AppConfig, LearnedRankerConfig
from api.feedback import FEEDBACK_STORE_PATH, load_feedback
from api.feedback_single_model import (
    build_single_model_feedback_labels,
    rank_stories_with_single_model,
    score_feedback_labels_oof,
    score_metrics_for_labels,
    train_single_model,
)
from api.learned_ranker import (
    _predict_ordinal_outputs,
    build_features,
    build_labels_from_feedback,
    evaluate_labeled_score_sources,
    train_model,
)
from api.rerank import get_embeddings
from evaluate_quality import (
    DEFAULT_K_METRICS,
    RankingEvaluator,
    _average_metrics,
    _compute_metrics,
    _finalize_ranked_ids,
    _print_metric_deltas,
    _print_metrics_report,
)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _load_feedback_eval_subset(path: Path):
    records = load_feedback(path)
    eligible = {key: record for key, record in records.items() if record.to_rank_result() is not None}
    if not eligible:
        raise SystemExit("No labeled dashboard feedback with stored rank diagnostics found.")

    learned_labels = build_labels_from_feedback(eligible)
    single_label_result = build_single_model_feedback_labels(eligible)
    single_labels = single_label_result.labels
    if len(learned_labels) != len(single_labels):
        raise SystemExit("Feedback label builders produced different row counts.")
    return learned_labels, single_labels, single_label_result.skipped_count


def _feedback_training_config(config: AppConfig) -> LearnedRankerConfig:
    return replace(
        config.learned_ranker,
        min_positive_labels=2,
        min_negative_labels=2,
    )


def _current_learned_oof_scores(
    labels,
    training_config: LearnedRankerConfig,
    *,
    max_folds: int,
) -> list[float]:
    positive_labels = sum(item.label == 2 for item in labels)
    negative_labels = sum(item.label == 0 for item in labels)
    fold_count = min(max_folds, positive_labels, negative_labels)
    if fold_count < 2:
        raise SystemExit("Need at least 2 upvote and 2 downvote labels for feedback OOF.")

    from sklearn.model_selection import StratifiedKFold

    y_binary = [item.legacy_binary_label for item in labels]
    learned_scores = [0.0 for _ in labels]
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=0)

    label_array = np.asarray(y_binary, dtype=np.int64)
    for train_indices, test_indices in splitter.split(np.zeros(len(y_binary)), label_array):
        fold_labels = [labels[int(i)] for i in train_indices]
        model = train_model(fold_labels, training_config)
        rows = [
            build_features(
                labels[int(i)].story,
                labels[int(i)].rank_result,
                now=labels[int(i)].feedback_updated_at or None,
                source_feature_weight=training_config.source_feature_weight,
            )
            for i in test_indices
        ]
        utility, _, _, _ = _predict_ordinal_outputs(
            model,
            np.asarray(rows, dtype=np.float32),
        )
        for offset, label_index in enumerate(test_indices):
            learned_scores[int(label_index)] = float(utility[offset])
    return learned_scores


def _candidate_metrics_for_single_model(
    evaluator: RankingEvaluator,
    config: AppConfig,
    model,
    *,
    k_metrics: list[int],
    final_list_count: int | None,
    cv: int,
    seed: int,
) -> dict[str, float]:
    if evaluator.dataset is None:
        raise ValueError("Dataset not loaded")
    dataset = evaluator.dataset

    if cv <= 0:
        results = rank_stories_with_single_model(
            dataset.candidates,
            dataset.train_stories,
            dataset.neg_stories,
            config,
            model,
        )
        ranked_ids = (
            _finalize_ranked_ids(results, dataset.candidates, final_list_count)
            if final_list_count is not None
            else [dataset.candidates[result.index].id for result in results]
        )
        return _compute_metrics(ranked_ids, dataset.test_ids, k_metrics)

    all_stories = dataset.train_stories + dataset.test_stories
    n = len(all_stories)
    indices = np.random.default_rng(seed).permutation(n)
    fold_size = n // cv
    all_metrics: list[dict[str, float]] = []

    for fold in range(cv):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < cv - 1 else n
        test_idx = set(indices[test_start:test_end])
        train_idx = [i for i in range(n) if i not in test_idx]
        train_stories = [all_stories[i] for i in train_idx]
        test_ids = {all_stories[i].id for i in test_idx}

        candidate_ids = {candidate.id for candidate in dataset.candidates}
        fold_candidates = list(dataset.candidates)
        for i in test_idx:
            if all_stories[i].id not in candidate_ids:
                fold_candidates.append(all_stories[i])

        results = rank_stories_with_single_model(
            fold_candidates,
            train_stories,
            dataset.neg_stories,
            config,
            model,
        )
        ranked_ids = (
            _finalize_ranked_ids(results, fold_candidates, final_list_count)
            if final_list_count is not None
            else [fold_candidates[result.index].id for result in results]
        )
        fold_metrics = _compute_metrics(ranked_ids, test_ids, k_metrics)
        all_metrics.append(fold_metrics)

        print(f"\nSingle-model fold {fold + 1}/{cv}:")
        _print_metrics_report(
            fold_metrics,
            k_metrics,
            include_std=False,
            include_hit=True,
            indent="  ",
        )

    return _average_metrics(all_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a single dashboard-feedback-trained CE-free model against the current runtime."
        )
    )
    parser.add_argument("username", help="HN username")
    parser.add_argument("--path", default=FEEDBACK_STORE_PATH, type=Path, help="Feedback store path")
    parser.add_argument("--holdout", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--candidates", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True, help="Enable hidden-story negatives in the holdout dataset")
    parser.add_argument("--cache-only", action="store_true", help="Use cached story data only")
    parser.add_argument("--age-matched", action="store_true", help="Fetch holdout candidates relative to test-story time")
    parser.add_argument("--limit-pos", type=int, default=200, help="Max positive stories to load")
    parser.add_argument("--limit-neg", type=int, default=100, help="Max negative stories to load")
    parser.add_argument("--cv", type=int, default=0, help="Number of holdout CV folds (0 = single split)")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic CV seed")
    parser.add_argument("--final-list", action="store_true", help="Evaluate final displayed list instead of raw ranked order")
    parser.add_argument("--count", type=int, default=40, help="Displayed story count when --final-list is enabled")
    parser.add_argument("--max-feedback-folds", type=int, default=5, help="Cross-validation fold cap for feedback OOF metrics")
    return parser


async def main() -> int:
    args = build_parser().parse_args()

    config = AppConfig.load(
        username=args.username,
        count=args.count,
        candidates=args.candidates,
    )
    evaluator = RankingEvaluator(args.username)
    success = await evaluator.load_data(
        holdout=args.holdout,
        limit_pos=args.limit_pos,
        limit_neg=args.limit_neg,
        candidate_count=args.candidates,
        use_classifier=args.classifier,
        cache_only=args.cache_only,
        allow_stale=args.cache_only,
        age_matched=args.age_matched,
    )
    if not success or evaluator.dataset is None:
        return 1

    history_positive_stories = evaluator.dataset.train_stories + evaluator.dataset.test_stories
    history_negative_stories = evaluator.dataset.neg_stories
    positive_embeddings = (
        None
        if not history_positive_stories
        else get_embeddings([story.text_content for story in history_positive_stories])
    )
    negative_embeddings = (
        None
        if not history_negative_stories
        else get_embeddings([story.text_content for story in history_negative_stories])
    )

    learned_labels, single_labels, skipped_feedback_rows = _load_feedback_eval_subset(args.path)
    training_config = _feedback_training_config(config)

    print("Feedback-label evaluation")
    print(
        f"Rows: {len(single_labels)} comparable | skipped {skipped_feedback_rows} missing-text rows"
    )
    current_learned_scores = _current_learned_oof_scores(
        learned_labels,
        training_config,
        max_folds=args.max_feedback_folds,
    )
    single_scores, feature_batch = score_feedback_labels_oof(
        single_labels,
        positive_embeddings,
        negative_embeddings,
        config,
        training_config,
        max_folds=args.max_feedback_folds,
    )
    feedback_comparisons = evaluate_labeled_score_sources(
        learned_labels,
        {
            "current_learned": current_learned_scores,
            "current_hybrid": [float(item.rank_result.hybrid_score) for item in learned_labels],
            "single_ce_free": single_scores,
        },
    )
    print(
        f"Single-model features: {feature_batch.rows.shape[1]} total = "
        f"{feature_batch.rows.shape[1] - feature_batch.derived_feature_dim - feature_batch.metadata_feature_dim} embeddings + "
        f"{feature_batch.derived_feature_dim} derived + {feature_batch.metadata_feature_dim} metadata"
    )
    print("")
    print(
        f"{'score':<18} {'pairwise':<10} {'roc_auc':<10} {'p@5':<8} {'p@10':<8} "
        f"{'neutral@10':<12} {'down@10':<10}"
    )
    print("-" * 88)
    for item in feedback_comparisons:
        metrics = item.metrics
        print(
            f"{item.label:<18} "
            f"{_format_metric(metrics.pairwise_accuracy):<10} "
            f"{_format_metric(metrics.roc_auc):<10} "
            f"{_format_metric(metrics.precision_at_5):<8} "
            f"{_format_metric(metrics.precision_at_10):<8} "
            f"{_format_metric(metrics.neutral_rate_at_10):<12} "
            f"{_format_metric(metrics.downvote_rate_at_10):<10}"
        )

    model, _ = train_single_model(
        single_labels,
        positive_embeddings,
        negative_embeddings,
        config,
        training_config,
    )

    k_metrics = (
        [k for k in DEFAULT_K_METRICS if k <= args.count] or [args.count]
        if args.final_list
        else DEFAULT_K_METRICS
    )

    print("")
    print(
        "Holdout evaluation "
        f"({'final list' if args.final_list else 'raw ranking'})"
    )
    baseline_label = "Current runtime"
    if args.cv > 0:
        print(f"Running {args.cv}-fold comparison...")
        baseline_metrics = evaluator.evaluate_cv(
            n_folds=args.cv,
            config=config,
            k_metrics=k_metrics,
            report_each=True,
            parallel=False,
            final_list_count=args.count if args.final_list else None,
            seed=args.seed,
        )
    else:
        baseline_metrics = evaluator.evaluate(
            config=config,
            k_metrics=k_metrics,
            final_list_count=args.count if args.final_list else None,
        )
    print(f"\n{baseline_label}")
    _print_metrics_report(
        baseline_metrics,
        k_metrics,
        include_std=args.cv > 0,
        include_hit=args.cv == 0,
    )

    single_metrics = _candidate_metrics_for_single_model(
        evaluator,
        config,
        model,
        k_metrics=k_metrics,
        final_list_count=args.count if args.final_list else None,
        cv=args.cv,
        seed=args.seed,
    )
    print("\nSingle feedback-trained CE-free model")
    _print_metrics_report(
        single_metrics,
        k_metrics,
        include_std=args.cv > 0,
        include_hit=args.cv == 0,
    )

    print("")
    _print_metric_deltas(baseline_metrics, single_metrics, k_metrics)

    single_feedback_metrics = score_metrics_for_labels(single_labels, single_scores)
    current_feedback_metrics = next(
        item.metrics for item in feedback_comparisons if item.label == "current_learned"
    )
    print("\nFeedback delta vs current_learned")
    print(
        f"pairwise {single_feedback_metrics.pairwise_accuracy - current_feedback_metrics.pairwise_accuracy:+.4f} | "
        f"roc_auc {(single_feedback_metrics.roc_auc or 0.0) - (current_feedback_metrics.roc_auc or 0.0):+.4f} | "
        f"p@10 {single_feedback_metrics.precision_at_10 - current_feedback_metrics.precision_at_10:+.4f} | "
        f"neutral@10 {single_feedback_metrics.neutral_rate_at_10 - current_feedback_metrics.neutral_rate_at_10:+.4f} | "
        f"down@10 {single_feedback_metrics.downvote_rate_at_10 - current_feedback_metrics.downvote_rate_at_10:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
