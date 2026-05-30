#!/usr/bin/env -S uv run
"""Evaluate ranking models using dashboard feedback as ground truth.

Labels and relevance come entirely from dashboard_feedback.json:
  - up    -> positive label, relevant for metrics
  - down  -> negative label, excluded from relevance
  - neutral -> ordinal middle label, excluded from relevance

Two split modes:
  default  -> time-based by feedback.updated_at
  --seed N -> deterministic random permutation for CV replicability

Candidate pool = held-out test feedback stories + cached HN distractors.
Model is trained from scratch each fold using only train-feedback labels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from api.config import AppConfig
from api.feedback import FeedbackRecord, load_feedback
from api.feedback_single_model import (
    SingleModelLabeledStory,
    build_single_model_feedback_labels,
    train_single_model_from_embeddings,
)
from api.models import Story
from api.ordinal_model import DOWNVOTE_LABEL, UPVOTE_LABEL
from api.rerank import get_embeddings, rank_stories
from api.url_utils import normalize_url
from evaluate_quality import _compute_metrics, _print_metrics_report

FEEDBACK_STORE_PATH = Path(".cache/user_feedback/dashboard_feedback.json")
DEFAULT_K_METRICS = [10, 20, 30, 50]


@dataclass
class FeedbackEvalDataset:
    train_labels: list[SingleModelLabeledStory]
    train_embeddings: NDArray[np.float32]
    pos_embeddings: NDArray[np.float32]
    neg_embeddings: NDArray[np.float32]
    test_up_keys: set[str]
    test_down_keys: set[str]
    candidates: list[Story]
    candidate_keys: list[str]


def _story_key(story: Story) -> str:
    url = normalize_url(story.url) if story.url else None
    if url:
        return url
    return f"{story.source}:{story.id}"


def _time_split(
    records: dict[str, FeedbackRecord],
    holdout: float,
) -> tuple[dict[str, FeedbackRecord], dict[str, FeedbackRecord]]:
    sorted_items = sorted(records.items(), key=lambda item: item[1].updated_at)
    n = len(sorted_items)
    n_test = max(1, int(n * holdout))
    return dict(sorted_items[:-n_test]), dict(sorted_items[-n_test:])


def _random_fold(
    records: dict[str, FeedbackRecord],
    fold: int,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, FeedbackRecord], dict[str, FeedbackRecord]]:
    items = list(records.items())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))
    fold_size = max(1, len(items) // n_folds)
    test_start = fold * fold_size
    test_end = test_start + fold_size if fold < n_folds - 1 else len(items)
    test_idx = set(perm[test_start:test_end])
    train_items = [items[i] for i in range(len(items)) if i not in test_idx]
    test_items = [items[i] for i in test_idx]
    return dict(train_items), dict(test_items)


def _build_train_data(
    records: dict[str, FeedbackRecord],
) -> tuple[
    list[SingleModelLabeledStory],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    label_result = build_single_model_feedback_labels(records)
    labels = label_result.labels
    if not labels:
        return (
            labels,
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    stories = [lb.story for lb in labels]
    emb = get_embeddings([s.text_content for s in stories])

    pos_mask = np.array([lb.label == UPVOTE_LABEL for lb in labels])
    neg_mask = np.array([lb.label == DOWNVOTE_LABEL for lb in labels])
    dim = emb.shape[1] if emb.ndim == 2 else 0
    pos_emb = (
        emb[pos_mask] if np.any(pos_mask) else np.zeros((0, dim), dtype=np.float32)
    )
    neg_emb = (
        emb[neg_mask] if np.any(neg_mask) else np.zeros((0, dim), dtype=np.float32)
    )
    return labels, emb, pos_emb, neg_emb


def _count_by_action(records: dict[str, FeedbackRecord]) -> tuple[int, int, int]:
    u = d = n = 0
    for r in records.values():
        if r.action == "up":
            u += 1
        elif r.action == "down":
            d += 1
        else:
            n += 1
    return u, d, n


def _evaluator_training_config(config: AppConfig) -> AppConfig:
    return replace(
        config,
        single_model=replace(
            config.single_model, min_positive_labels=1, min_negative_labels=1
        ),
    )


async def _fetch_distractors(
    exclude_ids: set[int],
    exclude_urls: set[str],
    cache_only: bool,
    count: int = 300,
) -> list[Story]:
    import httpx
    from api.fetching import get_best_stories

    async with httpx.AsyncClient(timeout=30.0) as _:
        return await get_best_stories(
            count,
            exclude_ids=exclude_ids,
            exclude_urls=exclude_urls,
            config=AppConfig(days=30, no_rss=True),
            cache_only=cache_only,
            allow_stale=cache_only,
        )


async def _build_dataset(
    records: dict[str, FeedbackRecord],
    train_records: dict[str, FeedbackRecord],
    test_records: dict[str, FeedbackRecord],
    cache_only: bool,
) -> FeedbackEvalDataset | None:
    labels, train_emb, pos_emb, neg_emb = _build_train_data(train_records)
    if len(labels) < 2:
        return None

    test_up_keys = {r.key for r in test_records.values() if r.action == "up"}
    test_down_keys = {r.key for r in test_records.values() if r.action == "down"}

    candidate_pool: list[Story] = []
    seen_keys: set[str] = set()

    for r in test_records.values():
        s = r.to_story()
        k = r.key if r.key else _story_key(s)
        if k not in seen_keys:
            candidate_pool.append(s)
            seen_keys.add(k)

    exclude_ids: set[int] = set()
    exclude_urls: set[str] = set()
    for r in records.values():
        if r.source == "hn" and r.id > 0:
            exclude_ids.add(r.id)
        if r.url:
            norm = normalize_url(r.url)
            if norm:
                exclude_urls.add(norm)

    distractors = await _fetch_distractors(exclude_ids, exclude_urls, cache_only)

    for s in distractors:
        k = _story_key(s)
        if k not in seen_keys:
            candidate_pool.append(s)
            seen_keys.add(k)

    candidate_keys = [_story_key(s) for s in candidate_pool]
    up_in_cand = sum(1 for k in candidate_keys if k in test_up_keys)
    down_in_cand = sum(1 for k in candidate_keys if k in test_down_keys)
    print(
        f"  Candidates: {len(candidate_pool)} ({up_in_cand}/{len(test_up_keys)} test up, {down_in_cand}/{len(test_down_keys)} test down)"
    )

    if up_in_cand == 0 and test_up_keys:
        print("  WARNING: no test upvotes in candidate pool — metrics will be zero.")

    return FeedbackEvalDataset(
        train_labels=labels,
        train_embeddings=train_emb,
        pos_embeddings=pos_emb,
        neg_embeddings=neg_emb,
        test_up_keys=test_up_keys,
        test_down_keys=test_down_keys,
        candidates=candidate_pool,
        candidate_keys=candidate_keys,
    )


def _run_evaluate(
    ds: FeedbackEvalDataset, config: AppConfig, final_list_count: int | None
) -> dict[str, float]:
    from dataclasses import dataclass

    @dataclass
    class _Result:
        index: int

    training_config = _evaluator_training_config(config)
    has_neg = ds.neg_embeddings.shape[0] > 0 if ds.neg_embeddings.ndim == 2 else False
    mtype = training_config.single_model.model_type

    if mtype in ("pairwise_logistic", "lambdarank"):
        raise ValueError(
            f"model_type={mtype!r} requires experimental_models which was removed."
        )
    else:
        if has_neg:
            model, _ = train_single_model_from_embeddings(
                ds.train_labels,
                ds.train_embeddings,
                ds.pos_embeddings,
                ds.neg_embeddings,
                training_config,
                training_config.single_model,
            )
            results = rank_stories(
                ds.candidates,
                model,
                positive_embeddings=ds.pos_embeddings,
                negative_embeddings=ds.neg_embeddings,
                config=training_config,
            )
        else:
            results = rank_stories(
                ds.candidates,
                positive_embeddings=ds.pos_embeddings,
                negative_embeddings=None,
                config=training_config,
            )

    ranked_keys = [ds.candidate_keys[r.index] for r in results]

    if final_list_count is not None:
        from generate_html import select_ranked_results

        selected = select_ranked_results(
            results,
            ds.candidates,
            cluster_labels=None,
            cluster_names={},
            cand_cluster_map={},
            count=final_list_count,
        )
        ranked_keys = [ds.candidate_keys[r.index] for r in selected]
        seen: set[str] = set()
        deduped: list[str] = []
        for k in ranked_keys:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        ranked_keys = deduped

    relevant_ids = {hash(k) for k in ds.test_up_keys}
    downvote_ids = {hash(k) for k in ds.test_down_keys}
    ranked_ids = [hash(k) for k in ranked_keys]
    metrics = _compute_metrics(
        ranked_ids, relevant_ids, DEFAULT_K_METRICS, downvote_ids=downvote_ids
    )

    down_ranks = [i + 1 for i, k in enumerate(ranked_keys) if k in ds.test_down_keys]
    if down_ranks and ds.test_down_keys:
        metrics["downvote_median_rank"] = float(np.median(down_ranks))
        metrics["downvote_min_rank"] = float(min(down_ranks))
        metrics["downvote_in_top10"] = sum(1 for r in down_ranks if r <= 10) / max(
            len(ds.test_down_keys), 1
        )

    return metrics


def _average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    avg: dict[str, float] = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        avg[key] = float(np.mean(values))
        avg[f"{key}_std"] = float(np.std(values))
    return avg


def _build_cli_config(args: argparse.Namespace) -> AppConfig:
    from api.config import SemanticConfig

    config = AppConfig.load()
    overrides: dict = {}

    if args.model_type is not None:
        overrides["model_type"] = args.model_type
    if args.svm_kernel is not None:
        overrides["svm_kernel"] = args.svm_kernel
    if args.svm_c is not None:
        overrides["svm_c"] = args.svm_c
    if args.svm_gamma is not None:
        gamma: str | float
        try:
            gamma = float(args.svm_gamma)
        except ValueError:
            gamma = args.svm_gamma
        overrides["svm_gamma"] = gamma
    for rf_attr in (
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "rf_min_samples_split",
        "rf_max_features",
    ):
        val = getattr(args, rf_attr, None)
        if val is not None:
            overrides[rf_attr] = val
    if overrides:
        config = replace(config, single_model=replace(config.single_model, **overrides))

    if args.knn is not None:
        config = replace(config, semantic=SemanticConfig(knn_neighbors=args.knn))

    from evaluate_quality import apply_evaluator_overrides

    config = apply_evaluator_overrides(
        config, pure_semantic=args.pure_semantic, use_new_features=args.use_new_features
    )

    return config


async def _run_holdout_eval(
    config: AppConfig,
    records: dict[str, FeedbackRecord],
    train_records: dict[str, FeedbackRecord],
    test_records: dict[str, FeedbackRecord],
    cache_only: bool,
    count: int | None,
) -> dict[str, float] | None:
    ds = await _build_dataset(records, train_records, test_records, cache_only)
    if ds is None:
        return None
    return _run_evaluate(ds, config, count)


async def _run_grid_search(
    base_config: AppConfig,
    records: dict[str, FeedbackRecord],
    args: argparse.Namespace,
) -> int:
    from itertools import product

    train_records, test_records = _time_split(records, args.holdout)
    tu, td, _ = _count_by_action(train_records)
    su, sd, _ = _count_by_action(test_records)
    print(
        f"Train: {len(train_records)} ({tu}u, {td}d)  Test: {len(test_records)} ({su}u, {sd}d)"
    )

    count = args.count if args.final_list else None
    results: list[dict[str, object]] = []

    up_weights = args.grid_upvote_weights
    down_penalties = args.grid_downvote_penalties
    max_downvote_probs = args.grid_max_downvote_probs
    total = len(up_weights) * len(down_penalties) * len(max_downvote_probs)

    print(f"Grid search: {total} combinations")
    print(f"  upvote_weights={up_weights}")
    print(f"  downvote_penalties={down_penalties}")
    print(f"  max_downvote_probs={max_downvote_probs}")

    for i, (w_up, w_down, max_dp) in enumerate(
        product(up_weights, down_penalties, max_downvote_probs)
    ):
        config = replace(
            base_config,
            single_model=replace(
                base_config.single_model,
                utility_upvote_weight=w_up,
                utility_downvote_penalty=w_down,
                max_downvote_prob=max_dp,
            ),
        )
        metrics = await _run_holdout_eval(
            config, records, train_records, test_records, args.cache_only, count
        )
        if metrics is None:
            continue

        entry = {
            "w_up": w_up,
            "w_down": w_down,
            "max_dp": max_dp,
            "ndcg@30": metrics.get("ndcg@30", 0.0),
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "recall@30": metrics.get("recall@30", 0.0),
            "recall@10": metrics.get("recall@10", 0.0),
            "downvote_rate@10": metrics.get("downvote_rate@10", 0.0),
            "downvote_rate@30": metrics.get("downvote_rate@30", 0.0),
            "mrr": metrics.get("mrr", 0.0),
        }
        results.append(entry)
        print(
            f"  [{i + 1}/{total}] w_up={w_up:.1f} w_down={w_down:.1f} "
            f"dp={max_dp:.1f}  NDCG@30={entry['ndcg@30']:.3f} "
            f"DV@10={entry['downvote_rate@10']:.2%} DV@30={entry['downvote_rate@30']:.2%}"
        )

    if not results:
        print("No valid evaluations.")
        return 1

    best_by_ndcg = max(results, key=lambda r: r["ndcg@30"])
    best_by_dv = min(results, key=lambda r: r["downvote_rate@30"])
    best_combined = max(
        results,
        key=lambda r: r["ndcg@30"] - r["downvote_rate@30"],
    )

    print(f"\n{'=' * 70}")
    print(f"  Grid Search Results ({total} combinations)")
    print(f"{'=' * 70}")
    print(
        f"{'w_up':>4} {'w_down':>6} {'max_dp':>6} | {'NDCG@10':>8} {'NDCG@30':>8} {'Rec@10':>8} {'Rec@30':>8} {'DV@10':>8} {'DV@30':>8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['w_up']:>4.1f} {r['w_down']:>6.1f} {r['max_dp']:>6.1f} | "
            f"{r['ndcg@10']:>8.3f} {r['ndcg@30']:>8.3f} "
            f"{r['recall@10']:>8.1%} {r['recall@30']:>8.1%} "
            f"{r['downvote_rate@10']:>8.1%} {r['downvote_rate@30']:>8.1%}"
        )
    print(
        f"\nBest NDCG@30:   w_up={best_by_ndcg['w_up']:.1f} w_down={best_by_ndcg['w_down']:.1f} "
        f"dp={best_by_ndcg['max_dp']:.1f}  NDCG@30={best_by_ndcg['ndcg@30']:.3f}"
    )
    print(
        f"Lowest DV@30:   w_up={best_by_dv['w_up']:.1f} w_down={best_by_dv['w_down']:.1f} "
        f"dp={best_by_dv['max_dp']:.1f}  DV@30={best_by_dv['downvote_rate@30']:.1%}"
    )
    print(
        f"Best combined:  w_up={best_combined['w_up']:.1f} w_down={best_combined['w_down']:.1f} "
        f"dp={best_combined['max_dp']:.1f}  NDCG@30={best_combined['ndcg@30']:.3f} "
        f"DV@30={best_combined['downvote_rate@30']:.1%}"
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"  Results saved to {args.json_output}")

    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ranking models using dashboard feedback as ground truth."
    )
    parser.add_argument("--feedback-path", type=Path, default=FEEDBACK_STORE_PATH)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", action="store_false", dest="cache_only")

    parser.add_argument("--model-type", default=None)
    parser.add_argument("--svm-kernel", default=None)
    parser.add_argument("--svm-c", type=float, default=None)
    parser.add_argument("--svm-gamma", default=None)

    parser.add_argument("--rf-n-estimators", type=int, default=None)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=None)
    parser.add_argument("--rf-min-samples-split", type=int, default=None)
    parser.add_argument("--rf-max-features", default=None)
    parser.add_argument("--knn", type=int, default=None)

    parser.add_argument("--use-new-features", action="store_true")
    parser.add_argument("--pure-semantic", action="store_true")

    parser.add_argument("--final-list", action="store_true")
    parser.add_argument("--count", type=int, default=40)

    parser.add_argument("--seed", type=int, default=None, help="RNG seed for CV folds")
    parser.add_argument(
        "--cv", type=int, default=0, help="Number of CV folds (0 = time holdout)"
    )

    parser.add_argument("--json-output", type=Path, default=None)

    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Grid search over utility weights and downvote penalty",
    )
    parser.add_argument(
        "--grid-upvote-weights",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
    )
    parser.add_argument(
        "--grid-downvote-penalties",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
    )
    parser.add_argument(
        "--grid-max-downvote-probs",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 1.0],
    )

    args = parser.parse_args()
    config = _build_cli_config(args)

    records = load_feedback(args.feedback_path)
    if not records:
        print("No feedback records found.")
        return 1

    up, down, neutral = _count_by_action(records)
    print(f"Feedback: {len(records)} total ({up} up, {down} down, {neutral} neutral)")
    if up < 2:
        print("Need at least 2 upvotes in feedback.")
        return 1

    if args.grid_search:
        return await _run_grid_search(config, records, args)

    model_label = args.model_type or config.single_model.model_type
    label_str = f" {model_label}"
    if args.final_list:
        label_str += f" final@{args.count}"
    if args.pure_semantic:
        label_str += " psem"

    if args.cv > 1:
        all_metrics: list[dict[str, float]] = []
        for fold in range(args.cv):
            seed = args.seed or 0
            train_records, test_records = _random_fold(records, fold, args.cv, seed)
            tu, td, _ = _count_by_action(train_records)
            su, sd, _ = _count_by_action(test_records)
            print(
                f"\n[{label_str}] Fold {fold + 1}/{args.cv}: train={len(train_records)} ({tu}u,{td}d) test={len(test_records)} ({su}u,{sd}d)"
            )

            ds = await _build_dataset(
                records, train_records, test_records, args.cache_only
            )
            if ds is None:
                print("  Skipping fold — too few training labels.")
                continue
            metrics = _run_evaluate(ds, config, args.count if args.final_list else None)
            all_metrics.append(metrics)

        if not all_metrics:
            print("No valid folds.")
            return 1

        avg = _average_metrics(all_metrics)
        print(f"\n{'=' * 50}")
        print(f"  {label_str}  CV@{args.cv} seed={args.seed or 0}")
        print(f"{'=' * 50}")
        _print_metrics_report(
            avg, DEFAULT_K_METRICS, include_std=True, include_hit=True
        )

        if "downvote_median_rank" in avg:
            dr = avg.get("downvote_median_rank", 0)
            drs = avg.get("downvote_median_rank_std", 0)
            d10 = avg.get("downvote_in_top10", 0)
            d10s = avg.get("downvote_in_top10_std", 0)
            print(
                f"\n  Downvote leakage: median_rank={dr:.0f}±{drs:.0f}, in_top10={d10:.1%}±{d10s:.1%}"
            )
    else:
        train_records, test_records = _time_split(records, args.holdout)
        tu, td, _ = _count_by_action(train_records)
        su, sd, _ = _count_by_action(test_records)
        print(
            f"Train: {len(train_records)} ({tu}u, {td}d)  Test: {len(test_records)} ({su}u, {sd}d)"
        )

        ds = await _build_dataset(records, train_records, test_records, args.cache_only)
        if ds is None:
            print("Too few training labels with content.")
            return 1
        metrics = _run_evaluate(ds, config, args.count if args.final_list else None)

        print()
        _print_metrics_report(
            metrics, DEFAULT_K_METRICS, include_std=False, include_hit=True
        )

        if "downvote_median_rank" in metrics:
            print(
                f"\n  Downvote leakage: median_rank={metrics['downvote_median_rank']:.0f}, in_top10={metrics.get('downvote_in_top10', 0):.1%}"
            )

        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
