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
import contextlib

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

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
    test_neutral_keys: set[str]
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
    count: int,
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
    candidate_count: int = 300,
) -> FeedbackEvalDataset | None:
    labels, train_emb, pos_emb, neg_emb = _build_train_data(train_records)
    if len(labels) < 2:
        return None

    test_up_keys = {r.key for r in test_records.values() if r.action == "up"}
    test_down_keys = {r.key for r in test_records.values() if r.action == "down"}
    test_neutral_keys = {r.key for r in test_records.values() if r.action == "neutral"}

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

    distractors = await _fetch_distractors(
        exclude_ids, exclude_urls, cache_only, candidate_count
    )

    for s in distractors:
        k = _story_key(s)
        if k not in seen_keys:
            candidate_pool.append(s)
            seen_keys.add(k)

    candidate_keys = [_story_key(s) for s in candidate_pool]
    up_in_cand = sum(1 for k in candidate_keys if k in test_up_keys)
    down_in_cand = sum(1 for k in candidate_keys if k in test_down_keys)
    neutral_in_cand = sum(1 for k in candidate_keys if k in test_neutral_keys)
    print(
        f"  Candidates: {len(candidate_pool)} ({up_in_cand}/{len(test_up_keys)} up, {down_in_cand}/{len(test_down_keys)} down, {neutral_in_cand}/{len(test_neutral_keys)} neutral)"
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
        test_neutral_keys=test_neutral_keys,
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
    ranked_ids = [hash(k) for k in ranked_keys]
    metrics = _compute_metrics(ranked_ids, relevant_ids, DEFAULT_K_METRICS)

    down_ranks = [i + 1 for i, k in enumerate(ranked_keys) if k in ds.test_down_keys]
    if down_ranks and ds.test_down_keys:
        metrics["downvote_median_rank"] = float(np.median(down_ranks))
        metrics["downvote_min_rank"] = float(min(down_ranks))
        metrics["downvote_in_top10"] = sum(1 for r in down_ranks if r <= 10) / max(
            len(ds.test_down_keys), 1
        )

    neutral_ranks = [
        i + 1 for i, k in enumerate(ranked_keys) if k in ds.test_neutral_keys
    ]
    if neutral_ranks and ds.test_neutral_keys:
        metrics["neutral_median_rank"] = float(np.median(neutral_ranks))
        metrics["neutral_min_rank"] = float(min(neutral_ranks))
        metrics["neutral_in_top10"] = sum(1 for r in neutral_ranks if r <= 10) / max(
            len(ds.test_neutral_keys), 1
        )

    nonpositive_keys = ds.test_down_keys | ds.test_neutral_keys
    nonpositive_ranks = [
        i + 1 for i, k in enumerate(ranked_keys) if k in nonpositive_keys
    ]
    if nonpositive_ranks and nonpositive_keys:
        metrics["nonpositive_median_rank"] = float(np.median(nonpositive_ranks))
        metrics["nonpositive_min_rank"] = float(min(nonpositive_ranks))
        metrics["nonpositive_in_top10"] = sum(
            1 for r in nonpositive_ranks if r <= 10
        ) / max(len(nonpositive_keys), 1)

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
        config,
        pure_semantic=args.pure_semantic,
        use_new_features=args.use_new_features,
        no_raw_embedding_features=args.no_raw_embedding_features,
    )

    return config


@contextlib.contextmanager
def _override_rerank(
    rerank_module: Any,
    model_dir: str | None,
    embedding_max_tokens: int | None,
) -> Iterator[None]:
    if model_dir is None and embedding_max_tokens is None:
        yield
        return

    orig_model = getattr(rerank_module, "_model", None)
    orig_max_tokens = rerank_module.TEXT_CONTENT_MAX_TOKENS
    try:
        if model_dir is not None:
            model = rerank_module.ONNXEmbeddingModel(model_dir=model_dir)
            rerank_module._model = model
        if embedding_max_tokens is not None:
            rerank_module.TEXT_CONTENT_MAX_TOKENS = embedding_max_tokens
        yield
    finally:
        rerank_module._model = orig_model
        rerank_module.TEXT_CONTENT_MAX_TOKENS = orig_max_tokens


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ranking models using dashboard feedback as ground truth."
    )
    parser.add_argument("--feedback-path", type=Path, default=FEEDBACK_STORE_PATH)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", action="store_false", dest="cache_only")
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=300,
        help="Number of distractor candidates from HN cache",
    )

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
    parser.add_argument("--no-raw-embedding-features", action="store_true")

    parser.add_argument("--final-list", action="store_true")
    parser.add_argument("--count", type=int, default=40)

    parser.add_argument("--seed", type=int, default=None, help="RNG seed for CV folds")
    parser.add_argument(
        "--cv", type=int, default=0, help="Number of CV folds (0 = time holdout)"
    )

    parser.add_argument("--json-output", type=Path, default=None)

    parser.add_argument(
        "--model-dir", type=str, default=None, help="Override embedding model directory"
    )
    parser.add_argument(
        "--embedding-max-tokens",
        type=int,
        default=None,
        help="Override TEXT_CONTENT_MAX_TOKENS",
    )

    args = parser.parse_args()
    config = _build_cli_config(args)

    from api import rerank

    with _override_rerank(rerank, args.model_dir, args.embedding_max_tokens):
        records = load_feedback(args.feedback_path)
        if not records:
            print("No feedback records found.")
            return 1

        up, down, neutral = _count_by_action(records)
        print(
            f"Feedback: {len(records)} total ({up} up, {down} down, {neutral} neutral)"
        )
        if up < 2:
            print("Need at least 2 upvotes in feedback.")
            return 1

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
                tu, td, tn = _count_by_action(train_records)
                su, sd, sn = _count_by_action(test_records)
                print(
                    f"\n[{label_str}] Fold {fold + 1}/{args.cv}: train={len(train_records)} ({tu}u,{td}d,{tn}n) test={len(test_records)} ({su}u,{sd}d,{sn}n)"
                )

                ds = await _build_dataset(
                    records,
                    train_records,
                    test_records,
                    args.cache_only,
                    candidate_count=args.candidate_count,
                )
                if ds is None:
                    print("  Skipping fold — too few training labels.")
                    continue
                metrics = _run_evaluate(
                    ds, config, args.count if args.final_list else None
                )
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
                    f"\n  Downvote leakage:   median_rank={dr:.0f}±{drs:.0f}, in_top10={d10:.1%}±{d10s:.1%}"
                )

            if "neutral_median_rank" in avg:
                nr = avg.get("neutral_median_rank", 0)
                nrs = avg.get("neutral_median_rank_std", 0)
                n10 = avg.get("neutral_in_top10", 0)
                n10s = avg.get("neutral_in_top10_std", 0)
                print(
                    f"  Neutral leakage:    median_rank={nr:.0f}±{nrs:.0f}, in_top10={n10:.1%}±{n10s:.1%}"
                )

            if "nonpositive_median_rank" in avg:
                nr = avg.get("nonpositive_median_rank", 0)
                nrs = avg.get("nonpositive_median_rank_std", 0)
                n10 = avg.get("nonpositive_in_top10", 0)
                n10s = avg.get("nonpositive_in_top10_std", 0)
                print(
                    f"  Nonpositive leakage: median_rank={nr:.0f}±{nrs:.0f}, in_top10={n10:.1%}±{n10s:.1%}"
                )
        else:
            train_records, test_records = _time_split(records, args.holdout)
            tu, td, tn = _count_by_action(train_records)
            su, sd, sn = _count_by_action(test_records)
            print(
                f"Train: {len(train_records)} ({tu}u, {td}d, {tn}n)  Test: {len(test_records)} ({su}u, {sd}d, {sn}n)"
            )

            ds = await _build_dataset(
                records,
                train_records,
                test_records,
                args.cache_only,
                candidate_count=args.candidate_count,
            )
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
                args.json_output.write_text(
                    json.dumps(metrics, indent=2, sort_keys=True)
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
