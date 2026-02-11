#!/usr/bin/env -S uv run
"""Evaluate ranking quality using holdout validation on user's upvote history."""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import httpx
import numpy as np
from numpy.typing import NDArray

def _ensure_joblib_settings() -> None:
    # Disable joblib multiprocessing in this environment to avoid SemLock
    # permission warnings; joblib falls back to serial either way.
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    tmp = os.environ.get("JOBLIB_TEMP_FOLDER") or os.environ.get("LOKY_TEMP_FOLDER")
    if not tmp:
        tmp = str(Path(__file__).resolve().parent / ".cache" / "joblib")
        os.environ["JOBLIB_TEMP_FOLDER"] = tmp
        os.environ["LOKY_TEMP_FOLDER"] = tmp
    Path(tmp).mkdir(parents=True, exist_ok=True)


_ensure_joblib_settings()

from api.client import HNClient  # noqa: E402
from api.fetching import fetch_story, get_best_stories  # noqa: E402
from api.models import Story  # noqa: E402
from api.rerank import get_embeddings, rank_stories  # noqa: E402
from api.constants import (  # noqa: E402
    KNN_NEIGHBORS,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_NEGATIVE_WEIGHT,
)


BASELINE_DEFAULT_PATH = ".cache/metrics_baseline.json"
DEFAULT_GUARD_METRICS = [
    "mrr",
    "ndcg@10",
    "ndcg@30",
    "map@10",
    "map@30",
    "recall@10",
    "recall@30",
]


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute NDCG@k - measures ranking quality with position discounting."""
    dcg = 0.0
    for i, sid in enumerate(ranked_ids[:k]):
        if sid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

    # Ideal DCG: all relevant items at top
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute Recall@k - fraction of relevant items found in top-k."""
    hits = sum(1 for sid in ranked_ids[:k] if sid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute Precision@k - fraction of top-k items that are relevant."""
    hits = sum(1 for sid in ranked_ids[:k] if sid in relevant_ids)
    return hits / k if k > 0 else 0.0


def mrr(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Compute MRR (Mean Reciprocal Rank) - 1/position of first relevant item."""
    for i, sid in enumerate(ranked_ids):
        if sid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def map_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute MAP@k (Mean Average Precision) - average precision at each relevant hit."""
    if not relevant_ids:
        return 0.0

    precisions: list[float] = []
    hits = 0
    for i, sid in enumerate(ranked_ids[:k]):
        if sid in relevant_ids:
            hits += 1
            precisions.append(hits / (i + 1))

    return sum(precisions) / min(k, len(relevant_ids)) if precisions else 0.0


def hit_rate_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute Hit@k - whether any relevant item appears in top-k."""
    return 1.0 if any(sid in relevant_ids for sid in ranked_ids[:k]) else 0.0


def mean_rank(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean rank of relevant items (lower is better)."""
    positions = [i + 1 for i, sid in enumerate(ranked_ids) if sid in relevant_ids]
    if not positions:
        return float(len(ranked_ids) + 1)
    return float(np.mean(positions))


def median_rank(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Median rank of relevant items (lower is better)."""
    positions = [i + 1 for i, sid in enumerate(ranked_ids) if sid in relevant_ids]
    if not positions:
        return float(len(ranked_ids) + 1)
    return float(np.median(positions))


def _format_metrics_line(metrics: dict[str, float], k: int) -> str:
    ndcg = metrics.get(f"ndcg@{k}", 0.0)
    map_k = metrics.get(f"map@{k}", 0.0)
    prec = metrics.get(f"precision@{k}", 0.0)
    rec = metrics.get(f"recall@{k}", 0.0)
    hit = metrics.get(f"hit@{k}", 0.0)
    return f"{k:<6} {ndcg:<8.3f} {map_k:<8.3f} {prec:<8.1%} {rec:<8.1%} {hit:<8.1%}"


def _load_baseline(path: str) -> dict[str, float] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): float(v) for k, v in data.items()}


def _save_baseline(path: str, metrics: dict[str, float]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def _guard_metrics(
    current: dict[str, float],
    baseline: dict[str, float],
    guard_metrics: list[str],
    tolerance: float,
) -> list[str]:
    failures: list[str] = []
    for key in guard_metrics:
        if key not in current or key not in baseline:
            continue
        if current[key] + tolerance < baseline[key]:
            failures.append(
                f"{key}: {current[key]:.4f} < {baseline[key]:.4f} (tol {tolerance:.4f})"
            )
    return failures


@dataclass
class EvaluationDataset:
    train_stories: list[Story]
    test_stories: list[Story]
    neg_stories: list[Story]
    candidates: list[Story]
    train_embeddings: NDArray[np.float32]
    neg_embeddings: NDArray[np.float32] | None
    pos_weights: NDArray[np.float32] | None
    test_ids: set[int]


@dataclass
class CrossValFold:
    """Single fold for cross-validation."""
    train_emb: NDArray[np.float32]
    test_ids: set[int]
    candidates: list[Story]


class RankingEvaluator:
    def __init__(self, username: str):
        self.username = username
        self.dataset: EvaluationDataset | None = None

    async def load_data(
        self,
        holdout: float = 0.2,
        limit_pos: int = 200,
        limit_neg: int = 100,
        candidate_count: int = 200,
        use_classifier: bool = True,
        use_recency: bool = False,
        cache_only: bool = False,
        allow_stale: bool = False,
    ) -> bool:
        """Load and prepare data for evaluation. Returns True if successful."""
        client = HNClient()
        if cache_only:
            print("Cache-only mode: using cached data (TTL ignored, RSS disabled).")
        print(f"Fetching upvotes for {self.username}...")
        user_data = await client.fetch_user_data(
            self.username,
            cache_only=cache_only,
            allow_stale=allow_stale,
        )

        all_positives = user_data["pos"] | user_data["upvoted"]
        hidden_ids = user_data.get("hidden", set())

        if len(all_positives) < 10:
            print(f"Need at least 10 upvotes, found {len(all_positives)}")
            return False

        print(f"Found {len(all_positives)} positive, {len(hidden_ids)} hidden")

        async with httpx.AsyncClient(timeout=30.0) as http:
            # Fetch positive stories
            print("Fetching positive stories...")
            pos_stories: list[Story] = []
            for sid in list(all_positives)[:limit_pos]:
                story = await fetch_story(
                    http,
                    sid,
                    cache_only=cache_only,
                    allow_stale=allow_stale,
                )
                if story and story.text_content:
                    pos_stories.append(story)

            # Fetch negative stories (hidden)
            neg_stories: list[Story] = []
            if use_classifier and hidden_ids:
                print("Fetching hidden stories...")
                for sid in list(hidden_ids)[:limit_neg]:
                    story = await fetch_story(
                        http,
                        sid,
                        cache_only=cache_only,
                        allow_stale=allow_stale,
                    )
                    if story and story.text_content:
                        neg_stories.append(story)
                print(f"Loaded {len(neg_stories)} hidden stories")

            if len(pos_stories) < 10:
                print(f"Only {len(pos_stories)} stories with content, need 10+")
                return False

            # Sort and split
            pos_stories.sort(key=lambda s: s.time, reverse=True)
            n_test = max(1, int(len(pos_stories) * holdout))

            test_stories = pos_stories[:n_test]
            train_stories = pos_stories[n_test:]
            train_ids = {s.id for s in train_stories}
            test_ids = {s.id for s in test_stories}

            print(f"Train: {len(train_stories)}, Test: {len(test_stories)}")

            # Compute embeddings
            print("Computing embeddings...")
            train_texts = [s.text_content for s in train_stories]
            train_emb = get_embeddings(train_texts)

            neg_emb: NDArray[np.float32] | None = None
            if neg_stories:
                neg_texts = [s.text_content for s in neg_stories]
                neg_emb = get_embeddings(neg_texts)

            # Recency weights
            pos_weights: NDArray[np.float32] | None = None
            if use_recency and train_stories:
                now = time.time()
                half_life = 90 * 24 * 3600
                times = np.array([s.time for s in train_stories], dtype=np.float32)
                ages = now - times
                pos_weights = np.exp(-ages * np.log(2) / half_life).astype(np.float32)

            # Fetch candidates (exclude both train and hidden story IDs)
            neg_ids = {s.id for s in neg_stories}
            print(f"Fetching {candidate_count} candidates...")
            candidates = await get_best_stories(
                limit=candidate_count,
                exclude_ids=train_ids | neg_ids,
                days=30,
                include_rss=False,
                cache_only=cache_only,
                allow_stale=allow_stale,
            )

            if not candidates:
                print("No candidates fetched")
                return False

            # Inject test stories
            candidate_ids = {c.id for c in candidates}
            for ts in test_stories:
                if ts.id not in candidate_ids:
                    candidates.append(ts)

            self.dataset = EvaluationDataset(
                train_stories=train_stories,
                test_stories=test_stories,
                neg_stories=neg_stories,
                candidates=candidates,
                train_embeddings=train_emb,
                neg_embeddings=neg_emb,
                pos_weights=pos_weights,
                test_ids=test_ids,
            )
            return True

    def evaluate(
        self,
        diversity: float = 0.45,
        knn: int = 2,
        neg_weight: float = 0.5,
        use_classifier: bool = True,
        k_metrics: list[int] | None = None,
    ) -> dict[str, float]:
        """Run ranking and return metrics."""
        if not self.dataset:
            raise ValueError("Dataset not loaded")

        if k_metrics is None:
            k_metrics = [10, 20, 30, 50]

        results = rank_stories(
            self.dataset.candidates,
            positive_embeddings=self.dataset.train_embeddings,
            negative_embeddings=self.dataset.neg_embeddings,
            positive_weights=self.dataset.pos_weights,
            use_classifier=use_classifier,
            diversity_lambda=diversity,
            knn_k=knn,
            neg_weight=neg_weight,
        )

        ranked_ids = [self.dataset.candidates[r.index].id for r in results]
        metrics: dict[str, float] = {}

        # Global metrics (not k-dependent)
        metrics["mrr"] = mrr(ranked_ids, self.dataset.test_ids)
        metrics["mean_rank"] = mean_rank(ranked_ids, self.dataset.test_ids)
        metrics["median_rank"] = median_rank(ranked_ids, self.dataset.test_ids)

        for k in k_metrics:
            metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, self.dataset.test_ids, k)
            metrics[f"recall@{k}"] = recall_at_k(ranked_ids, self.dataset.test_ids, k)
            metrics[f"precision@{k}"] = precision_at_k(ranked_ids, self.dataset.test_ids, k)
            metrics[f"map@{k}"] = map_at_k(ranked_ids, self.dataset.test_ids, k)
            metrics[f"hit@{k}"] = hit_rate_at_k(ranked_ids, self.dataset.test_ids, k)

        return metrics

    def evaluate_cv(
        self,
        n_folds: int = 5,
        diversity: float = RANKING_DIVERSITY_LAMBDA,
        knn: int = KNN_NEIGHBORS,
        neg_weight: float = RANKING_NEGATIVE_WEIGHT,
        use_classifier: bool = True,
        k_metrics: list[int] | None = None,
        report_each: bool = True,
        report_callback: Callable[[int, dict[str, float]], None] | None = None,
        parallel: bool = True,
    ) -> dict[str, float]:
        """Run k-fold cross-validation and return averaged metrics."""
        from concurrent.futures import ThreadPoolExecutor

        if not self.dataset:
            raise ValueError("Dataset not loaded")

        if k_metrics is None:
            k_metrics = [10, 20, 30, 50]

        # Combine train and test for CV splits
        all_stories = self.dataset.train_stories + self.dataset.test_stories
        all_emb = get_embeddings([s.text_content for s in all_stories])
        n = len(all_stories)

        # Build combined weights (train weights + uniform for test stories)
        if self.dataset.pos_weights is not None:
            n_test = len(self.dataset.test_stories)
            all_weights: NDArray[np.float32] | None = np.concatenate([
                self.dataset.pos_weights,
                np.ones(n_test, dtype=np.float32),
            ])
        else:
            all_weights = None

        # Shuffle indices for random folds (deterministic before threads)
        indices = np.random.permutation(n)
        fold_size = n // n_folds

        def _run_fold(fold: int) -> dict[str, float]:
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n
            test_idx = set(indices[test_start:test_end])
            train_idx = [i for i in range(n) if i not in test_idx]

            train_emb = all_emb[train_idx]
            fold_weights = all_weights[train_idx] if all_weights is not None else None
            test_ids = {all_stories[i].id for i in test_idx}

            candidate_ids = {c.id for c in self.dataset.candidates}
            fold_candidates = list(self.dataset.candidates)
            for i in test_idx:
                if all_stories[i].id not in candidate_ids:
                    fold_candidates.append(all_stories[i])

            results = rank_stories(
                fold_candidates,
                positive_embeddings=train_emb,
                negative_embeddings=self.dataset.neg_embeddings,
                positive_weights=fold_weights,
                use_classifier=use_classifier,
                diversity_lambda=diversity,
                knn_k=knn,
                neg_weight=neg_weight,
            )

            ranked_ids = [fold_candidates[r.index].id for r in results]

            # Debug assertion: hidden stories must never appear in ranked output
            neg_ids = {s.id for s in self.dataset.neg_stories}
            leaked = set(ranked_ids) & neg_ids
            assert not leaked, f"Hidden stories leaked into ranked results: {leaked}"

            fold_metrics: dict[str, float] = {"mrr": mrr(ranked_ids, test_ids)}
            fold_metrics["mean_rank"] = mean_rank(ranked_ids, test_ids)
            fold_metrics["median_rank"] = median_rank(ranked_ids, test_ids)

            for k in k_metrics:
                fold_metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, test_ids, k)
                fold_metrics[f"recall@{k}"] = recall_at_k(ranked_ids, test_ids, k)
                fold_metrics[f"precision@{k}"] = precision_at_k(ranked_ids, test_ids, k)
                fold_metrics[f"map@{k}"] = map_at_k(ranked_ids, test_ids, k)
                fold_metrics[f"hit@{k}"] = hit_rate_at_k(ranked_ids, test_ids, k)

            return fold_metrics

        # Run folds (parallel or serial)
        if parallel and n_folds > 1:
            with ThreadPoolExecutor(max_workers=n_folds) as pool:
                futures = [pool.submit(_run_fold, fold) for fold in range(n_folds)]
                all_metrics = [f.result() for f in futures]
        else:
            all_metrics = [_run_fold(fold) for fold in range(n_folds)]

        # Process results in fold order (reporting, callbacks, pruning)
        for fold, fold_metrics in enumerate(all_metrics):
            if report_each:
                print(f"\nFold {fold + 1}/{n_folds}:")
                print(f"  MRR: {fold_metrics['mrr']:.3f}")
                print(f"  Mean Rank: {fold_metrics['mean_rank']:.1f}")
                print(f"  Median Rank: {fold_metrics['median_rank']:.1f}")
                print(f"\n  {'k':<6} {'NDCG':<8} {'MAP':<8} {'Prec':<8} {'Recall':<8} {'Hit':<8}")
                print("  " + "-" * 50)
                for k in k_metrics:
                    print("  " + _format_metrics_line(fold_metrics, k))
            if report_callback:
                interim_metrics: dict[str, float] = {}
                for key in all_metrics[0]:
                    values = [all_metrics[i][key] for i in range(fold + 1)]
                    interim_metrics[key] = float(np.mean(values))
                report_callback(fold, interim_metrics)

        # Average across folds
        avg_metrics: dict[str, float] = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f"{key}_std"] = float(np.std(values))

        return avg_metrics


async def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking quality")
    parser.add_argument("username", help="HN username")
    parser.add_argument("--holdout", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--k", type=int, default=30, help="Cutoff for metrics")
    parser.add_argument("--candidates", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--classifier", action="store_true", help="Use classifier mode with hidden stories")
    parser.add_argument("--recency", action="store_true", help="Apply recency weighting to train embeddings")
    parser.add_argument("--diversity", type=float, default=RANKING_DIVERSITY_LAMBDA, help="MMR diversity lambda")
    parser.add_argument("--knn", type=int, default=KNN_NEIGHBORS, help="k-NN neighbors for scoring")
    parser.add_argument("--neg-weight", type=float, default=RANKING_NEGATIVE_WEIGHT, help="Weight for negative similarity penalty")
    parser.add_argument("--cv", type=int, default=0, help="Number of CV folds (0=single holdout)")
    parser.add_argument("--baseline", default=BASELINE_DEFAULT_PATH, help="Path to metrics baseline JSON")
    parser.add_argument("--save-baseline", action="store_true", help="Save current metrics as baseline")
    parser.add_argument("--no-guard", action="store_true", help="Disable baseline regression guard")
    parser.add_argument("--guard-metrics", nargs="*", default=DEFAULT_GUARD_METRICS, help="Metrics to guard")
    parser.add_argument("--guard-tolerance", type=float, default=0.0, help="Allowed drop vs baseline")
    parser.add_argument("--cache-only", action="store_true", help="Use cached data only (ignore TTL, no RSS)")
    args = parser.parse_args()

    evaluator = RankingEvaluator(args.username)
    success = await evaluator.load_data(
        holdout=args.holdout,
        candidate_count=args.candidates,
        use_classifier=args.classifier,
        use_recency=args.recency,
        cache_only=args.cache_only,
        allow_stale=args.cache_only,
    )

    if not success:
        return
    if evaluator.dataset is None:
        return
    dataset = evaluator.dataset

    if args.cv > 0:
        print(f"\nRunning {args.cv}-fold cross-validation...")
        metrics = evaluator.evaluate_cv(
            n_folds=args.cv,
            diversity=args.diversity,
            knn=args.knn,
            neg_weight=args.neg_weight,
            use_classifier=args.classifier,
            report_each=True,
        )
        print(f"\nMRR: {metrics.get('mrr', 0.0):.3f} (±{metrics.get('mrr_std', 0.0):.3f})")
        print(f"Mean Rank: {metrics.get('mean_rank', 0.0):.1f} (±{metrics.get('mean_rank_std', 0.0):.1f})")
        print(f"Median Rank: {metrics.get('median_rank', 0.0):.1f} (±{metrics.get('median_rank_std', 0.0):.1f})")
        print(f"\n{'k':<6} {'NDCG':<12} {'MAP':<12} {'Prec':<12} {'Recall':<12}")
        print("-" * 54)
        for k in [10, 20, 30, 50]:
            ndcg = metrics.get(f"ndcg@{k}", 0.0)
            ndcg_std = metrics.get(f"ndcg@{k}_std", 0.0)
            map_k = metrics.get(f"map@{k}", 0.0)
            map_std = metrics.get(f"map@{k}_std", 0.0)
            prec = metrics.get(f"precision@{k}", 0.0)
            rec = metrics.get(f"recall@{k}", 0.0)
            print(f"{k:<6} {ndcg:.3f}±{ndcg_std:.2f}  {map_k:.3f}±{map_std:.2f}  {prec:<8.1%}   {rec:<8.1%}")
    else:
        metrics = evaluator.evaluate(
            diversity=args.diversity,
            knn=args.knn,
            neg_weight=args.neg_weight,
            use_classifier=args.classifier,
        )
        print(f"\nMRR: {metrics.get('mrr', 0.0):.3f}")
        print(f"Mean Rank: {metrics.get('mean_rank', 0.0):.1f}")
        print(f"Median Rank: {metrics.get('median_rank', 0.0):.1f}")
        print(f"\n{'k':<6} {'NDCG':<8} {'MAP':<8} {'Prec':<8} {'Recall':<8} {'Hit':<8}")
        print("-" * 50)
        for k in [10, 20, 30, 50]:
            print(_format_metrics_line(metrics, k))

    if args.save_baseline:
        _save_baseline(args.baseline, metrics)
        print(f"\nSaved baseline to {args.baseline}")
    else:
        baseline = _load_baseline(args.baseline)
        if baseline and not args.no_guard:
            failures = _guard_metrics(
                metrics,
                baseline,
                guard_metrics=args.guard_metrics,
                tolerance=args.guard_tolerance,
            )
            if failures:
                print("\n[!] Metric regression detected:")
                for msg in failures:
                    print(f"  - {msg}")
                raise SystemExit(2)
        elif not baseline:
            print(f"\n[!] No baseline found at {args.baseline}; skipping guard.")

    # Just re-run ranking to get the results object for the verbose output
    # (The evaluate method only returns metrics)
    results = rank_stories(
        dataset.candidates,
        positive_embeddings=dataset.train_embeddings,
        negative_embeddings=dataset.neg_embeddings,
        positive_weights=dataset.pos_weights,
        use_classifier=args.classifier,
        diversity_lambda=args.diversity,
        knn_k=args.knn,
        neg_weight=args.neg_weight,
    )
    ranked_ids = [dataset.candidates[r.index].id for r in results]

    print("\nTest story positions in ranking:")
    for i, sid in enumerate(ranked_ids):
        if sid in dataset.test_ids:
            score = results[i].hybrid_score if i < len(results) else 0
            print(f"  #{i+1}: story {sid} (score: {score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
