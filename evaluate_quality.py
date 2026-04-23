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
from typing import TypedDict, cast

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
from api.models import RankResult, Story, StoryDict  # noqa: E402
from api.rerank import compute_recency_weights, get_embeddings, rank_stories  # noqa: E402
from api.url_utils import normalize_url  # noqa: E402
from api.constants import (  # noqa: E402
    KNN_NEIGHBORS,
    POSITIVE_RECENCY_ENABLED,
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
DEFAULT_K_METRICS = [10, 20, 30, 50]


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


def _compute_metrics(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k_metrics: list[int],
) -> dict[str, float]:
    metrics = {
        "mrr": mrr(ranked_ids, relevant_ids),
        "mean_rank": mean_rank(ranked_ids, relevant_ids),
        "median_rank": median_rank(ranked_ids, relevant_ids),
    }
    for k in k_metrics:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked_ids, relevant_ids, k)
        metrics[f"map@{k}"] = map_at_k(ranked_ids, relevant_ids, k)
        metrics[f"hit@{k}"] = hit_rate_at_k(ranked_ids, relevant_ids, k)
    return metrics


def _average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    avg_metrics: dict[str, float] = {}
    for key in all_metrics[0]:
        values = [metrics[key] for metrics in all_metrics]
        avg_metrics[key] = float(np.mean(values))
        avg_metrics[f"{key}_std"] = float(np.std(values))
    return avg_metrics


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _summarize_rank_diagnostics(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {}

    rank_calls = len(records)
    classifier_requested_count = sum(
        1 for record in records if bool(record.get("classifier_requested", False))
    )
    classifier_used_count = sum(
        1 for record in records if bool(record.get("classifier_used", False))
    )
    local_hidden_penalty_applied_count = sum(
        1
        for record in records
        if bool(record.get("local_hidden_penalty_applied", False))
    )

    def _avg(key: str) -> float:
        return float(np.mean([_as_float(record.get(key, 0.0)) for record in records]))

    failure_reasons: dict[str, int] = {}
    for record in records:
        reason = record.get("classifier_failure_reason")
        if not reason:
            continue
        failure_reasons[str(reason)] = failure_reasons.get(str(reason), 0) + 1

    return {
        "rank_calls": rank_calls,
        "classifier_requested_count": classifier_requested_count,
        "classifier_used_count": classifier_used_count,
        "classifier_fallback_count": classifier_requested_count - classifier_used_count,
        "classifier_used_rate": (
            float(classifier_used_count / classifier_requested_count)
            if classifier_requested_count
            else 0.0
        ),
        "local_hidden_penalty_applied_count": local_hidden_penalty_applied_count,
        "avg_positive_count": _avg("positive_count"),
        "avg_negative_count": _avg("negative_count"),
        "avg_base_feature_dim": _avg("base_feature_dim"),
        "avg_derived_feature_dim": _avg("derived_feature_dim"),
        "avg_local_hidden_penalty_mean": _avg("local_hidden_penalty_mean"),
        "avg_local_hidden_penalty_max": _avg("local_hidden_penalty_max"),
        "max_local_hidden_penalty_max": float(
            max(
                _as_float(record.get("local_hidden_penalty_max", 0.0))
                for record in records
            )
        ),
        "classifier_failure_reasons": failure_reasons,
    }


def _merge_rank_diagnostic_summaries(
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    if not summaries:
        return {}

    total_rank_calls = sum(_as_int(summary.get("rank_calls", 0)) for summary in summaries)
    classifier_requested_count = sum(
        _as_int(summary.get("classifier_requested_count", 0)) for summary in summaries
    )
    classifier_used_count = sum(
        _as_int(summary.get("classifier_used_count", 0)) for summary in summaries
    )
    local_hidden_penalty_applied_count = sum(
        _as_int(summary.get("local_hidden_penalty_applied_count", 0))
        for summary in summaries
    )

    def _weighted_avg(key: str) -> float:
        if total_rank_calls == 0:
            return 0.0
        weighted_total = sum(
            _as_float(summary.get(key, 0.0)) * _as_int(summary.get("rank_calls", 0))
            for summary in summaries
        )
        return float(weighted_total / total_rank_calls)

    failure_reasons: dict[str, int] = {}
    for summary in summaries:
        reasons = summary.get("classifier_failure_reasons", {})
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            failure_reasons[str(reason)] = (
                failure_reasons.get(str(reason), 0) + _as_int(count)
            )

    return {
        "rank_calls": total_rank_calls,
        "classifier_requested_count": classifier_requested_count,
        "classifier_used_count": classifier_used_count,
        "classifier_fallback_count": classifier_requested_count - classifier_used_count,
        "classifier_used_rate": (
            float(classifier_used_count / classifier_requested_count)
            if classifier_requested_count
            else 0.0
        ),
        "local_hidden_penalty_applied_count": local_hidden_penalty_applied_count,
        "avg_positive_count": _weighted_avg("avg_positive_count"),
        "avg_negative_count": _weighted_avg("avg_negative_count"),
        "avg_base_feature_dim": _weighted_avg("avg_base_feature_dim"),
        "avg_derived_feature_dim": _weighted_avg("avg_derived_feature_dim"),
        "avg_local_hidden_penalty_mean": _weighted_avg("avg_local_hidden_penalty_mean"),
        "avg_local_hidden_penalty_max": _weighted_avg("avg_local_hidden_penalty_max"),
        "max_local_hidden_penalty_max": float(
            max(
                _as_float(summary.get("max_local_hidden_penalty_max", 0.0))
                for summary in summaries
            )
        ),
        "classifier_failure_reasons": failure_reasons,
    }


def _finalize_ranked_results(
    results: list[RankResult],
    candidates: list[Story],
    count: int,
) -> list[RankResult]:
    from generate_html import select_ranked_results

    selected_results = select_ranked_results(
        results,
        candidates,
        cluster_labels=None,
        cluster_names={},
        cand_cluster_map={},
        count=count,
    )

    final_results: list[RankResult] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for result in selected_results:
        story = candidates[result.index]
        url = story.url
        title = story.title
        norm_url = normalize_url(url) if url else f"{story.source}:{story.id}"
        norm_title = title.lower().strip() if title else ""
        if norm_url in seen_urls or norm_title in seen_titles:
            continue
        if url:
            seen_urls.add(norm_url)
        if title:
            seen_titles.add(norm_title)
        final_results.append(result)
    return final_results


def _finalize_ranked_ids(
    results: list[RankResult],
    candidates: list[Story],
    count: int,
) -> list[int]:
    return [candidates[result.index].id for result in _finalize_ranked_results(results, candidates, count)]


def _format_metrics_line(metrics: dict[str, float], k: int) -> str:
    ndcg = metrics.get(f"ndcg@{k}", 0.0)
    map_k = metrics.get(f"map@{k}", 0.0)
    prec = metrics.get(f"precision@{k}", 0.0)
    rec = metrics.get(f"recall@{k}", 0.0)
    hit = metrics.get(f"hit@{k}", 0.0)
    return f"{k:<6} {ndcg:<8.3f} {map_k:<8.3f} {prec:<8.1%} {rec:<8.1%} {hit:<8.1%}"


def _print_metrics_report(
    metrics: dict[str, float],
    k_metrics: list[int],
    *,
    include_std: bool,
    include_hit: bool,
    indent: str = "",
) -> None:
    if include_std:
        print(
            f"{indent}MRR: {metrics.get('mrr', 0.0):.3f} "
            f"(±{metrics.get('mrr_std', 0.0):.3f})"
        )
        print(
            f"{indent}Mean Rank: {metrics.get('mean_rank', 0.0):.1f} "
            f"(±{metrics.get('mean_rank_std', 0.0):.1f})"
        )
        print(
            f"{indent}Median Rank: {metrics.get('median_rank', 0.0):.1f} "
            f"(±{metrics.get('median_rank_std', 0.0):.1f})"
        )
        print(
            f"\n{indent}{'k':<6} {'NDCG':<12} {'MAP':<12} {'Prec':<12} {'Recall':<12}"
        )
        print(indent + "-" * 54)
        for k in k_metrics:
            print(
                f"{indent}{k:<6} "
                f"{metrics.get(f'ndcg@{k}', 0.0):.3f}±{metrics.get(f'ndcg@{k}_std', 0.0):.2f}  "
                f"{metrics.get(f'map@{k}', 0.0):.3f}±{metrics.get(f'map@{k}_std', 0.0):.2f}  "
                f"{metrics.get(f'precision@{k}', 0.0):<8.1%}   "
                f"{metrics.get(f'recall@{k}', 0.0):<8.1%}"
            )
        return

    print(f"{indent}MRR: {metrics.get('mrr', 0.0):.3f}")
    print(f"{indent}Mean Rank: {metrics.get('mean_rank', 0.0):.1f}")
    print(f"{indent}Median Rank: {metrics.get('median_rank', 0.0):.1f}")
    if include_hit:
        print(
            f"\n{indent}{'k':<6} {'NDCG':<8} {'MAP':<8} {'Prec':<8} "
            f"{'Recall':<8} {'Hit':<8}"
        )
        print(indent + "-" * 50)
        for k in k_metrics:
            print(indent + _format_metrics_line(metrics, k))
        return

    print(
        f"\n{indent}{'k':<6} {'NDCG':<8} {'MAP':<8} {'Prec':<8} {'Recall':<8}"
    )
    print(indent + "-" * 42)
    for k in k_metrics:
        print(
            f"{indent}{k:<6} "
            f"{metrics.get(f'ndcg@{k}', 0.0):<8.3f} "
            f"{metrics.get(f'map@{k}', 0.0):<8.3f} "
            f"{metrics.get(f'precision@{k}', 0.0):<8.1%} "
            f"{metrics.get(f'recall@{k}', 0.0):<8.1%}"
        )


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


class SnapshotPayload(TypedDict):
    format_version: int
    username: str
    saved_at: int
    metadata: dict[str, object]
    train_stories: list[StoryDict]
    test_stories: list[StoryDict]
    neg_stories: list[StoryDict]
    candidates: list[StoryDict]
    pos_weights: list[float] | None
    test_ids: list[int]


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
        self.snapshot_metadata: dict[str, object] = {}
        self.last_diagnostics_summary: dict[str, object] = {}

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
        async with HNClient() as client:
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
            if use_recency and POSITIVE_RECENCY_ENABLED and train_stories:
                pos_weights = compute_recency_weights([s.time for s in train_stories])

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
            self.snapshot_metadata = {
                "source": "live",
                "holdout": holdout,
                "limit_pos": limit_pos,
                "limit_neg": limit_neg,
                "candidate_count": candidate_count,
                "use_classifier": use_classifier,
                "use_recency": use_recency,
                "cache_only": cache_only,
                "allow_stale": allow_stale,
            }
            return True

    def save_snapshot(
        self,
        path: str | Path,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        dataset = self.dataset
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload: SnapshotPayload = {
            "format_version": 1,
            "username": self.username,
            "saved_at": int(time.time()),
            "metadata": {
                **self.snapshot_metadata,
                **(metadata or {}),
            },
            "train_stories": [story.to_dict() for story in dataset.train_stories],
            "test_stories": [story.to_dict() for story in dataset.test_stories],
            "neg_stories": [story.to_dict() for story in dataset.neg_stories],
            "candidates": [story.to_dict() for story in dataset.candidates],
            "pos_weights": (
                dataset.pos_weights.astype(float).tolist()
                if dataset.pos_weights is not None
                else None
            ),
            "test_ids": sorted(dataset.test_ids),
        }
        snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return snapshot_path

    def load_snapshot(self, path: str | Path) -> bool:
        snapshot_path = Path(path)
        try:
            raw = cast(SnapshotPayload, json.loads(snapshot_path.read_text()))
        except Exception:
            return False

        train_stories = [
            Story.from_dict(story)
            for story in raw.get("train_stories", [])
        ]
        test_stories = [
            Story.from_dict(story)
            for story in raw.get("test_stories", [])
        ]
        neg_stories = [
            Story.from_dict(story)
            for story in raw.get("neg_stories", [])
        ]
        candidates = [
            Story.from_dict(story)
            for story in raw.get("candidates", [])
        ]
        if not train_stories or not test_stories or not candidates:
            return False

        train_embeddings = get_embeddings([story.text_content for story in train_stories])
        neg_embeddings: NDArray[np.float32] | None = None
        if neg_stories:
            neg_embeddings = get_embeddings([story.text_content for story in neg_stories])

        raw_weights = raw.get("pos_weights")
        pos_weights: NDArray[np.float32] | None = None
        if isinstance(raw_weights, list):
            pos_weights = np.array(raw_weights, dtype=np.float32)

        raw_test_ids = raw.get("test_ids", [])
        if isinstance(raw_test_ids, list):
            test_ids = {int(sid) for sid in raw_test_ids}
        else:
            test_ids = {story.id for story in test_stories}

        self.username = str(raw.get("username", self.username))
        metadata = raw.get("metadata", {})
        self.snapshot_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        self.snapshot_metadata["source"] = "snapshot"
        self.snapshot_metadata["snapshot_path"] = str(snapshot_path)
        self.dataset = EvaluationDataset(
            train_stories=train_stories,
            test_stories=test_stories,
            neg_stories=neg_stories,
            candidates=candidates,
            train_embeddings=train_embeddings,
            neg_embeddings=neg_embeddings,
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
        final_list_count: int | None = None,
        diagnostics_summary: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """Run ranking and return metrics."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        dataset = self.dataset

        if k_metrics is None:
            k_metrics = DEFAULT_K_METRICS

        rank_diagnostics: dict[str, object] | None = (
            {} if diagnostics_summary is not None else None
        )
        results = rank_stories(
            dataset.candidates,
            positive_embeddings=dataset.train_embeddings,
            negative_embeddings=dataset.neg_embeddings,
            positive_weights=dataset.pos_weights,
            use_classifier=use_classifier,
            diversity_lambda=diversity,
            knn_k=knn,
            neg_weight=neg_weight,
            diagnostics=rank_diagnostics,
        )

        summary = (
            _summarize_rank_diagnostics([rank_diagnostics])
            if rank_diagnostics is not None
            else {}
        )
        self.last_diagnostics_summary = summary
        if diagnostics_summary is not None:
            diagnostics_summary.clear()
            diagnostics_summary.update(summary)

        ranked_ids = (
            _finalize_ranked_ids(results, dataset.candidates, final_list_count)
            if final_list_count is not None
            else [dataset.candidates[r.index].id for r in results]
        )
        return _compute_metrics(ranked_ids, dataset.test_ids, k_metrics)

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
        final_list_count: int | None = None,
        diagnostics_summary: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """Run k-fold cross-validation and return averaged metrics."""
        from concurrent.futures import ThreadPoolExecutor

        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        dataset = self.dataset

        if k_metrics is None:
            k_metrics = DEFAULT_K_METRICS

        # Combine train and test for CV splits
        all_stories = dataset.train_stories + dataset.test_stories
        all_emb = get_embeddings([s.text_content for s in all_stories])
        n = len(all_stories)

        # Build combined weights (train weights + uniform for test stories)
        if dataset.pos_weights is not None:
            n_test = len(dataset.test_stories)
            all_weights: NDArray[np.float32] | None = np.concatenate([
                dataset.pos_weights,
                np.ones(n_test, dtype=np.float32),
            ])
        else:
            all_weights = None

        # Shuffle indices for random folds (deterministic before threads)
        indices = np.random.permutation(n)
        fold_size = n // n_folds

        def _run_fold(
            fold: int,
        ) -> tuple[dict[str, float], dict[str, object] | None]:
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n
            test_idx = set(indices[test_start:test_end])
            train_idx = [i for i in range(n) if i not in test_idx]

            train_emb = all_emb[train_idx]
            fold_weights = all_weights[train_idx] if all_weights is not None else None
            test_ids = {all_stories[i].id for i in test_idx}

            candidate_ids = {c.id for c in dataset.candidates}
            fold_candidates = list(dataset.candidates)
            for i in test_idx:
                if all_stories[i].id not in candidate_ids:
                    fold_candidates.append(all_stories[i])

            rank_diagnostics: dict[str, object] | None = (
                {} if diagnostics_summary is not None else None
            )
            results = rank_stories(
                fold_candidates,
                positive_embeddings=train_emb,
                negative_embeddings=dataset.neg_embeddings,
                positive_weights=fold_weights,
                use_classifier=use_classifier,
                diversity_lambda=diversity,
                knn_k=knn,
                neg_weight=neg_weight,
                diagnostics=rank_diagnostics,
            )

            ranked_ids = [fold_candidates[r.index].id for r in results]

            # Debug assertion: hidden stories must never appear in ranked output
            neg_ids = {s.id for s in dataset.neg_stories}
            leaked = {fold_candidates[r.index].id for r in results} & neg_ids
            assert not leaked, f"Hidden stories leaked into ranked results: {leaked}"

            if final_list_count is not None:
                ranked_ids = _finalize_ranked_ids(
                    results,
                    fold_candidates,
                    final_list_count,
                )
            else:
                ranked_ids = [fold_candidates[r.index].id for r in results]

            return _compute_metrics(ranked_ids, test_ids, k_metrics), rank_diagnostics

        # Run folds (parallel or serial)
        if parallel and n_folds > 1:
            with ThreadPoolExecutor(max_workers=n_folds) as pool:
                futures = [pool.submit(_run_fold, fold) for fold in range(n_folds)]
                fold_outputs = [f.result() for f in futures]
        else:
            fold_outputs = [_run_fold(fold) for fold in range(n_folds)]

        all_metrics = [metrics for metrics, _ in fold_outputs]
        diagnostics_records = [
            diag for _, diag in fold_outputs if diag is not None
        ]

        # Process results in fold order (reporting, callbacks, pruning)
        for fold, fold_metrics in enumerate(all_metrics):
            if report_each:
                print(f"\nFold {fold + 1}/{n_folds}:")
                _print_metrics_report(
                    fold_metrics,
                    k_metrics,
                    include_std=False,
                    include_hit=True,
                    indent="  ",
                )
            if report_callback:
                interim_metrics: dict[str, float] = {}
                for key in all_metrics[0]:
                    values = [all_metrics[i][key] for i in range(fold + 1)]
                    interim_metrics[key] = float(np.mean(values))
                report_callback(fold, interim_metrics)

        summary = _summarize_rank_diagnostics(diagnostics_records)
        self.last_diagnostics_summary = summary
        if diagnostics_summary is not None:
            diagnostics_summary.clear()
            diagnostics_summary.update(summary)

        return _average_metrics(all_metrics)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking quality")
    parser.add_argument("username", help="HN username")
    parser.add_argument("--holdout", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--k", type=int, default=30, help="Cutoff for metrics")
    parser.add_argument("--candidates", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--classifier", action="store_true", help="Use classifier mode with hidden stories")
    parser.add_argument(
        "--recency",
        action=argparse.BooleanOptionalAction,
        default=POSITIVE_RECENCY_ENABLED,
        help="Enable or disable recency weighting for positive signals",
    )
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
    parser.add_argument(
        "--final-list",
        action="store_true",
        help="Evaluate the final displayed list policy instead of raw rank_stories output",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Displayed story count when --final-list is enabled",
    )
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
    k_metrics = (
        [k for k in DEFAULT_K_METRICS if k <= args.count] or [args.count]
        if args.final_list
        else DEFAULT_K_METRICS
    )

    if args.cv > 0:
        print(f"\nRunning {args.cv}-fold cross-validation...")
        metrics = evaluator.evaluate_cv(
            n_folds=args.cv,
            diversity=args.diversity,
            knn=args.knn,
            neg_weight=args.neg_weight,
            use_classifier=args.classifier,
            k_metrics=k_metrics,
            report_each=True,
            final_list_count=args.count if args.final_list else None,
        )
        _print_metrics_report(
            metrics,
            k_metrics,
            include_std=True,
            include_hit=False,
        )
    else:
        metrics = evaluator.evaluate(
            diversity=args.diversity,
            knn=args.knn,
            neg_weight=args.neg_weight,
            use_classifier=args.classifier,
            k_metrics=k_metrics,
            final_list_count=args.count if args.final_list else None,
        )
        _print_metrics_report(
            metrics,
            k_metrics,
            include_std=False,
            include_hit=True,
        )

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
    if args.final_list:
        final_results = _finalize_ranked_results(results, dataset.candidates, args.count)
        ranked_results = final_results
        print("\nTest story positions in final list:")
    else:
        ranked_results = results
        print("\nTest story positions in ranking:")

    for i, result in enumerate(ranked_results):
        sid = dataset.candidates[result.index].id
        if sid in dataset.test_ids:
            print(f"  #{i+1}: story {sid} (score: {result.hybrid_score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
