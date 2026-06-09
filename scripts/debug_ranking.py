# ruff: noqa
#!/usr/bin/env -S uv run
"""Debug: dump actual ranking scores for Jina 256 vs 768."""

import asyncio
import contextlib
import sys
from pathlib import Path

import numpy as np

sys.argv = [
    "eval",
    "--feedback-path",
    ".cache/user_feedback/dashboard_feedback_debug.json",
]

from api.config import AppConfig
from api.feedback import load_feedback
from api.feedback_single_model import (
    train_single_model_from_embeddings,
)
from api.rerank import rank_stories
from scripts.evaluate_feedback_models import (
    _build_dataset,
    _count_by_action,
    _time_split,
    _override_rerank,
    _evaluator_training_config,
)


@contextlib.contextmanager
def _debug_override(model_dir: str | None, embedding_max_tokens: int | None):
    """Same as _override_rerank but we don't need the full module."""
    pass  # We'll handle model override manually


async def debug_ranking(
    model_dir: str | None, label: str, config: AppConfig, cv: int = 0
):
    records = load_feedback(Path(".cache/user_feedback/dashboard_feedback_debug.json"))
    if not records:
        print("No records")
        return

    up, down, neutral = _count_by_action(records)
    print(f"\n{'=' * 60}")
    print(f"  {label}: {len(records)} records ({up}u, {down}d, {neutral}n)")
    print(f"{'=' * 60}")

    from api import rerank as r_mod

    with _override_rerank(r_mod, model_dir, 4096):
        train_records, test_records = _time_split(records, 0.2)
        ds = await _build_dataset(records, train_records, test_records, cache_only=True)
        if ds is None:
            print("Dataset empty")
            return

        has_neg = (
            ds.neg_embeddings.shape[0] > 0 if ds.neg_embeddings.ndim == 2 else False
        )
        training_config = _evaluator_training_config(config)

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

        # Dump top 50 with labels
        print(f"\n{'Rank':>5} {'Score':>10} {'Label':>8} {'Story ID':>12}  Title")
        print("-" * 80)
        for i, r in enumerate(results[:50]):
            story = ds.candidates[r.index]
            key = ds.candidate_keys[r.index]
            label_str = (
                "UP"
                if key in ds.test_up_keys
                else (
                    "DN"
                    if key in ds.test_down_keys
                    else ("NE" if key in ds.test_neutral_keys else "  -")
                )
            )
            title = story.title[:60]
            print(
                f"{i + 1:>5} {r.model_score:>10.5f} {label_str:>8} {story.id:>12}  {title}"
            )

        # Score distributions
        scores = np.array([r.model_score for r in results])
        up_scores = np.array(
            [
                r.model_score
                for i, r in enumerate(results)
                if ds.candidate_keys[r.index] in ds.test_up_keys
            ]
        )
        down_scores = np.array(
            [
                r.model_score
                for i, r in enumerate(results)
                if ds.candidate_keys[r.index] in ds.test_down_keys
            ]
        )
        neutral_scores = np.array(
            [
                r.model_score
                for i, r in enumerate(results)
                if ds.candidate_keys[r.index] in ds.test_neutral_keys
            ]
        )
        dist_scores = np.array(
            [
                r.model_score
                for i, r in enumerate(results)
                if ds.candidate_keys[r.index] not in ds.test_up_keys
                and ds.candidate_keys[r.index] not in ds.test_down_keys
                and ds.candidate_keys[r.index] not in ds.test_neutral_keys
            ]
        )

        print("\n  Score distributions:")
        for name, arr in [
            ("UP", up_scores),
            ("DOWN", down_scores),
            ("NEUTRAL", neutral_scores),
            ("DIST", dist_scores),
        ]:
            if len(arr) > 0:
                print(
                    f"  {name:>8} (n={len(arr):>4}): mean={arr.mean():+.5f}, std={arr.std():.5f}, min={arr.min():+.5f}, max={arr.max():+.5f}, pos={sum(s > 0 for s in arr):>4}/{len(arr)}"
                )

        # Rank of test upvotes
        up_ranks = [
            i + 1
            for i, r in enumerate(results)
            if ds.candidate_keys[r.index] in ds.test_up_keys
        ]
        down_ranks = [
            i + 1
            for i, r in enumerate(results)
            if ds.candidate_keys[r.index] in ds.test_down_keys
        ]
        neutral_ranks = [
            i + 1
            for i, r in enumerate(results)
            if ds.candidate_keys[r.index] in ds.test_neutral_keys
        ]
        print(
            f"\n  Test up ranks:    mean={np.mean(up_ranks):.0f}, median={np.median(up_ranks):.0f}, top10={sum(1 for r in up_ranks if r <= 10)}/{len(up_ranks)}"
        )
        print(
            f"  Test down ranks:  mean={np.mean(down_ranks):.0f}, median={np.median(down_ranks):.0f}, top10={sum(1 for r in down_ranks if r <= 10)}/{len(down_ranks)}"
        )
        print(
            f"  Test neutral ranks: mean={np.mean(neutral_ranks):.0f}, median={np.median(neutral_ranks):.0f}"
        )
        print()


async def main():
    config = AppConfig.load()
    await debug_ranking(None, "MiniLM 256 (default)", config)
    await debug_ranking("onnx_model_jina_small", "Jina 256", config)
    await debug_ranking("onnx_model_jina_small_768", "Jina 768", config)


if __name__ == "__main__":
    asyncio.run(main())
