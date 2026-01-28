#!/usr/bin/env -S uv run
"""Evaluate ranking quality using holdout validation on user's upvote history."""

import argparse
import asyncio
import numpy as np
from numpy.typing import NDArray

from api.client import HNClient
from api.fetching import fetch_story, get_best_stories
from api.models import Story
from api.rerank import get_embeddings, rank_stories
import httpx


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute NDCG@k - measures ranking quality with position discounting."""
    dcg = 0.0
    for i, sid in enumerate(ranked_ids[:k]):
        if sid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

    # Ideal DCG: all relevant items at top
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Compute Hit Rate@k - fraction of relevant items in top-k."""
    hits = sum(1 for sid in ranked_ids[:k] if sid in relevant_ids)
    return hits / min(k, len(relevant_ids)) if relevant_ids else 0.0


async def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking quality")
    parser.add_argument("username", help="HN username")
    parser.add_argument("--holdout", type=float, default=0.2, help="Fraction for test set")
    parser.add_argument("--k", type=int, default=30, help="Cutoff for metrics")
    parser.add_argument("--candidates", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--classifier", action="store_true", help="Use classifier mode with hidden stories")
    args = parser.parse_args()

    client = HNClient()

    print(f"Fetching upvotes for {args.username}...")
    user_data = await client.fetch_user_data(args.username)

    # Combine favorites + upvoted as positive signals
    all_positives = user_data["pos"] | user_data["upvoted"]
    hidden_ids = user_data.get("hidden", set())

    if len(all_positives) < 10:
        print(f"Need at least 10 upvotes, found {len(all_positives)}")
        return

    print(f"Found {len(all_positives)} positive, {len(hidden_ids)} hidden")

    # Fetch story details for positives (need content for embeddings)
    print("Fetching story content...")
    async with httpx.AsyncClient(timeout=30.0) as http:
        pos_stories: list[Story] = []
        for sid in list(all_positives)[:200]:  # Cap to avoid API limits
            story = await fetch_story(http, sid)
            if story and story.text_content:
                pos_stories.append(story)

        # Fetch hidden stories for classifier
        neg_stories: list[Story] = []
        if args.classifier and hidden_ids:
            print("Fetching hidden stories for classifier...")
            for sid in list(hidden_ids)[:100]:
                story = await fetch_story(http, sid)
                if story and story.text_content:
                    neg_stories.append(story)
            print(f"Loaded {len(neg_stories)} hidden stories")

        if len(pos_stories) < 10:
            print(f"Only {len(pos_stories)} stories with content, need 10+")
            return

        # Sort by time (newest first) and split
        pos_stories.sort(key=lambda s: s.time, reverse=True)
        n_test = max(1, int(len(pos_stories) * args.holdout))

        test_stories = pos_stories[:n_test]
        train_stories = pos_stories[n_test:]

        test_ids = {s.id for s in test_stories}
        train_ids = {s.id for s in train_stories}

        print(f"Train: {len(train_stories)}, Test: {len(test_stories)}")

        # Build embeddings from train set
        print("Computing train embeddings...")
        train_texts = [s.text_content for s in train_stories]
        train_emb: NDArray[np.float32] = get_embeddings(train_texts)

        # Build negative embeddings if using classifier
        neg_emb: NDArray[np.float32] | None = None
        if args.classifier and neg_stories:
            print("Computing negative embeddings...")
            neg_texts = [s.text_content for s in neg_stories]
            neg_emb = get_embeddings(neg_texts)

        # Fetch candidates (exclude train set)
        print(f"Fetching {args.candidates} candidates...")
        candidates = await get_best_stories(
            limit=args.candidates,
            exclude_ids=train_ids,
            days=30
        )

        if not candidates:
            print("No candidates fetched")
            return

        # Inject test stories into candidates (simulates them appearing on HN)
        for ts in test_stories:
            if ts.id not in {c.id for c in candidates}:
                candidates.append(ts)

        print(f"Ranking {len(candidates)} candidates...")
        results = rank_stories(
            candidates,
            positive_embeddings=train_emb,
            negative_embeddings=neg_emb,
            use_classifier=args.classifier,
        )

        ranked_ids = [candidates[r.index].id for r in results]

        # Compute metrics at multiple k values
        print(f"\n{'k':<6} {'NDCG@k':<10} {'Hit Rate@k':<12}")
        print("-" * 30)
        for k in [10, 20, 30, 50]:
            n = ndcg_at_k(ranked_ids, test_ids, k)
            h = hit_rate_at_k(ranked_ids, test_ids, k)
            print(f"{k:<6} {n:<10.3f} {h:<12.1%}")

        # Show where test items ranked
        print("\nTest story positions in ranking:")
        for i, sid in enumerate(ranked_ids):
            if sid in test_ids:
                score = results[i].hybrid_score if i < len(results) else 0
                print(f"  #{i+1}: story {sid} (score: {score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
