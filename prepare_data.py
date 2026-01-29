#!/usr/bin/env -S uv run
"""Prepare training data for embedding fine-tuning using user's signals."""

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx
from rich.console import Console

from api.client import HNClient
from api.fetching import fetch_story

console = Console()


async def prepare_data(username: str, limit: int = 300) -> None:
    """Prepare contrastive training data from user's upvoted and hidden stories."""

    console.print(f"[cyan]Fetching signals for @{username}...[/cyan]")

    client = HNClient()
    user_data = await client.fetch_user_data(username)

    pos_ids = list(user_data.get("pos", set()) | user_data.get("upvoted", set()))
    neg_ids = list(user_data.get("hidden", set()))

    console.print(f"Found {len(pos_ids)} positive, {len(neg_ids)} negative signals")

    if len(pos_ids) < 20:
        console.print("[red]Need at least 20 positive signals for training.[/red]")
        return

    # Fetch story details
    pos_stories = []
    neg_stories = []

    async with httpx.AsyncClient(timeout=30.0) as http:
        console.print("[cyan]Fetching positive stories...[/cyan]")
        for sid in pos_ids[:limit]:
            story = await fetch_story(http, sid)
            if story and story.text_content and len(story.text_content) > 100:
                pos_stories.append(story)

        console.print("[cyan]Fetching negative stories...[/cyan]")
        for sid in neg_ids[:limit]:
            story = await fetch_story(http, sid)
            if story and story.text_content and len(story.text_content) > 100:
                neg_stories.append(story)

    console.print(f"Loaded {len(pos_stories)} positive, {len(neg_stories)} negative stories")

    if len(pos_stories) < 20:
        console.print("[red]Not enough positive stories with content.[/red]")
        return

    # Strategy 1: Positive pairs (anchor, positive)
    # For each positive story, pair with another positive story (similar interests)
    positive_pairs = []
    for i, anchor in enumerate(pos_stories):
        # Find another positive story as the positive example
        others = [s for j, s in enumerate(pos_stories) if j != i]
        if others:
            pos_example = random.choice(others)
            positive_pairs.append({
                "anchor": anchor.text_content[:2000],
                "positive": pos_example.text_content[:2000],
            })

    # Strategy 2: Triplets (anchor, positive, negative) for hard negative mining
    triplets = []
    if neg_stories:
        for anchor in pos_stories:
            pos_example = random.choice([s for s in pos_stories if s.id != anchor.id])
            neg_example = random.choice(neg_stories)
            triplets.append({
                "anchor": anchor.text_content[:2000],
                "positive": pos_example.text_content[:2000],
                "negative": neg_example.text_content[:2000],
            })

    # Shuffle and split
    random.shuffle(positive_pairs)
    random.shuffle(triplets)

    # Use 90% train, 10% val
    split_pairs = int(len(positive_pairs) * 0.9)
    split_triplets = int(len(triplets) * 0.9)

    train_pairs = positive_pairs[:split_pairs]
    val_pairs = positive_pairs[split_pairs:]

    train_triplets = triplets[:split_triplets]
    val_triplets = triplets[split_triplets:]

    # Save data
    Path("train_pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in train_pairs)
    )
    Path("val_pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in val_pairs)
    )
    Path("train_triplets.jsonl").write_text(
        "\n".join(json.dumps(t) for t in train_triplets)
    )
    Path("val_triplets.jsonl").write_text(
        "\n".join(json.dumps(t) for t in val_triplets)
    )

    console.print(f"[green]Saved {len(train_pairs)} training pairs[/green]")
    console.print(f"[green]Saved {len(train_triplets)} training triplets[/green]")
    console.print(f"[green]Saved {len(val_pairs)} validation pairs[/green]")
    console.print(f"[green]Saved {len(val_triplets)} validation triplets[/green]")

    # Also save legacy format for backward compatibility
    legacy_train = [{"query": p["anchor"][:500], "pos": p["positive"][:500]} for p in train_pairs]
    legacy_val = [{"query": p["anchor"][:500], "pos": p["positive"][:500]} for p in val_pairs]

    Path("train.jsonl").write_text("\n".join(json.dumps(p) for p in legacy_train))
    Path("val.jsonl").write_text("\n".join(json.dumps(p) for p in legacy_val))

    console.print("[green]Legacy format saved to train.jsonl and val.jsonl[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data")
    parser.add_argument("username", help="HN username")
    parser.add_argument("--limit", type=int, default=300, help="Max stories per signal type")
    args = parser.parse_args()

    asyncio.run(prepare_data(args.username, args.limit))
