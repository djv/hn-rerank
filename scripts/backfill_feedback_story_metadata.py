#!/usr/bin/env -S uv run
"""Backfill raw story score/comment metadata for dashboard feedback records."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

import httpx

from api.feedback import FEEDBACK_STORE_PATH, FeedbackRecord, load_feedback, save_feedback
from api.fetching import fetch_story


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill score/comment_count for dashboard feedback records."
    )
    parser.add_argument("--path", type=Path, default=FEEDBACK_STORE_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Max records to update (0 = all)")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _needs_backfill(record: FeedbackRecord) -> bool:
    return record.score is None or record.comment_count is None


async def main() -> None:
    args = parse_args()
    records = load_feedback(args.path)
    pending = [record for record in records.values() if _needs_backfill(record)]
    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print("No feedback records need metadata backfill.")
        return

    updated = 0
    unresolved = 0
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for record in pending:
            next_score = record.score
            next_comment_count = record.comment_count
            if record.source == "hn" and record.id > 0:
                story = await fetch_story(
                    client,
                    record.id,
                    cache_only=args.cache_only,
                    allow_stale=args.allow_stale,
                )
                if story is not None:
                    next_score = story.score
                    next_comment_count = story.comment_count if story.comment_count is not None else 0
            else:
                if next_score is None:
                    next_score = 0
                if next_comment_count is None:
                    next_comment_count = 0

            if next_score is None or next_comment_count is None:
                unresolved += 1
                continue

            records[record.key] = replace(
                record,
                score=int(next_score),
                comment_count=int(next_comment_count),
            )
            updated += 1

    print(
        f"Backfill complete: updated={updated}, unresolved={unresolved}, "
        f"dry_run={args.dry_run}"
    )
    if not args.dry_run and updated > 0:
        save_feedback(records, args.path)
        print(f"Saved feedback store: {args.path}")


if __name__ == "__main__":
    asyncio.run(main())
