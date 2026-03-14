#!/usr/bin/env -S uv run
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, UTC
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_quality import RankingEvaluator  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a benchmark dataset snapshot for reproducible ranking experiments."
    )
    parser.add_argument("username", help="HN username to snapshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Snapshot path (defaults under runs/benchmarks)",
    )
    parser.add_argument("--holdout", type=float, default=0.2, help="Holdout fraction")
    parser.add_argument("--candidates", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--limit-pos", type=int, default=200, help="Max positive stories")
    parser.add_argument("--limit-neg", type=int, default=100, help="Max hidden stories")
    parser.add_argument("--classifier", action="store_true", help="Include hidden-story negatives")
    parser.add_argument("--recency", action="store_true", help="Persist recency weights")
    parser.add_argument("--cache-only", action="store_true", help="Use cached data only")
    return parser


def default_output_path(username: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return Path("runs/benchmarks") / f"{username}_benchmark_{stamp}.json"


async def main() -> int:
    args = build_parser().parse_args()
    output = args.output or default_output_path(args.username)

    evaluator = RankingEvaluator(args.username)
    ok = await evaluator.load_data(
        holdout=args.holdout,
        limit_pos=args.limit_pos,
        limit_neg=args.limit_neg,
        candidate_count=args.candidates,
        use_classifier=args.classifier,
        use_recency=args.recency,
        cache_only=args.cache_only,
        allow_stale=args.cache_only,
    )
    if not ok:
        return 1

    path = evaluator.save_snapshot(
        output,
        metadata={
            "snapshot_name": output.stem,
        },
    )
    if evaluator.dataset is None:
        return 1

    dataset = evaluator.dataset
    print(f"Saved snapshot: {path}")
    print(f"  train stories: {len(dataset.train_stories)}")
    print(f"  test stories: {len(dataset.test_stories)}")
    print(f"  negative stories: {len(dataset.neg_stories)}")
    print(f"  candidates: {len(dataset.candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
