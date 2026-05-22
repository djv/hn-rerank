#!/usr/bin/env -S uv run
from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    ALGOLIA_MIN_POINTS,
    MIN_CANDIDATE_COMMENTS,
    TOP_COMMENTS_FOR_RANKING,
    TOP_COMMENTS_FOR_UI,
)
from api.content import ARTICLE_SEM, compose_story_text, fetch_full_text, strip_html
import api.fetching as fetching
from api.fetching import (
    get_best_stories,
)
from api.models import Story
from api.config import AppConfig

DEFAULT_BIGQUERY_PROJECT = "gen-lang-client-0444855014"
HN_FULL_TABLE = "bigquery-public-data.hacker_news.full"


def _clean_bigquery_comment(text: str) -> str:
    text = html.unescape(strip_html(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _timestamp_to_unix(value: Any) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.timestamp())
    return int(value or 0)


def story_from_bigquery_row(row: Any) -> Story | None:
    get = row.get if hasattr(row, "get") else row.__getitem__
    story_id = int(get("id") or 0)
    if story_id <= 0:
        return None

    title = html.unescape(str(get("title") or "")).strip()
    self_text = strip_html(str(get("text") or ""))
    url = str(get("url") or "").strip() or None
    score = int(get("score") or 0)
    created_at = _timestamp_to_unix(get("timestamp") or get("time"))
    comment_count = get("comment_count")
    comments_raw = get("comments") or []

    comments: list[str] = []
    for item in comments_raw:
        if isinstance(item, dict):
            text = str(item.get("text") or "")
        else:
            text = str(getattr(item, "text", "") or "")
        text = _clean_bigquery_comment(text)
        if text:
            comments.append(text)

    text_content = compose_story_text(
        title=title,
        self_text=self_text,
        comments=comments[:TOP_COMMENTS_FOR_RANKING],
    )
    if not text_content:
        return None

    return Story(
        id=story_id,
        title=title,
        url=url,
        score=score,
        time=created_at,
        discussion_url=f"https://news.ycombinator.com/item?id={story_id}",
        comments=comments[:TOP_COMMENTS_FOR_UI],
        text_content=text_content,
        source="hn",
        comment_count=int(comment_count) if comment_count is not None else None,
    )


def build_bigquery_sql(table: str = HN_FULL_TABLE) -> str:
    return f"""
WITH RECURSIVE candidate_stories AS (
  SELECT
    id,
    title,
    url,
    text,
    score,
    timestamp,
    descendants AS comment_count
  FROM `{table}`
  WHERE type = 'story'
    AND timestamp >= TIMESTAMP_SECONDS(@start_ts)
    AND timestamp < TIMESTAMP_SECONDS(@end_ts)
    AND IFNULL(score, 0) > @min_points
    AND (@min_comments <= 0 OR IFNULL(descendants, 0) >= @min_comments)
    AND NOT IFNULL(deleted, FALSE)
    AND NOT IFNULL(dead, FALSE)
  ORDER BY score DESC, timestamp DESC
  LIMIT @candidate_limit
),
descendant_items AS (
  SELECT id AS story_id, id AS item_id, 0 AS depth
  FROM candidate_stories
  UNION ALL
  SELECT d.story_id, child.id AS item_id, d.depth + 1 AS depth
  FROM descendant_items AS d
  JOIN `{table}` AS child
    ON child.parent = d.item_id
  WHERE d.depth < @max_comment_depth
    AND child.type = 'comment'
    AND NOT IFNULL(child.deleted, FALSE)
    AND NOT IFNULL(child.dead, FALSE)
),
ranked_comments AS (
  SELECT
    d.story_id,
    child.text,
    IFNULL(child.score, 0) AS score,
    ROW_NUMBER() OVER (
      PARTITION BY d.story_id
      ORDER BY IFNULL(child.score, 0) DESC, child.timestamp DESC
    ) AS rn
  FROM descendant_items AS d
  JOIN `{table}` AS child
    ON child.id = d.item_id
  WHERE d.depth > 0
    AND child.text IS NOT NULL
    AND child.text != ''
),
comments_agg AS (
  SELECT
    story_id,
    ARRAY_AGG(STRUCT(text, score) ORDER BY score DESC) AS comments
  FROM ranked_comments
  WHERE rn <= @comments_per_story
  GROUP BY story_id
)
SELECT
  cs.id,
  cs.title,
  cs.url,
  cs.text,
  cs.score,
  cs.timestamp,
  cs.comment_count,
  IFNULL(ca.comments, []) AS comments
FROM candidate_stories AS cs
LEFT JOIN comments_agg AS ca
  ON ca.story_id = cs.id
ORDER BY cs.score DESC, cs.timestamp DESC
"""


@dataclass(frozen=True)
class SourceRun:
    source: str
    seconds: float
    story_count: int
    ids: set[int]
    newest_time: int | None
    story_times: dict[int, int]


def summarize_runs(source: str, runs: list[SourceRun]) -> dict[str, Any]:
    seconds = [run.seconds for run in runs]
    newest = [run.newest_time for run in runs if run.newest_time is not None]
    return {
        "source": source,
        "runs": len(runs),
        "seconds": seconds,
        "median_seconds": statistics.median(seconds) if seconds else None,
        "mean_seconds": statistics.fmean(seconds) if seconds else None,
        "story_count": runs[-1].story_count if runs else 0,
        "newest_time": max(newest) if newest else None,
    }


def compare_last_runs(algolia: SourceRun, bigquery: SourceRun) -> dict[str, Any]:
    overlap = algolia.ids & bigquery.ids
    newest_times = [
        t for t in [algolia.newest_time, bigquery.newest_time] if t is not None
    ]
    freshness_lag_seconds = None
    if algolia.newest_time is not None and bigquery.newest_time is not None:
        freshness_lag_seconds = algolia.newest_time - bigquery.newest_time

    fair_algolia_ids = algolia.ids
    fair_bigquery_ids = bigquery.ids
    fair_overlap_count = len(overlap)
    fair_overlap_ratio = len(overlap) / len(algolia.ids) if algolia.ids else 0.0

    if algolia.newest_time is not None and bigquery.newest_time is not None:
        common_max_time = min(algolia.newest_time, bigquery.newest_time)
        fair_algolia_ids = {
            i for i in algolia.ids if algolia.story_times.get(i, 0) <= common_max_time
        }
        fair_bigquery_ids = {
            i for i in bigquery.ids if bigquery.story_times.get(i, 0) <= common_max_time
        }
        fair_overlap = fair_algolia_ids & fair_bigquery_ids
        fair_overlap_count = len(fair_overlap)
        fair_overlap_ratio = (
            len(fair_overlap) / len(fair_algolia_ids) if fair_algolia_ids else 0.0
        )

    return {
        "overlap_count": len(overlap),
        "algolia_only_count": len(algolia.ids - bigquery.ids),
        "bigquery_only_count": len(bigquery.ids - algolia.ids),
        "overlap_ratio_vs_algolia": len(overlap) / len(algolia.ids)
        if algolia.ids
        else 0.0,
        "freshness_lag_seconds": freshness_lag_seconds,
        "newest_time": max(newest_times) if newest_times else None,
        "fair_overlap_count": fair_overlap_count,
        "fair_overlap_ratio_vs_algolia": fair_overlap_ratio,
    }


async def _with_algolia_fetch_overrides(
    fetcher: Callable[[], Any],
    *,
    include_article_fetch: bool,
    use_existing_cache: bool,
) -> Any:
    original_fetch_full_text = fetching.fetch_full_text
    original_cache_path = fetching.CACHE_PATH
    original_candidate_cache_path = fetching.CANDIDATE_CACHE_PATH

    async def empty_article_text(_client: httpx.AsyncClient, _url: str) -> str:
        return ""

    with tempfile.TemporaryDirectory(prefix="hn-rerank-bench-") as tempdir:
        if not include_article_fetch:
            setattr(fetching, "fetch_full_text", empty_article_text)
        if not use_existing_cache:
            cache_root = Path(tempdir)
            fetching.CACHE_PATH = cache_root / "stories"
            fetching.CANDIDATE_CACHE_PATH = cache_root / "candidates"
            fetching.CACHE_PATH.mkdir(parents=True, exist_ok=True)
            fetching.CANDIDATE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        try:
            return await fetcher()
        finally:
            setattr(fetching, "fetch_full_text", original_fetch_full_text)
            fetching.CACHE_PATH = original_cache_path
            fetching.CANDIDATE_CACHE_PATH = original_candidate_cache_path


async def fetch_algolia(
    *,
    limit: int,
    days: int,
    include_article_fetch: bool,
    use_existing_cache: bool,
) -> list[Story]:
    async def run() -> list[Story]:
        config = AppConfig(days=days, no_rss=True)
        return await get_best_stories(limit=limit, config=config)

    return await _with_algolia_fetch_overrides(
        run,
        include_article_fetch=include_article_fetch,
        use_existing_cache=use_existing_cache,
    )


async def enrich_with_article_text(stories: list[Story]) -> list[Story]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        for story in stories:
            if not story.url:
                continue
            async with ARTICLE_SEM:
                article_text = await fetch_full_text(client, story.url)
            if article_text:
                story.text_content = compose_story_text(
                    title=story.title,
                    article_text=article_text,
                    comments=story.comments,
                )
    return stories


async def fetch_bigquery(
    *,
    project: str,
    days: int,
    limit: int,
    max_comment_depth: int,
    comments_per_story: int,
    include_article_fetch: bool,
) -> list[Story]:
    try:
        from google.cloud import bigquery  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Missing google-cloud-bigquery. Install it before running this live "
            "benchmark, for example: uv pip install google-cloud-bigquery"
        ) from exc

    client = bigquery.Client(project=project)
    end_ts = int(datetime.now(UTC).timestamp())
    start_ts = end_ts - days * 86400
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_ts", "INT64", start_ts),
            bigquery.ScalarQueryParameter("end_ts", "INT64", end_ts),
            bigquery.ScalarQueryParameter("candidate_limit", "INT64", limit),
            bigquery.ScalarQueryParameter("min_points", "INT64", ALGOLIA_MIN_POINTS),
            bigquery.ScalarQueryParameter(
                "min_comments", "INT64", MIN_CANDIDATE_COMMENTS
            ),
            bigquery.ScalarQueryParameter(
                "max_comment_depth", "INT64", max_comment_depth
            ),
            bigquery.ScalarQueryParameter(
                "comments_per_story", "INT64", comments_per_story
            ),
        ]
    )
    rows = client.query(build_bigquery_sql(), job_config=job_config).result()
    stories: list[Story] = []
    for row in rows:
        story = story_from_bigquery_row(row)
        if story is not None:
            stories.append(story)
    if include_article_fetch:
        stories = await enrich_with_article_text(stories)
    return stories


async def timed_source(source: str, fetcher: Any) -> SourceRun:
    start = time.perf_counter()
    result = fetcher()
    stories = await result if hasattr(result, "__await__") else result
    elapsed = time.perf_counter() - start
    ids = {story.id for story in stories}
    story_times = {story.id: story.time for story in stories if story.time}
    newest = max((story.time for story in stories if story.time), default=None)
    return SourceRun(
        source=source,
        seconds=elapsed,
        story_count=len(stories),
        ids=ids,
        newest_time=newest,
        story_times=story_times,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Algolia vs BigQuery as HN data sources."
    )
    parser.add_argument("--days", type=int, default=ALGOLIA_DEFAULT_DAYS)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--project",
        default=os.environ.get("HN_RERANK_BIGQUERY_PROJECT", DEFAULT_BIGQUERY_PROJECT),
        help="Google Cloud billing project for public dataset queries.",
    )
    parser.add_argument("--max-comment-depth", type=int, default=4)
    parser.add_argument(
        "--comments-per-story", type=int, default=TOP_COMMENTS_FOR_RANKING
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--include-article-fetch",
        action="store_true",
        help="Include external article extraction for both sources.",
    )
    parser.add_argument(
        "--use-existing-cache",
        action="store_true",
        help="Use normal Algolia story/candidate caches instead of an isolated temp cache.",
    )
    args = parser.parse_args()

    algolia_runs: list[SourceRun] = []
    bigquery_runs: list[SourceRun] = []
    for run_idx in range(args.runs):
        print(f"Run {run_idx + 1}/{args.runs}: Algolia")
        algolia_runs.append(
            await timed_source(
                "algolia",
                lambda: fetch_algolia(
                    limit=args.limit,
                    days=args.days,
                    include_article_fetch=args.include_article_fetch,
                    use_existing_cache=args.use_existing_cache,
                ),
            )
        )
        print(f"Run {run_idx + 1}/{args.runs}: BigQuery")
        bigquery_runs.append(
            await timed_source(
                "bigquery",
                lambda: fetch_bigquery(
                    project=args.project,
                    days=args.days,
                    limit=args.limit,
                    max_comment_depth=args.max_comment_depth,
                    comments_per_story=args.comments_per_story,
                    include_article_fetch=args.include_article_fetch,
                ),
            )
        )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "params": {
            "days": args.days,
            "limit": args.limit,
            "runs": args.runs,
            "project": args.project,
            "max_comment_depth": args.max_comment_depth,
            "comments_per_story": args.comments_per_story,
            "bigquery_table": HN_FULL_TABLE,
            "include_article_fetch": args.include_article_fetch,
            "use_existing_cache": args.use_existing_cache,
        },
        "sources": {
            "algolia": summarize_runs("algolia", algolia_runs),
            "bigquery": summarize_runs("bigquery", bigquery_runs),
        },
        "comparison": compare_last_runs(algolia_runs[-1], bigquery_runs[-1]),
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Saved benchmark report: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
