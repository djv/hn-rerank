from __future__ import annotations
import asyncio
import hashlib
import html
import json
import math
import os
import re
import time
import logging
from pathlib import Path
from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict, cast
from collections.abc import Awaitable, Callable
import httpx

from api.cache_utils import atomic_write_json, evict_old_cache_files
from api.constants import (
    ALGOLIA_MIN_POINTS,
    CANDIDATE_CACHE_VERSION,
    EXTERNAL_REQUEST_SEMAPHORE,
    MIN_COMMENT_LENGTH,
    MIN_CANDIDATE_COMMENTS,
    STORY_CACHE_VERSION,
    TOP_COMMENTS_FOR_RANKING,
    TOP_COMMENTS_FOR_UI,
    STORY_CACHE_DIR,
    STORY_CACHE_TTL,
    STORY_CACHE_MAX_FILES,
    CANDIDATE_CACHE_DIR,
    CANDIDATE_CACHE_TTL_SHORT,
    CANDIDATE_CACHE_TTL_LONG,
    RSS_MAX_FEEDS,
    RSS_OPML_URL,
    RSS_PER_FEED_LIMIT,
)
from api.content import ARTICLE_SEM, compose_story_text, fetch_full_text, strip_html
from api.models import Story, StoryDict
from api.rss import fetch_rss_stories
from api.url_utils import normalize_url
from api.config import AppConfig

logger = logging.getLogger(__name__)

ALGOLIA_BASE: str = "https://hn.algolia.com/api/v1"
HN_FULL_TABLE: str = "bigquery-public-data.hacker_news.full"
HN_BIGQUERY_PROJECT_ENV: str = "HN_RERANK_BIGQUERY_PROJECT"
HN_LIVE_WINDOW_DAYS: int = 4
HN_BIGQUERY_MAX_COMMENT_DEPTH: int = 4
HN_BIGQUERY_ARCHIVE_STORY_CACHE_TTL: int = CANDIDATE_CACHE_TTL_LONG
SEM: asyncio.Semaphore = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)
CACHE_PATH: Path = Path(STORY_CACHE_DIR)
CACHE_PATH.mkdir(parents=True, exist_ok=True)

CANDIDATE_CACHE_PATH: Path = Path(CANDIDATE_CACHE_DIR)
CANDIDATE_CACHE_PATH.mkdir(parents=True, exist_ok=True)


def get_cached_candidates(
    key: str,
    ttl: int,
    allow_stale: bool = False,
) -> list[int] | None:
    path = CANDIDATE_CACHE_PATH / f"{key}.json"
    if path.exists() and (allow_stale or time.time() - path.stat().st_mtime < ttl):
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def save_cached_candidates(key: str, ids: list[int]) -> None:
    path = CANDIDATE_CACHE_PATH / f"{key}.json"
    atomic_write_json(path, ids)


def _clean_text(txt: str) -> str | None:
    """
    Cleans comment text. Returns None if text should be discarded.
    """
    # 0. Decode HTML entities (&#x27; -> ', &#x2F; -> /, etc.)
    txt = html.unescape(txt)
    # 1. Braille/Symbol Blocks
    txt_clean = re.sub(r"[\u2800-\u28FF\u2500-\u27BF]+", "", txt)
    # 2. Excessive Punctuation
    txt_clean = re.sub(r"[#*^\\/\\|\\-_+]{3,}", "", txt_clean)

    if len(txt_clean) == 0:
        return None

    # 3. Alphanumeric Ratio Check
    alnum_count = sum(c.isalnum() for c in txt_clean)
    if len(txt_clean) > 0 and (alnum_count / len(txt_clean)) < 0.5:
        return None

    if len(txt_clean.strip()) <= 20:
        return None

    return txt_clean


class AlgoliaComment(TypedDict, total=False):
    type: str
    points: int
    text: str
    children: list[AlgoliaComment]


class AlgoliaItem(TypedDict, total=False):
    type: str
    title: str
    url: str
    text: str
    story_text: str
    points: int
    num_comments: int
    created_at_i: int
    children: list[AlgoliaComment]


class AlgoliaSearchHit(TypedDict):
    objectID: str


class AlgoliaSearchResponse(TypedDict, total=False):
    hits: list[AlgoliaSearchHit]


class CommentScore(TypedDict):
    text: str
    score: int


class CachedStory(TypedDict):
    ts: float
    version: str
    story: StoryDict | None


def _load_cached_story(
    sid: int,
    *,
    ttl: int,
    allow_stale: bool = False,
    require_comment_count: bool = False,
) -> Story | None:
    cache_file = CACHE_PATH / f"{sid}.json"
    if not cache_file.exists():
        return None
    try:
        data = cast(CachedStory, json.loads(cache_file.read_text()))
        if data.get("version") != STORY_CACHE_VERSION:
            return None
        if not allow_stale and time.time() - float(data["ts"]) >= ttl:
            return None
        cached_story = data.get("story")
        if not cached_story:
            return None
        story_dict = cast(StoryDict, cached_story)
        if (
            require_comment_count
            and story_dict.get("source") == "hn"
            and "comment_count" not in story_dict
        ):
            return None
        return Story.from_dict(story_dict)
    except Exception as e:
        logger.debug(f"Failed to load story cache {cache_file}: {e}")
        return None


def _save_cached_story(story: Story | None) -> None:
    sid = story.id if story is not None else None
    if sid is None:
        return
    cache_file = CACHE_PATH / f"{sid}.json"
    cache_payload = {
        "ts": time.time(),
        "version": STORY_CACHE_VERSION,
        "story": story.to_dict() if story else None,
    }

    atomic_write_json(cache_file, cache_payload)
    evict_old_cache_files(CACHE_PATH, "*.json", STORY_CACHE_MAX_FILES)


def _extract_comments_recursive(
    children: list[AlgoliaComment],
    depth: int = 0,
    max_depth: int = 3,
    parent_points: int = 0,
) -> list[CommentScore]:
    """
    Extract comments from Algolia's nested children structure.

    Returns comments ranked by points (higher = better).
    Uses parent's points for replies to maintain thread coherence.
    """
    DEPTH_PENALTY = 50  # Points penalty per depth level
    results: list[CommentScore] = []

    for child in children:
        if child.get("type") != "comment":
            continue

        # Use comment points, fallback to parent's points for replies
        points = child.get("points") or 0
        if depth > 0 and points == 0:
            points = parent_points

        # Score: higher points = lower score (for sorting), depth penalty
        score = -points + depth * DEPTH_PENALTY

        text = child.get("text", "")
        if text:
            # Strip HTML tags
            clean = re.sub(r"<[^>]+>", " ", text)
            clean = _clean_text(clean)
            # Filter short comments (low value)
            if clean and len(clean) >= MIN_COMMENT_LENGTH:
                results.append({"text": clean, "score": score})

        # Recurse into replies (limited depth to avoid deep threads)
        if depth < max_depth and child.get("children"):
            results.extend(
                _extract_comments_recursive(
                    child["children"], depth + 1, max_depth, parent_points=points
                )
            )
    return results


async def fetch_story(
    client: httpx.AsyncClient,
    sid: int,
    cache_only: bool = False,
    allow_stale: bool = False,
) -> Story | None:
    cache_file: Path = CACHE_PATH / f"{sid}.json"

    def cache_missing_story() -> None:
        atomic_write_json(
            cache_file,
            {
                "ts": time.time(),
                "version": STORY_CACHE_VERSION,
                "story": None,
            },
        )

    if cache_file.exists():
        try:
            data = cast(CachedStory, json.loads(cache_file.read_text()))
            if data.get("version") == STORY_CACHE_VERSION and (
                allow_stale or time.time() - float(data["ts"]) < STORY_CACHE_TTL
            ):
                cached_story = data.get("story")
                if cached_story:
                    story_dict = cast(StoryDict, cached_story)
                    if (
                        not cache_only
                        and story_dict.get("source") == "hn"
                        and "comment_count" not in story_dict
                    ):
                        logger.debug(
                            "Refetching story %s cache missing comment_count", sid
                        )
                    else:
                        return Story.from_dict(story_dict)
                else:
                    return None
        except Exception as e:
            logger.debug(f"Failed to load story cache {cache_file}: {e}")

    if cache_only:
        return None

    async with SEM:
        try:
            logger.debug(f"Fetching story {sid} details from Algolia")
            # Use Algolia API instead of scraping HN (avoids rate limits)
            resp: httpx.Response = await client.get(f"{ALGOLIA_BASE}/items/{sid}")
            if resp.status_code != 200:
                cache_missing_story()
                return None

            item = cast(AlgoliaItem, resp.json())

            # Validate it's a story
            if item.get("type") != "story":
                cache_missing_story()
                return None

            title = html.unescape(item.get("title", ""))
            url = item.get("url", "")
            score = item.get("points", 0) or 0
            comment_count = item.get("num_comments")
            created_at = item.get("created_at_i", 0) or 0
            story_text = strip_html(
                str(item.get("story_text") or item.get("text") or "")
            )

            # Extract comments from nested structure
            children = item.get("children", [])
            all_comments = _extract_comments_recursive(children)

            # Sort by score (position + depth penalty), lower = better
            all_comments.sort(key=lambda x: x["score"])
            selected = all_comments[:TOP_COMMENTS_FOR_RANKING]
            ui_comments = [c["text"] for c in selected[:TOP_COMMENTS_FOR_UI]]

            article_text = ""
            if url:
                async with ARTICLE_SEM:
                    article_text = await fetch_full_text(client, url)

            text_content = compose_story_text(
                title=title,
                self_text=story_text,
                article_text=article_text,
                comments=[c["text"] for c in selected],
            )

            if not text_content:
                cache_missing_story()
                return None

            story = Story(
                id=sid,
                title=title,
                url=url or None,
                score=score,
                time=created_at,
                discussion_url=f"https://news.ycombinator.com/item?id={sid}",
                comments=ui_comments,
                text_content=text_content,
                source="hn",
                comment_count=comment_count
                if comment_count is not None
                else len(all_comments),
            )
            cache_payload: CachedStory = {
                "ts": time.time(),
                "version": STORY_CACHE_VERSION,
                "story": story.to_dict(),
            }
            atomic_write_json(cache_file, cache_payload)
            evict_old_cache_files(CACHE_PATH, "*.json", STORY_CACHE_MAX_FILES)
            return story
        except Exception as e:
            # Don't cache transient errors (network, etc)
            logger.debug(f"Failed to fetch story {sid}: {e}")
            return None


def build_candidate_filters(ts_start: int, ts_end: int) -> list[str]:
    filters = [
        f"created_at_i>={ts_start}",
        f"points>{ALGOLIA_MIN_POINTS}",
    ]
    if MIN_CANDIDATE_COMMENTS > 0:
        filters.append(f"num_comments>={MIN_CANDIDATE_COMMENTS}")
    filters.append(f"created_at_i<{ts_end}")
    return filters


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


def build_bigquery_sql(table: str = HN_FULL_TABLE, has_exclude_ids: bool = False) -> str:
    exclude_clause = "AND id NOT IN UNNEST(@exclude_ids)" if has_exclude_ids else ""
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
    {exclude_clause}
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
    child.timestamp,
    ROW_NUMBER() OVER (
      PARTITION BY d.story_id
      ORDER BY IFNULL(child.score, 0) DESC, child.timestamp ASC
    ) AS rn
  FROM descendant_items AS d
  JOIN `{table}` AS child
    ON child.id = d.item_id
  WHERE d.depth > 0
    AND child.text IS NOT NULL
)
SELECT
  cs.id,
  cs.title,
  cs.url,
  cs.text,
  cs.score,
  cs.timestamp,
  cs.comment_count,
  ARRAY_AGG(
    IF(rc.rn <= @comments_per_story, STRUCT(rc.text AS text, rc.score AS score), NULL)
    IGNORE NULLS
    ORDER BY rc.score DESC
  ) AS comments
FROM candidate_stories AS cs
LEFT JOIN ranked_comments AS rc
  ON rc.story_id = cs.id
GROUP BY cs.id, cs.title, cs.url, cs.text, cs.score, cs.timestamp, cs.comment_count
ORDER BY cs.score DESC, cs.timestamp DESC
"""


def _query_bigquery_archive_sync(
    *,
    start_ts: int,
    end_ts: int,
    candidate_limit: int,
    exclude_ids: list[int] | None = None,
) -> list[Story]:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "BigQuery archive fetching requires google-cloud-bigquery"
        ) from exc

    project = os.environ.get(HN_BIGQUERY_PROJECT_ENV) or None
    client = bigquery.Client(project=project)
    
    query_parameters = [
        bigquery.ScalarQueryParameter("start_ts", "INT64", start_ts),
        bigquery.ScalarQueryParameter("end_ts", "INT64", end_ts),
        bigquery.ScalarQueryParameter("candidate_limit", "INT64", candidate_limit),
        bigquery.ScalarQueryParameter("min_points", "INT64", ALGOLIA_MIN_POINTS),
        bigquery.ScalarQueryParameter(
            "min_comments", "INT64", MIN_CANDIDATE_COMMENTS
        ),
        bigquery.ScalarQueryParameter(
            "max_comment_depth", "INT64", HN_BIGQUERY_MAX_COMMENT_DEPTH
        ),
        bigquery.ScalarQueryParameter(
            "comments_per_story", "INT64", TOP_COMMENTS_FOR_RANKING
        ),
    ]
    if exclude_ids:
        query_parameters.append(
            bigquery.ArrayQueryParameter("exclude_ids", "INT64", exclude_ids)
        )

    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    try:
        sql = build_bigquery_sql(has_exclude_ids=bool(exclude_ids))
        rows = client.query(sql, job_config=job_config).result()
    except Exception as e:
        logger.warning(f"BigQuery archive fetching failed (possibly quota exceeded): {e}")
        return []
    stories: list[Story] = []
    for row in rows:
        story = story_from_bigquery_row(row)
        if story is not None:
            stories.append(story)
    return stories


async def fetch_bigquery_archive_stories(
    *,
    http_client: httpx.AsyncClient,
    start_ts: int,
    end_ts: int,
    candidate_limit: int,
    exclude_ids: set[int],
    exclude_urls: set[str] | None,
    seen_urls: set[str] | None = None,
    cache_only: bool = False,
    allow_stale: bool = False,
) -> list[Story]:
    if cache_only:
        return []

    # Identify stories already in cache to avoid re-fetching them from BigQuery
    # This saves quota by excluding these IDs from the query
    cached_in_window: list[Story] = []
    
    # Simple heuristic: scan cache for stories in this time window
    # In a large cache this might be slow, but for typical use it's fine
    try:
        # Sort by mtime to check newest files first
        cache_files = sorted(
            CACHE_PATH.glob("*.json"), 
            key=lambda p: p.stat().st_mtime, 
            reverse=True
        )
        for p in cache_files:
            try:
                # We use the filename as ID to avoid reading the whole file if it's out of window
                sid = int(p.stem)
                if sid in exclude_ids:
                    continue
                
                # Check file mtime as a coarse filter before reading (1 day buffer)
                if p.stat().st_mtime < start_ts - 86400:
                    # Since we sorted by mtime, we can stop early
                    break

                story = _load_cached_story(
                    sid,
                    ttl=HN_BIGQUERY_ARCHIVE_STORY_CACHE_TTL,
                    allow_stale=allow_stale,
                )
                if story and start_ts <= story.time < end_ts:
                    # Standard filters
                    if story.score <= ALGOLIA_MIN_POINTS:
                        continue
                    if MIN_CANDIDATE_COMMENTS > 0 and (story.comment_count or 0) < MIN_CANDIDATE_COMMENTS:
                        continue
                    
                    if exclude_urls and story.url:
                        norm_url = normalize_url(story.url)
                        if norm_url and norm_url in exclude_urls:
                            continue
                    if seen_urls and story.url:
                        norm_url = normalize_url(story.url)
                        if norm_url and norm_url in seen_urls:
                            continue
                            
                    cached_in_window.append(story)
                    if len(cached_in_window) >= candidate_limit:
                        break
            except (ValueError, OSError):
                continue
    except Exception as e:
        logger.warning(f"Error scanning cache for BigQuery optimization: {e}")

    # Combine all known IDs to exclude from the BigQuery query
    # We prioritize recent IDs by sorting descending (HN IDs are chronological)
    ids_to_exclude = sorted(list(exclude_ids | {s.id for s in cached_in_window}), reverse=True)
    
    remaining_limit = candidate_limit
    stories: list[Story] = []
    
    # Cap the exclude list size to avoid giant queries
    # BigQuery can handle large arrays, but 10k is a safe middle ground for performance
    bq_exclude = ids_to_exclude[:10000] if len(ids_to_exclude) > 10000 else ids_to_exclude
    
    try:
        stories = await asyncio.to_thread(
            _query_bigquery_archive_sync,
            start_ts=start_ts,
            end_ts=end_ts,
            candidate_limit=remaining_limit,
            exclude_ids=bq_exclude,
        )
    except Exception as e:
        logger.warning(f"BigQuery archive fetching failed: {e}")
        # stories remains empty, fallback to cached only

    results: list[Story] = cached_in_window
    seen: set[int] = {s.id for s in cached_in_window}
    
    for story in stories:
        if story.id in seen or story.id in exclude_ids:
            continue
        
        # New story from BQ, already has full content
        results.append(story)
        seen.add(story.id)
        if story.url and seen_urls is not None:
            norm_url = normalize_url(story.url)
            if norm_url:
                seen_urls.add(norm_url)
        
        # Save to cache
        from api.cache_utils import atomic_write_json
        cache_file = CACHE_PATH / f"{story.id}.json"
        atomic_write_json(
            cache_file,
            {
                "ts": time.time(),
                "version": STORY_CACHE_VERSION,
                "story": story.to_dict(),
            },
        )

    # Sort by score and time to ensure the best candidates are kept regardless of provenance
    results.sort(key=lambda s: (s.score, s.time), reverse=True)
    return results[:candidate_limit]


async def get_best_stories(
    limit: int,
    exclude_ids: set[int] | None = None,
    exclude_urls: set[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    config: AppConfig = AppConfig(),
    cache_only: bool = False,
    allow_stale: bool = False,
    now_ts: int | None = None,
) -> list[Story]:
    if exclude_ids is None:
        exclude_ids = set()

    days = config.days
    include_rss = not config.no_rss

    ALGOLIA_MAX_PER_QUERY = 1000

    async with httpx.AsyncClient(timeout=30.0) as client:
        now_dt = datetime.fromtimestamp(now_ts, UTC) if now_ts else datetime.now(UTC)
        ts_now = int(now_dt.timestamp() // 900 * 900)  # Round to 15m
        cutoff_ts = int((now_dt - timedelta(days=days)).timestamp())
        live_start_ts = int((now_dt - timedelta(days=HN_LIVE_WINDOW_DAYS)).timestamp())
        algolia_start_ts = max(cutoff_ts, live_start_ts)
        archive_end_ts = min(live_start_ts, ts_now)
        has_archive_window = cutoff_ts < archive_end_ts

        candidate_budget = max(limit, math.ceil(limit * 1.5))
        lookback_seconds = max(ts_now - cutoff_ts, 1)
        live_seconds = max(ts_now - algolia_start_ts, 0)
        archive_seconds = max(archive_end_ts - cutoff_ts, 0)

        live_budget = 0
        if live_seconds > 0:
            live_budget = math.ceil(
                candidate_budget * (live_seconds / lookback_seconds)
            )
            live_budget = max(live_budget, limit // 4, 50)
            live_budget = min(live_budget, candidate_budget)

        archive_budget = 0
        if has_archive_window:
            archive_budget = max(candidate_budget - live_budget, 50)
            if archive_seconds > 0:
                archive_budget = max(
                    archive_budget,
                    math.ceil(candidate_budget * (archive_seconds / lookback_seconds)),
                )

        hits: set[int] = set()

        # Algolia live window: daily chunks over the last 4 days.
        live_windows: list[tuple[int, int, bool]] = []
        if ts_now > algolia_start_ts and live_budget > 0:
            current_end = ts_now
            while current_end > algolia_start_ts:
                prev_midnight = (current_end // 86400) * 86400
                if prev_midnight == current_end:
                    prev_midnight -= 86400

                ts_start = max(prev_midnight, algolia_start_ts)
                is_live = current_end == ts_now
                live_windows.append((ts_start, current_end, is_live))
                current_end = ts_start

        total_duration = sum(end - start for start, end, _ in live_windows)
        if total_duration == 0:
            total_duration = 1

        for ts_start, ts_end, is_live in live_windows:
            remaining_budget = live_budget - len(hits)
            if remaining_budget <= 0:
                break

            duration = ts_end - ts_start
            win_target = math.ceil(live_budget * (duration / total_duration))
            if is_live:
                win_target = max(win_target, limit // 4)
            win_target = max(win_target, 50)
            win_target = min(win_target, remaining_budget)

            key_suffix = f"{ts_start}" if is_live else f"{ts_start}-{ts_end}"
            cache_key = hashlib.md5(
                f"{CANDIDATE_CACHE_VERSION}-{key_suffix}-{ALGOLIA_MIN_POINTS}-{MIN_CANDIDATE_COMMENTS}".encode()
            ).hexdigest()

            ttl = CANDIDATE_CACHE_TTL_SHORT if is_live else CANDIDATE_CACHE_TTL_LONG
            cached_ids = get_cached_candidates(cache_key, ttl, allow_stale=allow_stale)
            page_ids: list[int] = []

            if cached_ids is not None and len(cached_ids) >= win_target:
                page_ids = cached_ids
            elif cache_only:
                if cached_ids:
                    page_ids = cached_ids
                else:
                    logger.info(
                        f"Cache-only: no cached candidates for window {ts_start}-{ts_end}"
                    )
                    continue
            else:
                logger.info(
                    f"Fetching Algolia live window {ts_start}-{ts_end} "
                    f"(target={win_target})."
                )
                fetch_count = ALGOLIA_MAX_PER_QUERY

                filters = build_candidate_filters(ts_start, ts_end)

                params: dict[str, str | int] = {
                    "tags": "story",
                    "numericFilters": ",".join(filters),
                    "hitsPerPage": fetch_count,
                }
                resp: httpx.Response = await client.get(
                    f"{ALGOLIA_BASE}/search", params=params
                )
                if resp.status_code != 200 or not resp.content:
                    # Fallback to whatever we had in cache if API fails
                    if cached_ids:
                        page_ids = cached_ids
                    continue
                try:
                    data = cast(AlgoliaSearchResponse, resp.json())
                    page_hits = data.get("hits", [])
                    page_ids = [int(h["objectID"]) for h in page_hits]
                    save_cached_candidates(cache_key, page_ids)
                except Exception:
                    if cached_ids:
                        page_ids = cached_ids
                    continue

            # Take only the required slice from this window
            window_hits = 0
            for oid in page_ids:
                if oid not in exclude_ids and oid not in hits:
                    hits.add(oid)
                    window_hits += 1
                    if window_hits >= win_target:
                        break

        results: list[Story] = []
        if hits:
            tasks: list[Awaitable[Story | None]] = [
                fetch_story(client, sid, cache_only=cache_only, allow_stale=allow_stale)
                for sid in hits
            ]
            for i, task in enumerate(asyncio.as_completed(tasks)):
                res: Story | None = await task
                if res:
                    if exclude_urls and res.url:
                        norm_url = normalize_url(res.url)
                        if norm_url and norm_url in exclude_urls:
                            continue
                    results.append(res)
                if progress_callback:
                    progress_callback(i + 1, len(hits))

        if has_archive_window and not cache_only:
            seen_urls = {
                norm_url
                for story in results
                if story.url
                for norm_url in [normalize_url(story.url)]
                if norm_url
            }
            archive_stories = await fetch_bigquery_archive_stories(
                http_client=client,
                start_ts=cutoff_ts,
                end_ts=archive_end_ts,
                candidate_limit=archive_budget,
                exclude_ids=exclude_ids | {story.id for story in results},
                exclude_urls=exclude_urls,
                seen_urls=seen_urls,
                cache_only=cache_only,
                allow_stale=allow_stale,
            )
            results.extend(archive_stories)

        if not results and not include_rss:
            return []

        rss_stories: list[Story] = []
        if include_rss and not cache_only:
            try:
                # Use progress callback for RSS fetching too
                rss_stories = await fetch_rss_stories(
                    opml_url=RSS_OPML_URL,
                    days=days,
                    max_feeds=RSS_MAX_FEEDS,
                    per_feed=RSS_PER_FEED_LIMIT,
                    exclude_urls=exclude_urls,
                    fetch_full_content=True,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.warning(f"RSS fetch failed: {e}")

        return results + rss_stories
