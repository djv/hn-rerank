from __future__ import annotations
import asyncio
import hashlib
import html
import json
import math
import re
import time
import logging
from pathlib import Path
from datetime import UTC, datetime, timedelta
from typing import Awaitable, Callable, Optional, TypedDict, cast
import httpx

from api.cache_utils import atomic_write_json, evict_old_cache_files
from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    ALGOLIA_MIN_POINTS,
    EXTERNAL_REQUEST_SEMAPHORE,
    MIN_COMMENT_LENGTH,
    MIN_STORY_COMMENTS,
    TOP_COMMENTS_FOR_RANKING,
    TOP_COMMENTS_FOR_UI,
    STORY_CACHE_DIR,
    STORY_CACHE_TTL,
    STORY_CACHE_MAX_FILES,
    CANDIDATE_CACHE_DIR,
    CANDIDATE_CACHE_TTL_SHORT,
    CANDIDATE_CACHE_TTL_LONG,
    CANDIDATE_CACHE_TTL_ARCHIVE,
    RSS_MAX_FEEDS,
    RSS_OPML_URL,
    RSS_PER_FEED_LIMIT,
)
from api.models import Story, StoryDict
from api.rss import fetch_rss_stories
from api.url_utils import normalize_url

logger = logging.getLogger(__name__)

ALGOLIA_BASE: str = "https://hn.algolia.com/api/v1"
SEM: asyncio.Semaphore = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)
CACHE_PATH: Path = Path(STORY_CACHE_DIR)
CACHE_PATH.mkdir(parents=True, exist_ok=True)

CANDIDATE_CACHE_PATH: Path = Path(CANDIDATE_CACHE_DIR)
CANDIDATE_CACHE_PATH.mkdir(parents=True, exist_ok=True)


def get_cached_candidates(
    key: str,
    ttl: int,
    allow_stale: bool = False,
) -> Optional[list[int]]:
    path = CANDIDATE_CACHE_PATH / f"{key}.json"
    if path.exists():
        if allow_stale or time.time() - path.stat().st_mtime < ttl:
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    return None


def save_cached_candidates(key: str, ids: list[int]) -> None:
    path = CANDIDATE_CACHE_PATH / f"{key}.json"
    atomic_write_json(path, ids)


def _clean_text(txt: str) -> Optional[str]:
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
    children: list["AlgoliaComment"]


class AlgoliaItem(TypedDict, total=False):
    type: str
    title: str
    url: str
    points: int
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
    story: StoryDict | None


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
) -> Optional[Story]:
    cache_file: Path = CACHE_PATH / f"{sid}.json"
    if cache_file.exists():
        try:
            data = cast(CachedStory, json.loads(cache_file.read_text()))
            if allow_stale or time.time() - float(data["ts"]) < STORY_CACHE_TTL:
                cached_story = data.get("story")
                return (
                    Story.from_dict(cast(StoryDict, cached_story))
                    if cached_story
                    else None
                )
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
                atomic_write_json(cache_file, {"ts": time.time(), "story": None})
                return None

            item = cast(AlgoliaItem, resp.json())

            # Validate it's a story
            if item.get("type") != "story":
                atomic_write_json(cache_file, {"ts": time.time(), "story": None})
                return None

            title = html.unescape(item.get("title", ""))
            url = item.get("url", "")
            score = item.get("points", 0) or 0
            created_at = item.get("created_at_i", 0) or 0

            # Extract comments from nested structure
            children = item.get("children", [])
            all_comments = _extract_comments_recursive(children)

            # Require minimum comments for meaningful signal
            if len(all_comments) < MIN_STORY_COMMENTS:
                atomic_write_json(cache_file, {"ts": time.time(), "story": None})
                return None

            # Sort by score (position + depth penalty), lower = better
            all_comments.sort(key=lambda x: x["score"])
            selected = all_comments[:TOP_COMMENTS_FOR_RANKING]
            top_for_rank = " ".join([c["text"] for c in selected])
            ui_comments = [c["text"] for c in selected[:TOP_COMMENTS_FOR_UI]]

            # Use natural title weighting
            title_context = f"{title}."
            story = Story(
                id=sid,
                title=title,
                url=url or None,
                score=score,
                time=created_at,
                comments=ui_comments,
                text_content=f"{title_context} {top_for_rank}",
            )
            cache_payload: CachedStory = {"ts": time.time(), "story": story.to_dict()}
            atomic_write_json(cache_file, cache_payload)
            evict_old_cache_files(CACHE_PATH, "*.json", STORY_CACHE_MAX_FILES)
            return story
        except Exception as e:
            # Don't cache transient errors (network, etc)
            logger.debug(f"Failed to fetch story {sid}: {e}")
            return None


async def get_best_stories(
    limit: int,
    exclude_ids: Optional[set[int]] = None,
    exclude_urls: Optional[set[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    days: int = ALGOLIA_DEFAULT_DAYS,
    include_rss: bool = True,
    cache_only: bool = False,
    allow_stale: bool = False,
) -> list[Story]:
    if exclude_ids is None:
        exclude_ids = set()

    # Algolia limits to 1000 results per query, so use time windows
    ALGOLIA_MAX_PER_QUERY = 1000

    # Calculate distribution target per window
    # num_windows calculation removed as it was unused

    hits: set[int] = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        now = datetime.now(UTC)

        # Align anchor to last Monday midnight UTC
        # This makes older windows stable for a full week, maximizing cache reuse
        days_since_monday = now.weekday()
        anchor = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        anchor_ts = int(anchor.timestamp())
        ts_now = int(now.timestamp() // 900 * 900)  # Round to 15m
        cutoff_ts = int((now - timedelta(days=days)).timestamp())
        recent_start = max(anchor_ts, cutoff_ts)

        # Construct windows
        # Tuple: (ts_start, ts_end, is_live_window)
        windows: list[tuple[int, int, bool]] = []

        # 1. Recent Windows: From recent_start to Now
        # Split into daily chunks to reduce refetch volume when Live window expires
        if ts_now > recent_start:
            current_end = ts_now
            while current_end > recent_start:
                prev_midnight = (current_end // 86400) * 86400
                if prev_midnight == current_end:
                    prev_midnight -= 86400

                ts_start = max(prev_midnight, recent_start)
                is_live = current_end == ts_now
                windows.append((ts_start, current_end, is_live))
                current_end = ts_start

        # 2. Archive Windows: 7-day chunks back from Anchor
        if cutoff_ts < anchor_ts:
            current_end = anchor_ts
            # Safety limit to prevent infinite loops if days is huge
            max_archive_weeks = math.ceil(days / 7) + 1

            for _ in range(max_archive_weeks):
                if current_end <= cutoff_ts:
                    break
                current_start = max(current_end - (7 * 86400), cutoff_ts)
                windows.append((current_start, current_end, False))
                current_end = current_start

        # Calculate targets based on duration
        total_duration = sum(end - start for start, end, _ in windows)
        if total_duration == 0:
            total_duration = 1  # prevent div/0

        for ts_start, ts_end, is_live in windows:
            duration = ts_end - ts_start
            # Proportional target, but at least 10% of limit or 20 items
            win_target = math.ceil(limit * (duration / total_duration))
            win_target = max(win_target, 20)

            # For live window, we use a stable key (ignoring ts_end) to respect TTL
            # For archive windows, ts_end is fixed/stable so we keep it
            key_suffix = f"{ts_start}" if is_live else f"{ts_start}-{ts_end}"
            cache_key = hashlib.md5(
                f"{key_suffix}-{ALGOLIA_MIN_POINTS}-{MIN_STORY_COMMENTS}".encode()
            ).hexdigest()

            # TTL Logic:
            # 1. Live window: Short TTL (CANDIDATE_CACHE_TTL_SHORT)
            # 2. Old Archive (>= 30 days): Long/Archive TTL (CANDIDATE_CACHE_TTL_ARCHIVE)
            # 3. Recent Archive: Weekly TTL (7 days)
            if is_live:
                ttl = CANDIDATE_CACHE_TTL_SHORT
            elif ts_end <= int(now.timestamp()) - (30 * 86400):
                ttl = CANDIDATE_CACHE_TTL_ARCHIVE
            else:
                ttl = CANDIDATE_CACHE_TTL_LONG

            cached_ids = get_cached_candidates(cache_key, ttl, allow_stale=allow_stale)
            page_ids: list[int] = []

            # Use cache if valid, even if fewer IDs than target (avoid wasteful refetches)
            if cached_ids is not None:
                page_ids = cached_ids
            elif cache_only:
                logger.info(
                    f"Cache-only: no cached candidates for window {ts_start}-{ts_end}"
                )
                continue
            else:
                # Cache miss or expired - fetch from API
                logger.info(
                    f"Cache miss for window {ts_start}-{ts_end} (live={is_live}). Fetching from Algolia."
                )
                fetch_count = min(max(win_target, 200), ALGOLIA_MAX_PER_QUERY)

                # Construct filters
                # Use >= for start to include boundary stories (prevents gaps between windows)
                filters = [
                    f"created_at_i>={ts_start}",
                    f"points>{ALGOLIA_MIN_POINTS}",
                    f"num_comments>={MIN_STORY_COMMENTS}",
                ]
                # Always add upper bound (< end) to prevent overlap and bound live window
                filters.append(f"created_at_i<{ts_end}")

                params: dict[str, str | int] = {
                    "tags": "story",
                    "numericFilters": ",".join(filters),
                    "hitsPerPage": fetch_count,
                }
                resp: httpx.Response = await client.get(
                    f"{ALGOLIA_BASE}/search", params=params
                )
                if resp.status_code != 200 or not resp.content:
                    continue
                try:
                    data = cast(AlgoliaSearchResponse, resp.json())
                    page_hits = data.get("hits", [])
                    page_ids = [int(h["objectID"]) for h in page_hits]
                    save_cached_candidates(cache_key, page_ids)
                except Exception:
                    continue

            # Take only the required slice from this window
            window_hits = 0
            for oid in page_ids:
                if oid not in exclude_ids and oid not in hits:
                    hits.add(oid)
                    window_hits += 1
                    if window_hits >= win_target:
                        break

            # Stop if we have enough total hits (optional, but good for speed)
            if len(hits) >= limit * 1.5:  # Buffer
                break

        results: list[Story] = []
        if not hits:
            return []

        tasks: list[Awaitable[Optional[Story]]] = [
            fetch_story(client, sid, cache_only=cache_only, allow_stale=allow_stale)
            for sid in hits
        ]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            res: Optional[Story] = await task
            if res:
                # URL-based exclusion for duplicates/hidden items
                if exclude_urls and res.url:
                    norm_url = normalize_url(res.url)
                    if norm_url and norm_url in exclude_urls:
                        continue
                results.append(res)
            if progress_callback:
                progress_callback(i + 1, len(hits))

        rss_stories: list[Story] = []
        if include_rss and not cache_only:
            try:
                rss_stories = await fetch_rss_stories(
                    opml_url=RSS_OPML_URL,
                    days=days,
                    max_feeds=RSS_MAX_FEEDS,
                    per_feed=RSS_PER_FEED_LIMIT,
                    exclude_urls=exclude_urls,
                    fetch_full_content=True,
                )
            except Exception as e:
                logger.warning(f"RSS fetch failed: {e}")

        return results + rss_stories
