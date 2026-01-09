from __future__ import annotations
import asyncio
import html
import json
import os
import re
import time
from pathlib import Path
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Optional
import httpx

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
)

ALGOLIA_BASE: str = "https://hn.algolia.com/api/v1"
SEM: asyncio.Semaphore = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)
CACHE_PATH: Path = Path(STORY_CACHE_DIR)
CACHE_PATH.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically using temp file + rename."""
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _evict_old_cache_files() -> None:
    """Remove oldest cache files if over STORY_CACHE_MAX_FILES limit (LRU)."""
    cache_files = list(CACHE_PATH.glob("*.json"))
    if len(cache_files) <= STORY_CACHE_MAX_FILES:
        return
    # Sort by modification time (oldest first)
    cache_files.sort(key=lambda p: p.stat().st_mtime)
    # Remove oldest files to get under limit
    for f in cache_files[: len(cache_files) - STORY_CACHE_MAX_FILES]:
        try:
            f.unlink()
        except OSError:
            pass


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


def _extract_comments_recursive(
    children: list[dict[str, Any]],
    depth: int = 0,
    max_depth: int = 3,
    parent_points: int = 0,
) -> list[dict[str, Any]]:
    """
    Extract comments from Algolia's nested children structure.

    Returns comments ranked by points (higher = better).
    Uses parent's points for replies to maintain thread coherence.
    """
    DEPTH_PENALTY = 50  # Points penalty per depth level
    results: list[dict[str, Any]] = []

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


async def fetch_story(client: httpx.AsyncClient, sid: int) -> Optional[dict[str, Any]]:
    cache_file: Path = CACHE_PATH / f"{sid}.json"
    if cache_file.exists():
        try:
            data: dict[str, Any] = json.loads(cache_file.read_text())
            if time.time() - float(data["ts"]) < STORY_CACHE_TTL:
                return data.get("story")  # Returns None if cached as None
        except Exception:
            pass

    async with SEM:
        try:
            # Use Algolia API instead of scraping HN (avoids rate limits)
            resp: httpx.Response = await client.get(f"{ALGOLIA_BASE}/items/{sid}")
            if resp.status_code != 200:
                return None

            item = resp.json()

            # Validate it's a story
            if item.get("type") != "story":
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
                return None

            # Sort by score (position + depth penalty), lower = better
            all_comments.sort(key=lambda x: x["score"])
            selected = all_comments[:TOP_COMMENTS_FOR_RANKING]
            top_for_rank = " ".join([c["text"] for c in selected])
            ui_comments = [c["text"] for c in selected[:TOP_COMMENTS_FOR_UI]]

            # Repeat title 4x to weight it higher than comments in embeddings
            title_weighted = f"{title}. {title}. {title}. {title}."
            story: dict[str, Any] = {
                "id": sid,
                "title": title,
                "url": url,
                "score": score,
                "time": created_at,
                "comments": ui_comments,
                "text_content": f"{title_weighted} {top_for_rank}",
            }
            _atomic_write_json(cache_file, {"ts": time.time(), "story": story})
            _evict_old_cache_files()
            return story
        except Exception:
            return None


async def get_best_stories(
    limit: int,
    exclude_ids: Optional[set[int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    days: int = ALGOLIA_DEFAULT_DAYS,
) -> list[dict[str, Any]]:
    if exclude_ids is None:
        exclude_ids = set()

    # Algolia limits to 1000 results per query, so use time windows
    ALGOLIA_MAX_PER_QUERY = 1000
    WINDOW_DAYS = 3  # Days per window (smaller = more queries but finer control)
    hits: set[int] = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        now = datetime.now(UTC)

        # Iterate through time windows from newest to oldest
        window = 0
        while len(hits) < limit and window * WINDOW_DAYS < days:
            window_end = now - timedelta(days=window * WINDOW_DAYS)
            window_start = now - timedelta(days=min((window + 1) * WINDOW_DAYS, days))

            params: dict[str, Any] = {
                "tags": "story",
                "numericFilters": (
                    f"created_at_i>{int(window_start.timestamp())},"
                    f"created_at_i<{int(window_end.timestamp())},"
                    f"points>{ALGOLIA_MIN_POINTS},"
                    f"num_comments>={MIN_STORY_COMMENTS}"
                ),
                "hitsPerPage": ALGOLIA_MAX_PER_QUERY,
            }
            resp: httpx.Response = await client.get(
                f"{ALGOLIA_BASE}/search", params=params
            )
            if resp.status_code != 200 or not resp.content:
                window += 1
                continue
            try:
                data = resp.json()
            except Exception:
                window += 1
                continue
            page_hits = data.get("hits", [])

            for h in page_hits:
                oid = int(h["objectID"])
                if oid not in exclude_ids and oid not in hits:
                    hits.add(oid)
                    if len(hits) >= limit:
                        break

            window += 1

        results: list[dict[str, Any]] = []
        if not hits:
            return []

        tasks: list[Any] = [fetch_story(client, sid) for sid in hits]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            res: Optional[dict[str, Any]] = await task
            if res:
                results.append(res)
            if progress_callback:
                progress_callback(i + 1, len(hits))

        return results
