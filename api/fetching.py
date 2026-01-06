"""
Data fetching module for HN stories and user data.
"""
import asyncio
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import trafilatura

from api.client import HNClient
from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    ALGOLIA_MIN_POINTS,
    ARTICLE_RANKING_LENGTH,
    ARTICLE_SNIPPET_LENGTH,
    EXTERNAL_REQUEST_SEMAPHORE,
    MAX_COMMENTS_COLLECTED,
    MAX_USER_STORIES,
    STORY_CACHE_TTL,
    TEXT_CONTENT_MAX_LENGTH,
    TOP_COMMENTS_FOR_RANKING,
    TOP_COMMENTS_FOR_UI,
)

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA_API_BASE = "https://hn.algolia.com/api/v1"
STORY_CACHE_DIR = Path(".cache/stories")
STORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_SEM = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)


async def fetch_article_text(url: str) -> str:
    """Fetch and extract main text from a URL using trafilatura."""
    if not url or "news.ycombinator.com" in url:
        return ""
    try:
        def _download_and_extract():
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                # Extract text, favoring clean and concise output
                return trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return ""

        content = await asyncio.to_thread(_download_and_extract)
        return content or ""
    except Exception:
        return ""


async def fetch_story_with_comments(
    client: httpx.AsyncClient, story_id: int, fetch_article: bool = False
) -> dict | None:
    # 1. Check Cache
    cache_path = STORY_CACHE_DIR / f"{story_id}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)

            data = cached["data"]
            if time.time() - cached["retrieved_at"] < STORY_CACHE_TTL:
                return data
        except Exception:
            pass

    # 2. Fetch from Algolia (Fast, includes comments)
    try:
        async with EXTERNAL_SEM:
            resp = await client.get(f"{ALGOLIA_API_BASE}/items/{story_id}", timeout=5.0)

        if resp.status_code == 200:
            data = resp.json()
            story = {
                "id": data.get("id"),
                "title": data.get("title") or data.get("story_title") or "Untitled",
                "url": data.get("url") or data.get("story_url"),
                "score": data.get("points", 0),
                "descendants": len(data.get("children", [])),
                "time": data.get("created_at_i", 0),
                "comments": [],
            }
            if not story["title"]:
                return None

            text_parts: list[str] = [str(story["title"])]

            # Optionally fetch article text for deeper semantic matching
            if fetch_article:
                url = story.get("url")
                if isinstance(url, str) and "news.ycombinator.com" not in url:
                    async with EXTERNAL_SEM:
                        article_text = await fetch_article_text(url)
                    if article_text:
                        text_parts.append(article_text[:ARTICLE_RANKING_LENGTH])
                        story["article_snippet"] = article_text[:ARTICLE_SNIPPET_LENGTH]

            # Extract top comments
            all_comments: list[tuple[int, str]] = []

            def collect(nodes: list[dict]) -> None:
                for node in nodes:
                    if node.get("text"):
                        # Basic HTML cleanup
                        text = re.sub("<[^<]+?>", "", node["text"])
                        all_comments.append((node.get("points", 0) or 0, text))
                    if len(all_comments) > MAX_COMMENTS_COLLECTED:
                        break
                    collect(node.get("children", []))

            collect(data.get("children", []))
            all_comments.sort(key=lambda x: x[0], reverse=True)

            # Use top N for ranking text
            for _, t in all_comments[:TOP_COMMENTS_FOR_RANKING]:
                text_parts.append(t)

            # Store top N for UI
            story["comments"] = [t for _, t in all_comments[:TOP_COMMENTS_FOR_UI]]
            story["text_content"] = " ".join(text_parts)[:TEXT_CONTENT_MAX_LENGTH]

            with open(cache_path, "w") as f:
                json.dump({"retrieved_at": time.time(), "data": story}, f)
            return story
    except Exception:
        pass
    return None


async def get_top_stories(limit: int) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{HN_API_BASE}/topstories.json")
        ids = resp.json()[:limit]
        tasks = [fetch_story_with_comments(client, i) for i in ids]
        stories = await asyncio.gather(*tasks)
        return [s for s in stories if s]


async def get_user_data(username: str) -> tuple[list[dict], list[dict], set[int]]:
    client = HNClient()
    try:
        fav_ids = await client.fetch_favorites(username)
        upvoted_ids, hidden_ids, submitted_ids = set(), set(), set()

        if await client.check_session():
            upvoted_ids = await client.fetch_upvoted(username)
            hidden_ids = await client.fetch_hidden(username)
            submitted_ids = await client.fetch_submitted(username)

        pos_ids = fav_ids.union(upvoted_ids)
        # All IDs to exclude from candidates
        exclude_ids = pos_ids.union(hidden_ids).union(submitted_ids)

        async def fetch_batch(ids: set[int]) -> list[dict]:
            subset = sorted(list(ids), reverse=True)[:MAX_USER_STORIES]
            async with httpx.AsyncClient(timeout=5.0) as ac:
                res = await asyncio.gather(
                    *[fetch_story_with_comments(ac, int(idx)) for idx in subset]
                )
                return [s for s in res if s]

        pos_data, neg_data = await asyncio.gather(
            fetch_batch(pos_ids), fetch_batch(hidden_ids)
        )
        return pos_data, neg_data, exclude_ids
    finally:
        await client.close()


async def get_best_stories(
    limit: int,
    days: int = ALGOLIA_DEFAULT_DAYS,
    exclude_ids: set[int] | None = None,
    progress_callback: callable = None,
) -> list[dict]:
    if exclude_ids is None:
        exclude_ids = set()
    async with httpx.AsyncClient(timeout=10.0) as client:
        start_time = int(
            (datetime.now(UTC) - timedelta(days=days)).timestamp()
        )
        params = {
            "tags": "story",
            "numericFilters": f"created_at_i>{start_time},points>{ALGOLIA_MIN_POINTS}",
            "hitsPerPage": limit,
        }
        resp = await client.get(f"{ALGOLIA_API_BASE}/search", params=params)
        if resp.status_code != 200:
            return []

        hit_ids = [
            int(h["objectID"])
            for h in resp.json().get("hits", [])
            if int(h["objectID"]) not in exclude_ids
        ]
        
        results = []
        batch_size = 50
        for i in range(0, len(hit_ids), batch_size):
            batch = hit_ids[i : i + batch_size]
            batch_res = await asyncio.gather(
                *[fetch_story_with_comments(client, sid) for sid in batch]
            )
            results.extend([s for s in batch_res if s])
            if progress_callback:
                progress_callback(len(results), len(hit_ids))
                
        return results
