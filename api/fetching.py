from __future__ import annotations
import asyncio
import json
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
    TOP_COMMENTS_FOR_RANKING,
    TOP_COMMENTS_FOR_UI,
    STORY_CACHE_DIR,
    STORY_CACHE_TTL,
)

ALGOLIA_BASE: str = "https://hn.algolia.com/api/v1"
SEM: asyncio.Semaphore = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)
CACHE_PATH: Path = Path(STORY_CACHE_DIR)
CACHE_PATH.mkdir(parents=True, exist_ok=True)


async def fetch_story(client: httpx.AsyncClient, sid: int) -> Optional[dict[str, Any]]:
    cache_file: Path = CACHE_PATH / f"{sid}.json"
    if cache_file.exists():
        try:
            data: dict[str, Any] = json.loads(cache_file.read_text())
            if time.time() - float(data["ts"]) < STORY_CACHE_TTL:
                return data["story"]
        except Exception:
            pass

    async with SEM:
        try:
            resp: httpx.Response = await client.get(f"{ALGOLIA_BASE}/items/{sid}")
            data = resp.json()

            comments: list[tuple[int, str]] = []

            def extract(nodes: list[dict[str, Any]]) -> None:
                for n in nodes:
                    if n.get("text"):
                        txt: str = re.sub("<[^<]+?>", "", str(n["text"]))
                        comments.append((int(n.get("points", 0) or 0), txt))
                    extract(list(n.get("children", [])))

            extract(list(data.get("children", [])))
            comments.sort(reverse=True)

            # Detect if story is a dupe/closed (HN mods often move comments)
            is_moved: bool = any(
                "Comments moved to" in (str(c.get("text") or ""))
                for c in list(data.get("children", []))[:5]
            )
            if is_moved:
                return None

            top_for_rank = " ".join(
                [c[1] for c in comments[:TOP_COMMENTS_FOR_RANKING]]
            )
            
            # Clean noise (Braille art, excessive symbols)
            # This regex targets common Braille art blocks
            top_for_rank = re.sub(r"[\u2800-\u28FF]{3,}", "", top_for_rank)
            # Remove excessive punctuation/symbols often used in ASCII art
            top_for_rank = re.sub(r"[#*^\\/\\|\\-_+]{5,}", "", top_for_rank)

            story: dict[str, Any] = {
                "id": int(data["id"]),
                "title": str(data["title"]),
                "url": data.get("url"),
                "score": int(data.get("points", 0)),
                "time": int(data.get("created_at_i", 0)),
                "comments": [c[1] for c in comments[:TOP_COMMENTS_FOR_UI]],
                # Boost title weight by including it twice
                "text_content": f"{data['title']}. {data['title']}. {top_for_rank}",
            }
            cache_file.write_text(json.dumps({"ts": time.time(), "story": story}))
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
    async with httpx.AsyncClient() as client:
        ts: int = int((datetime.now(UTC) - timedelta(days=days)).timestamp())
        params: dict[str, Any] = {
            "tags": "story",
            "numericFilters": f"created_at_i>{ts},points>{ALGOLIA_MIN_POINTS}",
            "hitsPerPage": limit + len(exclude_ids),
        }
        resp: httpx.Response = await client.get(f"{ALGOLIA_BASE}/search", params=params)
        hits: list[int] = [
            int(h["objectID"])
            for h in resp.json()["hits"]
            if int(h["objectID"]) not in exclude_ids
        ][:limit]

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