import asyncio
import json
import re
import time
from pathlib import Path
from datetime import UTC, datetime, timedelta
import httpx
from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    ALGOLIA_MIN_POINTS,
    EXTERNAL_REQUEST_SEMAPHORE,
    TOP_COMMENTS_FOR_RANKING,
    TOP_COMMENTS_FOR_UI,
    STORY_CACHE_DIR,
    STORY_CACHE_TTL
)

ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
SEM = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)
CACHE_PATH = Path(STORY_CACHE_DIR)
CACHE_PATH.mkdir(parents=True, exist_ok=True)

async def fetch_story(client, sid) -> dict:
    cache_file = CACHE_PATH / f"{sid}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if time.time() - data["ts"] < STORY_CACHE_TTL:
                return data["story"]
        except Exception:
            pass

    async with SEM:
        try:
            resp = await client.get(f"{ALGOLIA_BASE}/items/{sid}")
            data = resp.json()
            
            comments = []
            def extract(nodes):
                for n in nodes:
                    if n.get("text"):
                        txt = re.sub("<[^<]+?>", "", n["text"])
                        comments.append((n.get("points", 0) or 0, txt))
                    extract(n.get("children", []))
            
            extract(data.get("children", []))
            comments.sort(reverse=True)
            
            top_for_rank = " ".join([c[1] for c in comments[:TOP_COMMENTS_FOR_RANKING]])
            
            story = {
                "id": data["id"],
                "title": data["title"],
                "url": data.get("url"),
                "score": data.get("points", 0),
                "time": data.get("created_at_i", 0),
                "comments": [c[1] for c in comments[:TOP_COMMENTS_FOR_UI]],
                "text_content": f"{data['title']} {top_for_rank}"
            }
            cache_file.write_text(json.dumps({"ts": time.time(), "story": story}))
            return story
        except Exception:
            return None

async def get_best_stories(limit, exclude_ids=None, progress_callback=None) -> list[dict]:
    if exclude_ids is None:
        exclude_ids = set()
    async with httpx.AsyncClient() as client:
        ts = int((datetime.now(UTC) - timedelta(days=ALGOLIA_DEFAULT_DAYS)).timestamp())
        params = {
            "tags": "story",
            "numericFilters": f"created_at_i>{ts},points>{ALGOLIA_MIN_POINTS}",
            "hitsPerPage": limit + len(exclude_ids)
        }
        resp = await client.get(f"{ALGOLIA_BASE}/search", params=params)
        hits = [h["objectID"] for h in resp.json()["hits"] if int(h["objectID"]) not in exclude_ids][:limit]
        
        results = []
        if not hits:
            return []
        
        tasks = [fetch_story(client, sid) for sid in hits]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            res = await task
            if res:
                results.append(res)
            if progress_callback:
                progress_callback(i + 1, len(hits))
                
        return results