import asyncio
import httpx
import json
import time
import re
from pathlib import Path
from typing import List, Optional, Tuple
from api.client import HNClient
from api.logging_config import logger
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HN Rerank API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class VoteRequest(BaseModel):
    story_id: int
    direction: str = "up"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
async def login_route(req: LoginRequest):
    async with HNClient() as client:
        success, msg = await client.login(req.username, req.password)
        if not success:
            raise HTTPException(status_code=401, detail=msg)
        return {"status": "success", "username": req.username}


@app.post("/vote")
async def vote_route(req: VoteRequest):
    async with HNClient() as client:
        if req.direction == "up":
            success, msg = await client.vote(req.story_id, "up")
        else:
            success, msg = await client.hide(req.story_id)
        if not success:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "success"}


HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA_API_BASE = "https://hn.algolia.com/api/v1"
STORY_CACHE_DIR = Path(".cache/stories")
STORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 86400

EXTERNAL_SEM = asyncio.Semaphore(50)


async def fetch_article_text(url: str) -> str:
    # Skip Body fetching for bulk ranking to improve speed by 10x
    return ""


async def fetch_story_with_comments(
    client: httpx.AsyncClient, story_id: int
) -> Optional[dict]:
    # 1. Check Cache
    cache_path = STORY_CACHE_DIR / f"{story_id}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            if time.time() - cached["retrieved_at"] < CACHE_TTL:
                return cached["data"]
        except Exception as e:
            logger.warning(f"Failed to load story cache: {e}")

    # 2. Fetch from Algolia (Fast, includes comments)
    try:
        resp = await client.get(f"{ALGOLIA_API_BASE}/items/{story_id}", timeout=3.0)
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

            text_parts = [story["title"]]

            # Extract top 10 comments
            all_comments = []

            def collect(nodes):
                for node in nodes:
                    if node.get("text"):
                        # Basic HTML cleanup
                        text = re.sub("<[^<]+?>", "", node["text"])
                        all_comments.append((node.get("points", 0) or 0, text))
                    if len(all_comments) > 40:
                        break
                    collect(node.get("children", []))

            collect(data.get("children", []))
            all_comments.sort(key=lambda x: x[0], reverse=True)

            # Use top 5 for ranking text
            for _, t in all_comments[:5]:
                text_parts.append(t)

            # Store top 10 for UI
            story["comments"] = [t for _, t in all_comments[:10]]
            story["text_content"] = " ".join(text_parts)[:3000]

            with open(cache_path, "w") as f:
                json.dump({"retrieved_at": time.time(), "data": story}, f)
            return story
    except Exception as e:
        logger.error(f"Error fetching story {story_id}: {e}")
    return None


async def get_top_stories(limit: int) -> List[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{HN_API_BASE}/topstories.json")
        ids = resp.json()[:limit]
        tasks = [fetch_story_with_comments(client, i) for i in ids]
        stories = await asyncio.gather(*tasks)
        return [s for s in stories if s]


async def get_user_data(username: str) -> Tuple[List[dict], List[dict], set]:
    async with HNClient() as client:
        fav_ids = await client.fetch_favorites(username)
        upvoted_ids, hidden_ids, submitted_ids = set(), set(), set()

        if await client.check_session():
            upvoted_ids = await client.fetch_upvoted(username)
            hidden_ids = await client.fetch_hidden(username)
            submitted_ids = await client.fetch_submitted(username)

        pos_ids = fav_ids.union(upvoted_ids)
        # All IDs to exclude from candidates
        exclude_ids = pos_ids.union(hidden_ids).union(submitted_ids)

        async def fetch_batch(ids):
            subset = sorted(list(ids), reverse=True)[:50]
            async with httpx.AsyncClient(timeout=5.0) as ac:
                res = await asyncio.gather(
                    *[fetch_story_with_comments(ac, int(idx)) for idx in subset]
                )
                return [s for s in res if s]

        pos_data, neg_data = await asyncio.gather(
            fetch_batch(pos_ids), fetch_batch(hidden_ids)
        )
        return pos_data, neg_data, exclude_ids


async def get_best_stories(
    limit: int, days: int = 30, exclude_ids: Optional[set] = None
) -> List[dict]:
    if exclude_ids is None:
        exclude_ids = set()

    found_stories = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        start_time = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        )

        # Pagination Loop
        page = 0
        max_pages = 10  # Safety limit to avoid infinite loops

        while len(found_stories) < limit and page < max_pages:
            params = {
                "tags": "story",
                "numericFilters": f"created_at_i>{start_time},points>20",
                "hitsPerPage": limit, # Fetch 'limit' items per page
                "page": page
            }

            try:
                resp = await client.get(f"{ALGOLIA_API_BASE}/search", params=params)
                if resp.status_code != 200:
                    logger.error(f"Algolia search failed: {resp.status_code}")
                    break

                data = resp.json()
                hits = data.get("hits", [])
                if not hits:
                    break

                # Filter out seen IDs
                new_ids = []
                for h in hits:
                    sid = int(h["objectID"])
                    if sid not in exclude_ids:
                        new_ids.append(sid)

                if new_ids:
                    # Fetch details
                    tasks = [fetch_story_with_comments(client, sid) for sid in new_ids]
                    batch_results = await asyncio.gather(*tasks)

                    # Add valid stories to our list
                    for s in batch_results:
                        if s:
                            found_stories.append(s)
                            if len(found_stories) >= limit:
                                break

                page += 1

            except Exception as e:
                logger.error(f"Error in get_best_stories loop: {e}")
                break

    return found_stories[:limit]
