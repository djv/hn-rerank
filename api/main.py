import asyncio
import httpx
import json
import time
import re
from pathlib import Path
from typing import List, Optional, Tuple
from api.client import HNClient
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
    client = HNClient()
    try:
        success, msg = await client.login(req.username, req.password)
        if not success:
            raise HTTPException(status_code=401, detail=msg)
        return {"status": "success", "username": req.username}
    finally:
        await client.close()


@app.post("/vote")
async def vote_route(req: VoteRequest):
    client = HNClient()
    try:
        if req.direction == "up":
            success, msg = await client.vote(req.story_id, "up")
        else:
            success, msg = await client.hide(req.story_id)
        if not success:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "success"}
    finally:
        await client.close()


HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA_API_BASE = "https://hn.algolia.com/api/v1"
STORY_CACHE_DIR = Path(".cache/stories")
STORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 86400

EXTERNAL_SEM = asyncio.Semaphore(50)


import trafilatura

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
    client: httpx.AsyncClient, story_id: int
) -> Optional[dict]:
    # 1. Check Cache
    cache_path = STORY_CACHE_DIR / f"{story_id}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            
            # Check if cache is recent and has the new fields
            # We want article_snippet if it's an external link
            data = cached["data"]
            has_snippet = "article_snippet" in data or not data.get("url") or "news.ycombinator.com" in data.get("url", "")
            
            if time.time() - cached["retrieved_at"] < CACHE_TTL and has_snippet:
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

            text_parts = [story["title"]]

            # Try to fetch article text for deeper semantic matching
            if story.get("url") and "news.ycombinator.com" not in story["url"]:
                async with EXTERNAL_SEM:
                    article_text = await fetch_article_text(story["url"])
                if article_text:
                    # Use first 2000 chars of article for ranking
                    text_parts.append(article_text[:2000])
                    story["article_snippet"] = article_text[:1000] # Store snippet for UI

            # Extract top comments
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
            story["text_content"] = " ".join(text_parts)[:5000]

            with open(cache_path, "w") as f:
                json.dump({"retrieved_at": time.time(), "data": story}, f)
            return story
    except Exception:
        pass
    return None


async def get_top_stories(limit: int) -> List[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{HN_API_BASE}/topstories.json")
        ids = resp.json()[:limit]
        tasks = [fetch_story_with_comments(client, i) for i in ids]
        stories = await asyncio.gather(*tasks)
        return [s for s in stories if s]


async def get_user_data(username: str) -> Tuple[List[dict], List[dict], set]:
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
    finally:
        await client.close()


async def get_best_stories(
    limit: int, days: int = 30, exclude_ids: Optional[set] = None
) -> List[dict]:
    if exclude_ids is None:
        exclude_ids = set()
    async with httpx.AsyncClient(timeout=10.0) as client:
        start_time = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        )
        params = {
            "tags": "story",
            "numericFilters": f"created_at_i>{start_time},points>20",
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
        res = await asyncio.gather(
            *[fetch_story_with_comments(client, sid) for sid in hit_ids]
        )
        return [s for s in res if s]
