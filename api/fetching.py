from __future__ import annotations
import asyncio
import json
import os
import re
import time
from pathlib import Path
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Optional
import httpx
from bs4 import BeautifulSoup
from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    ALGOLIA_MIN_POINTS,
    EXTERNAL_REQUEST_SEMAPHORE,
    TOP_COMMENTS_FOR_RANKING,
    TOP_COMMENTS_FOR_UI,
    RANKING_DEPTH_PENALTY,
    STORY_CACHE_DIR,
    STORY_CACHE_TTL,
    STORY_CACHE_MAX_FILES,
)

ALGOLIA_BASE: str = "https://hn.algolia.com/api/v1"
HN_BASE: str = "https://news.ycombinator.com"
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


async def fetch_story(client: httpx.AsyncClient, sid: int) -> Optional[dict[str, Any]]:
    cache_file: Path = CACHE_PATH / f"{sid}.json"
    if cache_file.exists():
        try:
            data: dict[str, Any] = json.loads(cache_file.read_text())
            if time.time() - float(data["ts"]) < STORY_CACHE_TTL:
                return data.get("story") # Returns None if cached as None
        except Exception:
            pass

    async with SEM:
        try:
            # We scrape HN directly to get comments in ranked order (score/quality),
            # as APIs do not expose comment scores or reliable ranking.
            resp: httpx.Response = await client.get(
                f"{HN_BASE}/item?id={sid}",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code != 200:
                # Cache failure to prevent retry
                _atomic_write_json(cache_file, {"ts": time.time(), "story": None})
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            
            # 1. Metadata
            title_tag = soup.find("span", class_="titleline")
            if not title_tag:
                # Likely a comment or job or invalid page. Cache as None.
                _atomic_write_json(cache_file, {"ts": time.time(), "story": None})
                return None
            
            title_link = title_tag.find("a")
            title = title_link.get_text()
            url = title_link["href"]
            
            score_span = soup.find("span", class_="score")
            score = int(score_span.get_text().split()[0]) if score_span else 0
            
            age_span = soup.find("span", class_="age")
            # title attribute has format "YYYY-MM-DDTHH:MM:SS UNIX_TIMESTAMP"
            ts_str = age_span["title"].split()[-1] if age_span and "title" in age_span.attrs else "0"
            created_at = int(ts_str) if ts_str.isdigit() else 0

            # 2. Comments (Ranked by HN)
            # We scan a larger pool to prioritize "Breadth-First" selection.
            # This avoids getting stuck in a single deep thread (the "Tree Problem").
            comment_rows = soup.find_all("tr", class_="comtr")
            
            candidates: list[dict[str, Any]] = []
            
            for i, row in enumerate(comment_rows[:TOP_COMMENTS_FOR_RANKING * 5]):
                # Parse Indent
                ind_td = row.find("td", class_="ind")
                indent = int(ind_td.get("indent")) if ind_td else 0
                
                commtext = row.find(class_="commtext")
                if not commtext:
                    continue
                
                # Remove reply link
                for reply in commtext.find_all("div", class_="reply"):
                    reply.decompose()
                    
                txt = commtext.get_text(" ", strip=True) # preserve spacing slightly
                
                clean_txt = _clean_text(txt)
                if clean_txt:
                    candidates.append({
                        "original_index": i,
                        "indent": indent,
                        "text": clean_txt
                    })

            # Weighted Sort: Index + (Indent * Penalty)
            # This balances "Page Rank" (Index) with "Tree Depth" (Indent).
            # A low penalty (e.g. 10) allows top replies to beat late roots,
            # but penalizes deep nested threads.
            candidates.sort(key=lambda x: x["original_index"] + (x["indent"] * RANKING_DEPTH_PENALTY))

            # Select Top N
            selected = candidates[:TOP_COMMENTS_FOR_RANKING]
            
            # For the text content (embedding), the order matters less, but logical flow is nice.
            # We'll just join them.
            top_for_rank: str = " ".join([c["text"] for c in selected])

            # For UI, we take the top M from this diverse set.
            # We might want to sort them back by original index to show "flow" if they are close,
            # but since we are cherry-picking roots, "original index" order puts them in 'Page Order'.
            # Page Order for roots = "Best" algorithm order. This is desirable.
            ui_candidates = selected[:TOP_COMMENTS_FOR_UI]
            ui_candidates.sort(key=lambda x: x["original_index"])
            
            ui_comments = [c["text"] for c in ui_candidates]

            story: dict[str, Any] = {
                "id": sid,
                "title": title,
                "url": url,
                "score": score,
                "time": created_at,
                "comments": ui_comments,
                "text_content": f"{title}. {top_for_rank}",
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