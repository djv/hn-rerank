from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import time
from pathlib import Path
from collections.abc import Sequence

import httpx
import trafilatura
from bs4 import BeautifulSoup
from bs4.element import Tag

from api.cache_utils import atomic_write_json, evict_old_cache_files
from api.constants import (
    ARTICLE_SNIPPET_LENGTH,
    EXTERNAL_REQUEST_SEMAPHORE,
    RSS_ARTICLE_CACHE_TTL,
    RSS_CACHE_DIR,
    RSS_CACHE_MAX_FILES,
)
from api.models import Story

logger = logging.getLogger(__name__)

CONTENT_CACHE_PATH: Path = Path(RSS_CACHE_DIR)
CONTENT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

ARTICLE_SEM: asyncio.Semaphore = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)


def _write_cache_json(path: Path, data: dict[str, object] | Sequence[object]) -> None:
    atomic_write_json(path, data)
    evict_old_cache_files(CONTENT_CACHE_PATH, "*.json", RSS_CACHE_MAX_FILES)


def _load_cached_text(path: Path, ttl: int) -> str | None:
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                text = raw.get("text")
                if isinstance(text, str):
                    return text
        except Exception:
            return None
    return None


def _cache_text(path: Path, text: str) -> None:
    _write_cache_json(path, {"text": text})


def strip_html(txt: str) -> str:
    if not txt:
        return ""
    if "<" not in txt and ">" not in txt and "&" not in txt:
        clean = html.unescape(txt)
        clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    clean = BeautifulSoup(txt, "html.parser").get_text(" ", strip=True)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def truncate_text(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean

    window = clean[: max_chars + 1]
    cut = window.rfind(" ")
    if cut >= int(max_chars * 0.75):
        window = window[:cut]
    else:
        window = clean[:max_chars]
    return window.rstrip(" .,;:") + "..."


def _fallback_extract_text(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    article = soup.find("article")
    target: Tag | BeautifulSoup
    if isinstance(article, Tag):
        target = article
    elif isinstance(soup.body, Tag):
        target = soup.body
    else:
        target = soup

    text = target.get_text(" ", strip=True)
    text = html.unescape(text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cache_path(prefix: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return CONTENT_CACHE_PATH / f"{prefix}-{digest}.json"


async def fetch_full_text(client: httpx.AsyncClient, url: str) -> str:
    if not url:
        return ""
    cache_path = _cache_path("article", url)
    cached = _load_cached_text(cache_path, RSS_ARTICLE_CACHE_TTL)
    if cached is not None:
        return cached
    try:
        resp = await client.get(url, follow_redirects=True)
    except Exception as e:
        logger.debug(f"Failed to fetch article {url}: {e}")
        return ""
    if resp.status_code != 200 or not resp.text:
        return ""
    content_type = (resp.headers.get("content-type") or "").lower()
    if "html" not in content_type:
        return ""
    extracted = trafilatura.extract(
        resp.text,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not extracted:
        extracted = _fallback_extract_text(resp.text)
    clean = strip_html(extracted)
    if not clean:
        return ""
    _cache_text(cache_path, clean)
    return clean


async def enrich_stories_with_full_text(
    client: httpx.AsyncClient,
    stories: list[Story],
) -> None:
    if not stories:
        return

    tasks: list[tuple[Story, asyncio.Task[str]]] = []
    for story in stories:
        url = getattr(story, "url", None)
        if not url:
            continue

        async def _wrapped_fetch(target_url: str) -> str:
            async with ARTICLE_SEM:
                return await fetch_full_text(client, target_url)

        tasks.append((story, asyncio.create_task(_wrapped_fetch(url))))

    for story, task in tasks:
        try:
            full_text = await task
        except Exception as e:
            logger.debug(f"Failed to fetch full text for {getattr(story, 'url', None)}: {e}")
            continue
        if full_text:
            title = getattr(story, "title", "")
            story.text_content = compose_story_text(title=title, article_text=full_text)


def compose_story_text(
    title: str,
    self_text: str = "",
    article_text: str = "",
    comments: Sequence[str] | None = None,
) -> str:
    clean_title = strip_html(title)
    clean_self = truncate_text(strip_html(self_text), ARTICLE_SNIPPET_LENGTH // 2)
    clean_article = truncate_text(strip_html(article_text), ARTICLE_SNIPPET_LENGTH)
    clean_comments = " ".join(strip_html(comment) for comment in (comments or []) if comment)

    parts: list[str] = []
    if clean_title:
        parts.append(f"{clean_title}.")
    if clean_self:
        parts.append(clean_self)
    if clean_article:
        parts.append(clean_article)
    if clean_comments:
        parts.append(clean_comments)

    return " ".join(part for part in parts if part).strip()
