from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import time
import calendar
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional, Sequence, cast
from xml.etree import ElementTree as ET

import httpx
import feedparser
import trafilatura
from bs4 import BeautifulSoup
from bs4.element import Tag

from api.cache_utils import atomic_write_json, evict_old_cache_files
from api.url_utils import normalize_url
from api.constants import (
    EXTERNAL_REQUEST_SEMAPHORE,
    RSS_EXTRA_FEEDS,
    RSS_ARTICLE_CACHE_TTL,
    RSS_CACHE_DIR,
    RSS_CACHE_MAX_FILES,
    RSS_FEED_CACHE_TTL,
    RSS_OPML_CACHE_TTL,
)
from api.models import Story, StoryDict

logger = logging.getLogger(__name__)

RSS_CACHE_PATH: Path = Path(RSS_CACHE_DIR)
RSS_CACHE_PATH.mkdir(parents=True, exist_ok=True)

ARTICLE_SEM: asyncio.Semaphore = asyncio.Semaphore(EXTERNAL_REQUEST_SEMAPHORE)


def _write_cache_json(path: Path, data: dict[str, object] | Sequence[object]) -> None:
    atomic_write_json(path, data)
    evict_old_cache_files(RSS_CACHE_PATH, "*.json", RSS_CACHE_MAX_FILES)


def _load_cached_text(path: Path, ttl: int) -> Optional[str]:
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


def _strip_html(txt: str) -> str:
    if not txt:
        return ""
    clean = BeautifulSoup(txt, "html.parser").get_text(" ", strip=True)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


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


def _parse_date(text: str) -> Optional[int]:
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    except Exception:
        pass
    try:
        iso = text.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _extract_opml_feed_urls(opml_text: str) -> list[str]:
    try:
        root = ET.fromstring(opml_text)
    except Exception as e:
        logger.warning(f"Failed to parse OPML: {e}")
        return []

    urls: list[str] = []
    for outline in root.iter("outline"):
        xml_url = outline.attrib.get("xmlUrl")
        if xml_url:
            urls.append(xml_url)
    return urls


def _make_story_id(feed_url: str, link: str, title: str) -> int:
    key = f"{feed_url}|{link}|{title}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return -int(digest[:12], 16)


def _parse_feed_entries(
    feed_xml: str,
    feed_url: str,
    max_items: int,
    min_ts: int,
) -> list[Story]:
    parsed = feedparser.parse(feed_xml)
    if parsed.bozo and not parsed.entries:
        logger.warning(f"Failed to parse feed XML for {feed_url}: {parsed.bozo_exception}")
        return []

    now_ts = int(time.time())
    stories: list[Story] = []

    for entry in parsed.entries:
        title = str(entry.get("title", "")).strip()
        link = str(entry.get("link", "") or "").strip()

        summary = ""
        content_list = entry.get("content") or []
        if isinstance(content_list, list) and content_list:
            content_val = content_list[0].get("value")
            if isinstance(content_val, str):
                summary = content_val
        if not summary:
            summary = (
                entry.get("summary")
                or entry.get("description")
                or ""
            )

        ts = None
        for key in ("published_parsed", "updated_parsed", "created_parsed"):
            parsed_time = entry.get(key)
            if parsed_time:
                ts = int(calendar.timegm(parsed_time))
                break
        if ts is None:
            date_text = entry.get("published") or entry.get("updated") or ""
            ts = _parse_date(str(date_text)) or now_ts

        if ts < min_ts:
            continue

        text = _strip_html(str(summary))
        text_content = f"{title}. {text}".strip()
        story = Story(
            id=_make_story_id(feed_url, link, title),
            title=title,
            url=link or None,
            score=0,
            time=ts,
            comments=[],
            text_content=text_content,
        )
        stories.append(story)
        if len(stories) >= max_items:
            break

    return stories


def _cache_path(prefix: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return RSS_CACHE_PATH / f"{prefix}-{digest}.json"


async def _fetch_full_text(client: httpx.AsyncClient, url: str) -> str:
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
    clean = _strip_html(extracted)
    if not clean:
        return ""
    _cache_text(cache_path, clean)
    return clean


async def _enrich_stories_with_full_text(
    client: httpx.AsyncClient, stories: list[Story]
) -> None:
    if not stories:
        return

    tasks: list[tuple[Story, asyncio.Task[str]]] = []
    for story in stories:
        if not story.url:
            continue

        async def _wrapped_fetch(url: str) -> str:
            async with ARTICLE_SEM:
                return await _fetch_full_text(client, url)

        tasks.append((story, asyncio.create_task(_wrapped_fetch(story.url))))

    for story, task in tasks:
        try:
            full_text = await task
        except Exception as e:
            logger.debug(f"Failed to fetch full text for {story.url}: {e}")
            continue
        if full_text:
            story.text_content = f"{story.title}. {full_text}".strip()


def _load_cached_urls(path: Path, ttl: int) -> Optional[list[str]]:
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _load_cached_stories(path: Path, ttl: int) -> Optional[list[Story]]:
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            raw = json.loads(path.read_text())
            return [Story.from_dict(cast(StoryDict, d)) for d in raw]
        except Exception:
            return None
    return None


async def fetch_rss_stories(
    opml_url: str,
    days: int,
    max_feeds: int,
    per_feed: int,
    exclude_urls: Optional[set[str]] = None,
    fetch_full_content: bool = True,
) -> list[Story]:
    exclude_urls = exclude_urls or set()
    min_ts = int(time.time()) - days * 86400

    opml_cache = _cache_path("opml", opml_url)
    feed_urls = _load_cached_urls(opml_cache, RSS_OPML_CACHE_TTL)

    async with httpx.AsyncClient(timeout=30.0) as client:
        if feed_urls is None:
            try:
                resp = await client.get(opml_url)
                if resp.status_code == 200 and resp.text:
                    feed_urls = _extract_opml_feed_urls(resp.text)
                    _write_cache_json(opml_cache, feed_urls)
                else:
                    feed_urls = []
            except Exception as e:
                logger.warning(f"Failed to fetch OPML: {e}")
                feed_urls = []

        if not feed_urls:
            return []

        if RSS_EXTRA_FEEDS:
            feed_urls = list(feed_urls) + list(RSS_EXTRA_FEEDS)

        unique_feeds: list[str] = []
        seen = set()
        for url in feed_urls:
            if url not in seen:
                unique_feeds.append(url)
                seen.add(url)
            if max_feeds > 0 and len(unique_feeds) >= max_feeds:
                break

        stories: list[Story] = []
        seen_urls: set[str] = set(exclude_urls)
        seen_titles: set[str] = set()

        for feed_url in unique_feeds:
            feed_cache = _cache_path("feed", feed_url)
            cached = _load_cached_stories(feed_cache, RSS_FEED_CACHE_TTL)
            if cached is not None:
                candidates = [s for s in cached if s.time >= min_ts][:per_feed]
            else:
                try:
                    resp = await client.get(feed_url)
                except Exception as e:
                    logger.warning(f"Failed to fetch feed {feed_url}: {e}")
                    continue
                if resp.status_code != 200 or not resp.text:
                    continue
                candidates = _parse_feed_entries(resp.text, feed_url, per_feed, min_ts)
                _write_cache_json(feed_cache, [s.to_dict() for s in candidates])

            for story in candidates:
                if story.url:
                    norm_url = normalize_url(story.url)
                    if norm_url and norm_url in seen_urls:
                        continue
                    if norm_url:
                        seen_urls.add(norm_url)
                norm_title = story.title.lower().strip()
                if norm_title:
                    if norm_title in seen_titles:
                        continue
                    seen_titles.add(norm_title)
                stories.append(story)

        if fetch_full_content and stories:
            await _enrich_stories_with_full_text(client, stories)

        return stories
