from __future__ import annotations

import hashlib
import json
import logging
import time
import calendar
from datetime import datetime, UTC
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import cast
from collections.abc import Sequence
from xml.etree import ElementTree as ET

import httpx
import feedparser

from api.cache_utils import atomic_write_json, evict_old_cache_files
from api.content import compose_story_text, enrich_stories_with_full_text, strip_html
from api.url_utils import normalize_url
from api.constants import (
    RSS_EXTRA_FEEDS,
    RSS_CACHE_DIR,
    RSS_CACHE_MAX_FILES,
    RSS_FEED_CACHE_TTL,
    RSS_OPML_CACHE_TTL,
)
from api.models import Story, StoryDict

logger = logging.getLogger(__name__)

RSS_CACHE_PATH: Path = Path(RSS_CACHE_DIR)
RSS_CACHE_PATH.mkdir(parents=True, exist_ok=True)


def _write_cache_json(path: Path, data: dict[str, object] | Sequence[object]) -> None:
    atomic_write_json(path, data)
    evict_old_cache_files(RSS_CACHE_PATH, "*.json", RSS_CACHE_MAX_FILES)


def _parse_date(text: str) -> int | None:
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp())
    except Exception:
        pass
    try:
        iso = text.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
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

        text = strip_html(str(summary))
        text_content = compose_story_text(title=title, article_text=text)
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


def _load_cached_urls(path: Path, ttl: int) -> list[str] | None:
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _load_cached_stories(path: Path, ttl: int) -> list[Story] | None:
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
    exclude_urls: set[str] | None = None,
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
            await enrich_stories_with_full_text(client, stories)

        return stories
