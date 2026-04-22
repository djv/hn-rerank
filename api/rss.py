from __future__ import annotations

import calendar
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TypedDict, cast
from xml.etree import ElementTree as ET

import feedparser
import httpx
from langdetect import DetectorFactory, LangDetectException, detect

from api.cache_utils import atomic_write_json, evict_old_cache_files
from api.content import compose_story_text, enrich_stories_with_full_text, strip_html
from api.constants import (
    RSS_ALLOWED_SOURCE_LANGUAGES,
    RSS_CACHE_DIR,
    RSS_CACHE_MAX_FILES,
    RSS_CURATED_NEWS_PER_FEED_LIMIT,
    RSS_EXTRA_FEEDS,
    RSS_FEED_CACHE_TTL,
    RSS_FEED_CACHE_VERSION,
    RSS_OPML_CACHE_TTL,
)
from api.models import Story, StoryDict, StorySource
from api.url_utils import normalize_url

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0

RSS_CACHE_PATH: Path = Path(RSS_CACHE_DIR)
RSS_CACHE_PATH.mkdir(parents=True, exist_ok=True)


class FeedCache(TypedDict):
    version: int
    language: str | None
    stories: list[StoryDict]


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


def _feed_source(feed_url: str, link: str) -> StorySource:
    haystacks = (feed_url.lower(), link.lower())
    if any("lobste.rs" in value for value in haystacks):
        return "lobsters"
    if any("tildes.net" in value for value in haystacks):
        return "tildes"
    return "rss"


def _feed_item_limit(feed_url: str, default_limit: int) -> int:
    source = _feed_source(feed_url, feed_url)
    if source in {"lobsters", "tildes"}:
        return max(default_limit, RSS_CURATED_NEWS_PER_FEED_LIMIT)
    return default_limit


def _normalize_language(value: str | None) -> str | None:
    if not value:
        return None
    tag = value.strip().lower().replace("_", "-")
    if not tag:
        return None
    base = tag.split("-", 1)[0]
    if len(base) == 2 and base.isalpha():
        return base
    return None


def _is_allowed_feed_language(language: str | None) -> bool:
    return language in RSS_ALLOWED_SOURCE_LANGUAGES


def _feed_language_sample(parsed: feedparser.FeedParserDict, max_entries: int = 5) -> str:
    parts: list[str] = []
    feed_meta = getattr(parsed, "feed", {})
    for key in ("title", "subtitle", "description", "info"):
        value = str(feed_meta.get(key, "") or "").strip()
        if value:
            parts.append(strip_html(value))

    for entry in parsed.entries[:max_entries]:
        for key in ("title", "summary", "description"):
            value = str(entry.get(key, "") or "").strip()
            if value:
                parts.append(strip_html(value))

    return " ".join(part for part in parts if part)


def _detect_parsed_feed_language(parsed: feedparser.FeedParserDict) -> str | None:
    feed_meta = getattr(parsed, "feed", {})
    for key in ("language", "lang"):
        language = _normalize_language(str(feed_meta.get(key, "") or ""))
        if language is not None:
            return language

    sample = _feed_language_sample(parsed)
    if sum(ch.isalpha() for ch in sample) < 40:
        return None

    try:
        return _normalize_language(detect(sample))
    except LangDetectException:
        return None


def _parse_feed(
    feed_xml: str,
    feed_url: str,
    max_items: int,
    min_ts: int,
) -> tuple[str | None, list[Story]]:
    parsed = feedparser.parse(feed_xml)
    if parsed.bozo and not parsed.entries:
        logger.warning(f"Failed to parse feed XML for {feed_url}: {parsed.bozo_exception}")
        return None, []

    feed_language = _detect_parsed_feed_language(parsed)
    if not _is_allowed_feed_language(feed_language):
        logger.info(
            "Skipping feed %s due to unsupported language %s",
            feed_url,
            feed_language or "unknown",
        )
        return feed_language, []

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
            summary = entry.get("summary") or entry.get("description") or ""

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
        discussion_url = str(entry.get("comments") or "").strip() or None
        story = Story(
            id=_make_story_id(feed_url, link, title),
            title=title,
            url=link or None,
            score=0,
            time=ts,
            discussion_url=discussion_url,
            comments=[],
            text_content=text_content,
            source=_feed_source(feed_url, link),
        )
        stories.append(story)
        if len(stories) >= max_items:
            break

    return feed_language, stories


def _parse_feed_entries(
    feed_xml: str,
    feed_url: str,
    max_items: int,
    min_ts: int,
) -> list[Story]:
    _, stories = _parse_feed(feed_xml, feed_url, max_items, min_ts)
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


def _load_cached_feed(path: Path, ttl: int) -> tuple[str | None, list[Story]] | None:
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            raw = json.loads(path.read_text())
        except Exception:
            return None
        if not isinstance(raw, dict) or raw.get("version") != RSS_FEED_CACHE_VERSION:
            return None
        raw_stories = raw.get("stories")
        if not isinstance(raw_stories, list):
            return None
        language = _normalize_language(cast(str | None, raw.get("language")))
        stories = [Story.from_dict(cast(StoryDict, d)) for d in raw_stories]
        return language, stories
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

        feed_urls = list(feed_urls or [])
        if RSS_EXTRA_FEEDS:
            feed_urls.extend(RSS_EXTRA_FEEDS)
        if not feed_urls:
            return []

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
            item_limit = _feed_item_limit(feed_url, per_feed)
            feed_cache = _cache_path("feed", feed_url)
            cached = _load_cached_feed(feed_cache, RSS_FEED_CACHE_TTL)
            if cached is not None:
                feed_language, cached_stories = cached
                if not _is_allowed_feed_language(feed_language):
                    continue
                candidates = [s for s in cached_stories if s.time >= min_ts][:item_limit]
            else:
                try:
                    resp = await client.get(feed_url)
                except Exception as e:
                    logger.warning(f"Failed to fetch feed {feed_url}: {e}")
                    continue
                if resp.status_code != 200 or not resp.text:
                    continue
                feed_language, candidates = _parse_feed(resp.text, feed_url, item_limit, min_ts)
                cache_payload: dict[str, object] = {
                    "version": RSS_FEED_CACHE_VERSION,
                    "language": feed_language,
                    "stories": [s.to_dict() for s in candidates],
                }
                _write_cache_json(feed_cache, cache_payload)
                if not _is_allowed_feed_language(feed_language):
                    continue

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
