from __future__ import annotations

import calendar
import hashlib
import json
import logging
import re
import time
from collections.abc import Sequence, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal, TypedDict, cast
from xml.etree import ElementTree as ET
from urllib.parse import urljoin, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup
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
    RSS_PER_FEED_LIMIT,
)
from api.external_sources import detect_source, get_spec
from api.models import Story, StoryDict, StorySource
from api.url_utils import normalize_url

logger = logging.getLogger(__name__)

RssProgressPhase = Literal["feeds", "content"]


class RssProgress(TypedDict):
    phase: RssProgressPhase
    current: int
    total: int
    label: str


RssProgressCallback = Callable[[RssProgress], None]

DetectorFactory.seed = 0
RSS_USER_AGENT = "linux:hn_rerank:1.0 (local-first dashboard)"
DIGG_AI_URL = "https://digg.com/ai"
DIGG_AI_STORY_PATTERN = re.compile(
    r'\\"id\\":\\"(?P<id>[^\\"]+)\\"'
    r',\\"shortId\\":\\"(?P<short_id>[^\\"]+)\\"'
    r',\\"title\\":\\"(?P<title>(?:\\\\.|[^\\"])*)\\"'
    r',\\"tldr\\":\\"(?P<tldr>(?:\\\\.|[^\\"])*)\\"'
    r',\\"createdAt\\":\\"(?P<created_at>[^\\"]+)\\"'
    r',\\"firstPostAt\\":\\"[^\\"]+\\"'
    r',\\"lastFrozenPostAt\\":\\"[^\\"]+\\"'
    r',\\"postCount\\":(?P<post_count>\d+)'
)

RSS_CACHE_PATH: Path = Path(RSS_CACHE_DIR)
RSS_CACHE_PATH.mkdir(parents=True, exist_ok=True)


def _log_rss_request_error(label: str, url: str, exc: httpx.RequestError) -> None:
    logger.warning("%s %s failed: %s", label, url, exc)


def _write_cache_json(path: Path, data: dict[str, object] | Sequence[object]) -> None:
    atomic_write_json(path, data)
    evict_old_cache_files(RSS_CACHE_PATH, "*.json", RSS_CACHE_MAX_FILES)


def _parse_date(text: str) -> int | None:
    if not text:
        return None
    first_error: Exception | None = None
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp())
    except Exception as exc:
        first_error = exc
    second_error: Exception | None = None
    try:
        iso = text.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())
    except Exception as exc:
        second_error = exc
    logger.debug(
        "Failed to parse RSS date %r (parsedate_to_datetime=%s, fromisoformat=%s)",
        text,
        first_error,
        second_error,
    )
    return None


def _extract_opml_feed_urls(opml_text: str) -> list[str]:
    try:
        root = ET.fromstring(opml_text)
    except Exception:
        logger.exception("Failed to parse OPML")
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
    return cast(StorySource, detect_source(feed_url, link).source)


def _feed_item_limit(feed_url: str, default_limit: int) -> int:
    spec = detect_source(feed_url, feed_url)
    if spec.curated:
        return max(default_limit, RSS_CURATED_NEWS_PER_FEED_LIMIT)
    return max(default_limit, RSS_PER_FEED_LIMIT)


def _preserves_feed_text_content(source: StorySource) -> bool:
    return get_spec(source).preserves_feed_text


def _is_digg_ai_source(feed_url: str) -> bool:
    return detect_source(feed_url, feed_url).parser == "digg_ai"


def _parsed_feed_timestamp(parsed: feedparser.FeedParserDict) -> int | None:
    feed_meta = getattr(parsed, "feed", {})
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed_time = feed_meta.get(key)
        if parsed_time:
            return int(calendar.timegm(parsed_time))

    date_text = feed_meta.get("published") or feed_meta.get("updated") or ""
    return _parse_date(str(date_text))


def _decode_next_string(value: str) -> str:
    try:
        decoded = json.loads(f'"{value}"')
    except json.JSONDecodeError:
        decoded = value
    return str(decoded).strip()


def _parse_digg_ai_page(
    html_text: str,
    feed_url: str,
    max_items: int,
    min_ts: int,
) -> list[Story]:
    stories: list[Story] = []
    seen_urls: set[str] = set()

    for match in DIGG_AI_STORY_PATTERN.finditer(html_text):
        title = _decode_next_string(match.group("title"))
        if not title:
            continue

        ts = _parse_date(match.group("created_at"))
        if ts is None or ts < min_ts:
            continue

        story_path_id = match.group("id") or match.group("short_id")
        link = urljoin(DIGG_AI_URL, f"/ai/{story_path_id}")
        if link in seen_urls:
            continue
        seen_urls.add(link)

        summary = _decode_next_string(match.group("tldr"))
        story = Story(
            id=_make_story_id(feed_url, link, title),
            title=title,
            url=link,
            score=0,
            time=ts,
            discussion_url=None,
            comments=[],
            text_content=compose_story_text(title=title, article_text=summary),
            source="digg",
            comment_count=int(match.group("post_count")),
        )
        stories.append(story)
        if len(stories) >= max_items:
            break

    return stories


def _is_reddit_internal_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {"reddit.com", "www.reddit.com", "old.reddit.com"}:
        return False
    return parsed.path.startswith(("/r/", "/gallery/"))


def _extract_reddit_urls(content_html: str, entry_link: str) -> tuple[str, str | None]:
    discussion_url = entry_link
    submitted_url = entry_link
    soup = BeautifulSoup(content_html, "html.parser")

    for anchor in soup.find_all("a", href=True):
        href = urljoin("https://www.reddit.com", str(anchor.get("href") or "").strip())
        label = anchor.get_text(" ", strip=True).lower()
        if label == "[comments]":
            discussion_url = href
        elif label == "[link]":
            submitted_url = href

    if _is_reddit_internal_url(submitted_url):
        submitted_url = discussion_url
    return submitted_url, discussion_url or None


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


def _feed_language_sample(
    parsed: feedparser.FeedParserDict, max_entries: int = 5
) -> str:
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


def _extract_score(
    entry: feedparser.FeedParserDict, source: StorySource, summary_text: str
) -> int:
    """Extract story score/points from various RSS metadata sources."""
    # 1. Direct attribute check (some feed types or custom parser results)
    # Reddit RSS often has reddit_score via feedparser if using the right namespace
    source_score_keys = [f"{source}_score"]
    if get_spec(source).is_reddit_like:
        source_score_keys.append("reddit_score")
    source_score_keys.extend(["score", "points", "rank_score"])
    for key in source_score_keys:
        val = entry.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    # 2. Textual extraction from summary (common for Tildes and some RSS variations)
    # Tildes uses "<p>Votes: 1</p>"
    # Standard format: "score: 123" or "(score: 123)"
    patterns = [
        r"score:\s*(\d+)",
        r"Votes:\s*(\d+)",
        r"Points:\s*(\d+)",
        r"Score\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, summary_text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return 0


def _extract_comment_count(
    entry: feedparser.FeedParserDict, source: StorySource
) -> int | None:
    attrs = get_spec(source).comment_count_attrs
    if not attrs:
        return None
    for key in attrs:
        value = entry.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _parse_feed(
    feed_xml: str,
    feed_url: str,
    max_items: int,
    min_ts: int,
) -> tuple[str | None, list[Story]]:
    parsed = feedparser.parse(feed_xml)
    if parsed.bozo and not parsed.entries:
        logger.warning(
            f"Failed to parse feed XML for {feed_url}: {parsed.bozo_exception}"
        )
        return None, []

    feed_language = _detect_parsed_feed_language(parsed)
    if not _is_allowed_feed_language(feed_language):
        logger.info(
            "Skipping feed %s due to unsupported language %s",
            feed_url,
            feed_language or "unknown",
        )
        return feed_language, []

    stories: list[Story] = []
    feed_timestamp = _parsed_feed_timestamp(parsed)

    for entry in parsed.entries:
        title = str(entry.get("title", "")).strip()
        link = str(entry.get("link", "") or "").strip()
        source = _feed_source(feed_url, link)

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
            ts = _parse_date(str(date_text))

        if ts is None and get_spec(source).uses_feed_date:
            ts = feed_timestamp

        if ts is None:
            continue

        if ts < min_ts:
            continue

        story_url = link or None
        discussion_url = str(entry.get("comments") or "").strip() or None
        if get_spec(source).is_reddit_like:
            story_url, discussion_url = _extract_reddit_urls(str(summary), link)

        text = strip_html(str(summary))
        text_content = compose_story_text(title=title, article_text=text)
        story = Story(
            id=_make_story_id(feed_url, link, title),
            title=title,
            url=story_url,
            score=_extract_score(entry, source, text),
            time=ts,
            discussion_url=discussion_url,
            comments=[],
            text_content=text_content,
            source=source,
            comment_count=_extract_comment_count(entry, source),
        )
        stories.append(story)
        if len(stories) >= max_items:
            break

    return feed_language, stories


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
    progress_callback: RssProgressCallback | None = None,
) -> list[Story]:
    exclude_urls = exclude_urls or set()
    min_ts = int(time.time()) - days * 86400

    opml_cache = _cache_path("opml", opml_url)
    feed_urls = _load_cached_urls(opml_cache, RSS_OPML_CACHE_TTL)

    async with httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": RSS_USER_AGENT}
    ) as client:
        if feed_urls is None:
            if opml_url.startswith(("http://", "https://")):
                try:
                    resp = await client.get(opml_url)
                    if resp.status_code == 200 and resp.text:
                        feed_urls = _extract_opml_feed_urls(resp.text)
                        _write_cache_json(opml_cache, feed_urls)
                    else:
                        feed_urls = []
                except httpx.RequestError as exc:
                    _log_rss_request_error("OPML fetch", opml_url, exc)
                    feed_urls = []
                except Exception:
                    logger.exception("Failed to fetch OPML")
                    feed_urls = []
            else:
                try:
                    with open(opml_url, "r", encoding="utf-8") as f:
                        opml_text = f.read()
                    feed_urls = _extract_opml_feed_urls(opml_text)
                    _write_cache_json(opml_cache, feed_urls)
                except Exception:
                    logger.exception(f"Failed to read local OPML file: {opml_url}")
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

        total_feeds = len(unique_feeds)
        for feed_index, feed_url in enumerate(unique_feeds):
            try:
                item_limit = _feed_item_limit(feed_url, per_feed)
                feed_cache = _cache_path("feed", feed_url)
                cached = _load_cached_feed(feed_cache, RSS_FEED_CACHE_TTL)
                if cached is not None:
                    feed_language, cached_stories = cached
                    if not _is_allowed_feed_language(feed_language):
                        continue
                    candidates = [s for s in cached_stories if s.time >= min_ts][
                        :item_limit
                    ]
                else:
                    try:
                        resp = await client.get(feed_url)
                    except httpx.RequestError as exc:
                        _log_rss_request_error("Feed fetch", feed_url, exc)
                        continue
                    if resp.status_code != 200 or not resp.text:
                        continue
                    if _is_digg_ai_source(feed_url):
                        feed_language = "en"
                        candidates = _parse_digg_ai_page(
                            resp.text, feed_url, item_limit, min_ts
                        )
                    else:
                        feed_language, candidates = _parse_feed(
                            resp.text, feed_url, item_limit, min_ts
                        )
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
            finally:
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "feeds",
                            "current": feed_index + 1,
                            "total": total_feeds,
                            "label": "Fetching external feeds",
                        }
                    )

        stories_for_full_content = [
            story for story in stories if not _preserves_feed_text_content(story.source)
        ]
        if fetch_full_content and stories_for_full_content:

            def content_progress(curr: int, total: int) -> None:
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "content",
                            "current": curr,
                            "total": total,
                            "label": "Fetching external article text",
                        }
                    )

            await enrich_stories_with_full_text(
                client, stories_for_full_content, progress_callback=content_progress
            )

        return stories
