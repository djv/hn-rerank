from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import httpx
import numpy as np
import onnxruntime as ort
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from numpy.typing import NDArray
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from database import Database, Story
from transformers import AutoTokenizer


@dataclass(frozen=True)
class ModelConfig:
    svm_c: float = 0.3
    svm_gamma: float | str = 0.1
    svm_kernel: str = "rbf"
    neutral_weight: float = 0.0
    diversity_threshold: float = 0.50


# Comment text refetch thresholds (used during regen to refresh stale text_content)
COMMENT_GROWTH_THRESHOLD = 0.30  # 30% growth since last text fetch triggers refetch
COMMENT_REFETCH_MAX_AGE_HOURS = 24  # Only refetch stories younger than this
MAX_REFETCH_PER_REGEN = 10  # Bound Algolia cost per regen


@dataclass(frozen=True)
class RssConfig:
    enabled: bool = True
    per_feed_limit: int = 70
    feeds: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    username: str = "user"
    db_path: str = "hn_rewrite.db"
    output: str = "public/index.html"
    days: int = 30
    count: int = 40
    onnx_model_dir: str = "onnx_model"
    server_port: int = 8765
    regen_interval_seconds: int = 10800
    model: ModelConfig = field(default_factory=ModelConfig)
    rss: RssConfig = field(default_factory=RssConfig)

    @classmethod
    def load(cls, path: str = "config.toml") -> Config:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            return cls()

        main_cfg = data.get("hn_rewrite", {})
        model_cfg = main_cfg.get("model", {})
        rss_cfg = main_cfg.get("rss", {})

        return cls(
            username=main_cfg.get("username", "user"),
            db_path=main_cfg.get("db_path", "hn_rewrite.db"),
            output=main_cfg.get("output", "public/index.html"),
            days=main_cfg.get("days", 30),
            count=main_cfg.get("count", 40),
            onnx_model_dir=main_cfg.get("onnx_model_dir", "onnx_model"),
            server_port=main_cfg.get("server_port", 8765),
            regen_interval_seconds=main_cfg.get("regen_interval_seconds", 10800),
            model=ModelConfig(
                svm_c=model_cfg.get("svm_c", 0.3),
                svm_gamma=model_cfg.get("svm_gamma", 0.1),
                svm_kernel=model_cfg.get("svm_kernel", "rbf"),
                neutral_weight=model_cfg.get("neutral_weight", 0.0),
                diversity_threshold=model_cfg.get("diversity_threshold", 0.50),
            ),
            rss=RssConfig(
                enabled=rss_cfg.get("enabled", True),
                per_feed_limit=rss_cfg.get("per_feed_limit", 70),
                feeds=tuple(rss_cfg.get("feeds", [])),
            ),
        )


@dataclass(frozen=True)
class RankedStory:
    story: Story
    score: float
    best_match_title: str
    prob_down: float | None = None
    prob_neutral: float | None = None
    prob_up: float | None = None
    is_uncertain: bool = False
    is_novel: bool = False
    is_discussion_rich: bool = False
    is_high_engagement: bool = False
    is_hot: bool = False
    is_similar: bool = False


# Text processing helpers
def clean_text(raw_text: str, min_len: int = 0) -> str:
    if not raw_text:
        return ""
    if "<" not in raw_text and ">" not in raw_text and "&" not in raw_text:
        txt = html.unescape(raw_text)
    else:
        try:
            txt = BeautifulSoup(raw_text, "html.parser").get_text(" ", strip=True)
        except Exception:
            txt = re.sub(r"<[^>]*>", " ", raw_text)
        txt = html.unescape(txt)

    txt = re.sub(r"[\u2800-\u28FF\u2500-\u27BF]+", "", txt)
    txt = re.sub(r"[#*^\\/|\\-_+]{3,}", "", txt)
    txt = re.sub(r"\s+([.,;:!?])", r"\1", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    if len(txt) <= min_len:
        return ""
    alnum = sum(c.isalnum() for c in txt)
    if len(txt) > 0 and (alnum / len(txt)) < 0.5:
        return ""
    return txt


def _extract_comments_recursive(
    children: list,
    depth: int = 0,
    max_depth: int = 3,
    parent_points: int = 0,
) -> list[dict]:
    DEPTH_PENALTY = 50
    MIN_COMMENT_LENGTH = 30
    results = []
    for child in children:
        if not isinstance(child, dict) or child.get("type") != "comment":
            continue
        points = child.get("points") or 0
        if depth > 0 and points == 0:
            points = parent_points
        score = -points + depth * DEPTH_PENALTY
        text = child.get("text", "")
        if text:
            clean = clean_text(text, min_len=MIN_COMMENT_LENGTH)
            if clean:
                results.append({"text": clean, "score": score})
        if depth < max_depth and child.get("children"):
            results.extend(
                _extract_comments_recursive(
                    child["children"], depth + 1, max_depth, parent_points=points
                )
            )
    return results


def compose_story_text(
    title: str,
    self_text: str = "",
    comments: str = "",
    article_body: str = "",
) -> str:
    clean_title = clean_text(title)
    clean_self = clean_text(self_text)[:6000]
    clean_comments = clean_text(comments)[:6000]
    clean_article = clean_text(article_body)[:4000]

    parts = []
    if clean_title:
        parts.append(f"{clean_title}.")
    if clean_self:
        parts.append(clean_self)
    if clean_article:
        parts.append(clean_article)
    if clean_comments:
        parts.append(clean_comments)

    return " ".join(parts).strip()


# Algolia Fetching
def _empty_story(sid: int) -> Story:
    return Story(
        id=sid, title="", url=None, score=0, time=0, text_content="", source="hn"
    )


async def fetch_story(
    client: httpx.AsyncClient, sid: int, db: Database
) -> Story | None:
    story = db.get_story(sid)
    if story is not None:
        if story.text_content == "":
            if story.title == "":
                pass  # corrupted _empty_story, fall through to API re-fetch
            else:
                return None
        comments_stale = story.top_comments == "" or (story.comment_count or 0) > (
            story.comment_count_at_fetch or 0
        )
        if not comments_stale:
            return story
        if story.top_comments != "" and (story.comment_count_at_fetch or 0) > 50:
            return story

    url = f"https://hn.algolia.com/api/v1/items/{sid}"
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            if story is None:
                db.upsert_story(_empty_story(sid))
            return story if story else None

        item = resp.json()
        if not item or item.get("type") != "story":
            if story is None:
                db.upsert_story(_empty_story(sid))
            return story if story else None

        title = html.unescape(item.get("title", ""))
        story_url = item.get("url")
        score = item.get("points") or 0
        comment_count = item.get("num_comments")
        created_at = item.get("created_at_i") or 0
        story_text = clean_text(str(item.get("story_text") or item.get("text") or ""))

        children = item.get("children", [])
        all_comments = _extract_comments_recursive(children)
        all_comments.sort(key=lambda x: x["score"])
        selected = all_comments[:24]
        top_comment_texts = " ".join(c["text"] for c in selected)[:6000]

        text_content = compose_story_text(
            title=title,
            self_text=story_text,
            comments=top_comment_texts,
            article_body="",
        )

        if not text_content:
            if story is None:
                db.upsert_story(
                    Story(
                        id=sid,
                        title="",
                        url=None,
                        score=0,
                        time=0,
                        text_content="",
                        source="hn",
                    )
                )
            return story if story else None

        story = Story(
            id=sid,
            title=title,
            url=story_url or None,
            score=score,
            time=created_at,
            text_content=text_content,
            source="hn",
            comment_count=comment_count
            if comment_count is not None
            else len(all_comments),
            discussion_url=f"https://news.ycombinator.com/item?id={sid}",
            comment_count_at_fetch=comment_count
            if comment_count is not None
            else len(all_comments),
            self_text=story_text,
            top_comments=top_comment_texts,
            article_body="",
        )

        db.upsert_story(story)
        return story
    except Exception as e:
        logging.error(f"Error fetching story {sid}: {e}")
        return story if story else None


async def refetch_story_text(
    client: httpx.AsyncClient,
    db: Database,
    embedder: Embedder,
    story_id: int,
    current_count: int,
) -> Story | None:
    """Re-fetch comment text from Algolia items API, recompose text_content, re-embed.

    Used during regen when a story's comment_count has grown >= 30% since the last
    text fetch. Returns the updated Story, or None on failure (caller keeps stale data).

    Safety: caller is responsible for excluding stories with feedback to avoid
    invalidating the cached embedding for voted stories.
    """
    try:
        resp = await client.get(f"https://hn.algolia.com/api/v1/items/{story_id}")
        if resp.status_code != 200:
            logging.warning(
                f"refetch_story_text({story_id}): algolia status={resp.status_code}, "
                f"keeping stale text_content"
            )
            return None

        item = resp.json()
        if not item or item.get("type") != "story":
            logging.warning(
                f"refetch_story_text({story_id}): algolia returned non-story, "
                f"keeping stale text_content"
            )
            return None

        children = item.get("children", [])
        all_comments = _extract_comments_recursive(children)
        all_comments.sort(key=lambda x: x["score"])
        selected = all_comments[:24]
        top_comment_texts = " ".join(c["text"] for c in selected)[:6000]

        existing = db.get_story(story_id)
        if existing is None:
            return None

        story_text = clean_text(str(item.get("story_text") or item.get("text") or ""))
        new_text_content = compose_story_text(
            title=item.get("title", "") or existing.title,
            self_text=story_text,
            comments=top_comment_texts,
            article_body=existing.article_body,
        )
        if not new_text_content:
            logging.warning(
                f"refetch_story_text({story_id}): empty composed text, "
                f"keeping stale text_content"
            )
            return None

        updated = Story(
            id=story_id,
            title=html.unescape(item.get("title", "")) or existing.title,
            url=item.get("url") or existing.url,
            score=item.get("points") or existing.score,
            time=item.get("created_at_i") or existing.time,
            text_content=new_text_content,
            source="hn",
            comment_count=item.get("num_comments") or current_count,
            discussion_url=existing.discussion_url,
            comment_count_at_fetch=item.get("num_comments") or current_count,
            self_text=story_text,
            top_comments=top_comment_texts,
            article_body=existing.article_body,
        )
        db.upsert_story(updated)

        # Re-embed the new text_content. Force a fresh computation by
        # overwriting the cached embedding.
        new_vec = embedder.encode([new_text_content])[0]
        import hashlib
        new_hash = hashlib.sha256(new_text_content.encode("utf-8")).hexdigest()
        db.upsert_embedding(story_id, "all-MiniLM-L6-v2|mean|norm|512", new_hash, new_vec)

        return updated
    except Exception as e:
        logging.warning(
            f"refetch_story_text({story_id}): failed ({e!r}), keeping stale text_content"
        )
        return None


async def fetch_stories_by_id(
    ids: list[int], db: Database, client: httpx.AsyncClient | None = None
) -> list[Story]:
    if not ids:
        return []

    stories = db.get_stories(ids)
    valid_stories: list[Story] = []
    found_ids: set[int] = set()
    for s in stories:
        if s.text_content == "":
            continue
        valid_stories.append(s)
        comments_fresh = s.top_comments != "" and (s.comment_count or 0) <= (
            s.comment_count_at_fetch or 0
        )
        if comments_fresh:
            found_ids.add(s.id)
    missing_ids = [sid for sid in ids if sid not in found_ids]

    # Corrupted stories first, then cap stale-comment refetches at 100
    if missing_ids:
        db_ids = {s.id for s in stories}
        story_map = {s.id: s for s in stories}
        corrupted_refetch = [
            sid
            for sid in missing_ids
            if sid in db_ids
            and story_map[sid].title == ""
            and story_map[sid].text_content != ""
        ]
        stale_refetch = sorted(
            [
                sid
                for sid in missing_ids
                if sid in db_ids and sid not in corrupted_refetch
            ],
            reverse=True,
        )[:100]
        new_ids = [sid for sid in missing_ids if sid not in db_ids]
        missing_ids = corrupted_refetch + stale_refetch + new_ids

    if not missing_ids:
        return valid_stories

    created_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)
        created_client = True

    try:
        sem = asyncio.Semaphore(10)

        async def _fetch_and_cache(sid: int) -> Story | None:
            async with sem:
                return await fetch_story(client, sid, db)

        tasks = [_fetch_and_cache(sid) for sid in missing_ids]
        fetched = await asyncio.gather(*tasks)

        for s in fetched:
            if s:
                valid_stories.append(s)
    finally:
        if created_client:
            await client.aclose()

    return valid_stories


def _is_refetch_eligible(
    sid: int,
    comment_count_at_fetch: int,
    new_comment_count: int | None,
    story_time: int,
    feedback_ids: set[int],
    now_ts: int,
) -> tuple[bool, int]:
    """Pure eligibility check for refetch_story_text.

    Returns (eligible, new_comment_count). When new_comment_count is None
    (algolia search did not include num_comments), the story is not eligible
    and 0 is returned for the new count.
    """
    if new_comment_count is None or new_comment_count <= comment_count_at_fetch:
        return False, 0
    if comment_count_at_fetch <= 0:
        return False, 0
    growth = (new_comment_count - comment_count_at_fetch) / comment_count_at_fetch
    if growth < COMMENT_GROWTH_THRESHOLD:
        return False, 0
    max_age_seconds = COMMENT_REFETCH_MAX_AGE_HOURS * 3600
    if (now_ts - story_time) >= max_age_seconds:
        return False, 0
    if sid in feedback_ids:
        return False, 0
    return True, new_comment_count


def _select_refetch_ids(
    candidates: list[Story],
    fresh_metadata: dict[int, dict],
    feedback_ids: set[int],
    now_ts: int,
) -> list[int]:
    """Select up to MAX_REFETCH_PER_REGEN story IDs to refetch, ordered by
    growth ratio (highest first)."""
    eligible: list[tuple[float, int, int]] = []
    for s in candidates:
        if s.id not in fresh_metadata:
            continue
        meta = fresh_metadata[s.id]
        new_comments = meta.get("comment_count")
        is_ok, new_count = _is_refetch_eligible(
            sid=s.id,
            comment_count_at_fetch=s.comment_count_at_fetch,
            new_comment_count=new_comments,
            story_time=s.time,
            feedback_ids=feedback_ids,
            now_ts=now_ts,
        )
        if is_ok:
            growth = (new_count - s.comment_count_at_fetch) / s.comment_count_at_fetch
            eligible.append((growth, s.id, new_count))

    eligible.sort(key=lambda x: x[0], reverse=True)
    return [sid for _, sid, _ in eligible[:MAX_REFETCH_PER_REGEN]]


async def fetch_candidates(
    config: Config,
    exclude_ids: set[int],
    exclude_urls: set[str],
    db: Database,
    embedder: Embedder | None = None,
) -> tuple[list[Story], int]:
    candidate_ids = set()
    fresh_metadata = {}
    now_ts = int(time.time())
    cutoff_ts = now_ts - (config.days * 86400)
    live_start_ts = now_ts - (7 * 86400)

    # 1. Archive Window from DB
    rows = db.execute(
        "SELECT id FROM stories WHERE source = 'hn' AND time >= ? AND time < ?",
        (cutoff_ts, live_start_ts),
    )
    for row in rows:
        sid = row[0]
        if sid not in exclude_ids:
            candidate_ids.add(sid)

    # 2. Live Window daily chunks from Algolia
    async with httpx.AsyncClient(timeout=30.0) as client:
        for day in range(7):
            end_ts = now_ts - (day * 86400)
            start_ts = now_ts - ((day + 1) * 86400)
            filters = [
                f"created_at_i>={start_ts}",
                f"created_at_i<{end_ts}",
            ]
            page = 0
            target_hits = 350
            day_ids = []

            while len(day_ids) < target_hits:
                params = {
                    "tags": "story",
                    "numericFilters": ",".join(filters),
                    "hitsPerPage": 100,
                    "page": page,
                }
                try:
                    resp = await client.get(
                        "https://hn.algolia.com/api/v1/search", params=params
                    )
                    if resp.status_code != 200:
                        break
                    data = resp.json()
                    hits = data.get("hits", [])
                    if not hits:
                        break
                    for h in hits:
                        oid = int(h["objectID"])
                        points = int(h.get("points") or 0)
                        if points <= 5:
                            continue
                        day_ids.append(oid)
                        fresh_metadata[oid] = {
                            "score": points,
                            "comment_count": h.get("num_comments"),
                        }
                    page += 1
                    if len(hits) < 100:
                        break
                except Exception:
                    break

            for oid in day_ids:
                if oid not in exclude_ids:
                    candidate_ids.add(oid)

        candidates = await fetch_stories_by_id(list(candidate_ids), db, client)

        # Refresh points and comment counts for HN candidates using Algolia search response data
        for i, s in enumerate(candidates):
            if s.id in fresh_metadata:
                meta = fresh_metadata[s.id]
                has_changes = False
                new_score = s.score
                new_comments = s.comment_count
                if s.score != meta["score"]:
                    new_score = meta["score"]
                    has_changes = True
                if (
                    meta["comment_count"] is not None
                    and s.comment_count != meta["comment_count"]
                ):
                    new_comments = meta["comment_count"]
                    has_changes = True
                if has_changes:
                    updated_s = replace(s, score=new_score, comment_count=new_comments)
                    candidates[i] = updated_s
                    db.upsert_story(updated_s)

        # Detect growth candidates and refetch comment text for top stories.
        # Triggers refetch when comment_count has grown >= COMMENT_GROWTH_THRESHOLD
        # since the last text fetch, story is < 24h old, and story has no feedback
        # (feedback exclusion protects the 1,647-row training contract from
        # embedding churn).
        if embedder is not None and candidates:
            now_ts_local = int(time.time())
            refetch_ids = _select_refetch_ids(
                candidates=candidates,
                fresh_metadata=fresh_metadata,
                feedback_ids=exclude_ids,
                now_ts=now_ts_local,
            )
            for sid in refetch_ids:
                try:
                    updated = await refetch_story_text(
                        client, db, embedder, sid, fresh_metadata[sid]["comment_count"]
                    )
                    if updated is not None:
                        for i, c in enumerate(candidates):
                            if c.id == sid:
                                candidates[i] = updated
                                break
                except Exception as e:
                    logging.warning(f"refetch_story_text({sid}) raised: {e!r}")

    # Dedup
    deduped_candidates = []
    seen_urls = set()
    for s in candidates:
        if s.url:
            if s.url in exclude_urls or s.url in seen_urls:
                continue
            seen_urls.add(s.url)
        deduped_candidates.append(s)

    # 3. RSS feeds
    if config.rss.enabled:
        rss_stories = await fetch_rss_feeds(
            feeds=list(config.rss.feeds),
            per_feed=config.rss.per_feed_limit,
            days=config.days,
            exclude_urls=exclude_urls | seen_urls,
            db=db,
        )
        deduped_candidates.extend(rss_stories)

    return deduped_candidates, len(deduped_candidates)


# RSS Fetching
async def fetch_rss_feeds(
    feeds: list[str],
    per_feed: int,
    days: int,
    exclude_urls: set[str],
    db: Database,
) -> list[Story]:
    now = time.time()
    cutoff = now - (days * 86400)

    async def fetch_and_parse(feed_url: str) -> list[Story]:
        try:
            domain = urlparse(feed_url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
            source_name = f"rss_{domain.replace('.', '_')}"

            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                resp = await client.get(feed_url)
                if resp.status_code != 200:
                    return []
                content = resp.text

            parsed = feedparser.parse(content)
            stories = []

            for entry in parsed.entries[:per_feed]:
                link = entry.get("link")
                if not link:
                    continue
                if link in exclude_urls:
                    continue

                published_parsed = entry.get("published_parsed")
                if published_parsed:
                    pub_time = time.mktime(published_parsed)
                else:
                    updated_parsed = entry.get("updated_parsed")
                    if updated_parsed:
                        pub_time = time.mktime(updated_parsed)
                    else:
                        pub_time = now

                if pub_time < cutoff:
                    continue

                title = entry.get("title", "Untitled")

                summary = ""
                if "content" in entry and entry.content:
                    summary = entry.content[0].value
                elif "summary" in entry:
                    summary = entry.summary

                clean_summary = clean_text(summary)
                snippet = clean_summary[:1000]
                text_content = f"{title}. {snippet}".strip()

                h = hashlib.md5(link.encode("utf-8")).digest()
                val = int.from_bytes(h[:4], "big")
                synthetic_id = -(val % (2**31))

                story = Story(
                    id=synthetic_id,
                    title=title,
                    url=link,
                    score=0,
                    time=int(pub_time),
                    text_content=text_content,
                    source=source_name,
                    comment_count=None,
                    discussion_url=None,
                )
                stories.append(story)

            return stories
        except Exception as e:
            logging.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []

    tasks = [fetch_and_parse(f) for f in feeds]
    feed_results = await asyncio.gather(*tasks)

    all_stories = []
    for res in feed_results:
        for s in res:
            db.upsert_story(s)
            all_stories.append(s)

    return all_stories


# Embedder


class Embedder:
    def __init__(self, model_dir: str = "onnx_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(
            str(Path(model_dir) / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.max_tokens = 512

    def encode(self, texts: list[str], batch_size: int = 64) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="np",
            )

            onnx_inputs = {}
            for input_meta in self.session.get_inputs():
                name = input_meta.name
                if name in inputs:
                    onnx_inputs[name] = inputs[name]

            outputs = self.session.run(None, onnx_inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"]

            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(
                np.float32
            )
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(
                np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
            )
            mean_embeddings = sum_embeddings / sum_mask

            norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            normalized_embeddings = mean_embeddings / norms

            embeddings.append(normalized_embeddings)

        return np.concatenate(embeddings, axis=0)


def get_or_compute_embeddings(
    stories: list[Story],
    embedder: Embedder,
    db: Database,
) -> NDArray[np.float32]:
    if not stories:
        return np.empty((0, 384), dtype=np.float32)

    import hashlib
    story_hashes = {
        s.id: hashlib.sha256(s.text_content.encode("utf-8")).hexdigest()
        for s in stories
    }

    ids = [s.id for s in stories]
    model_version = "all-MiniLM-L6-v2|mean|norm|512"

    cached = db.get_embeddings_batch(ids, model_version, story_hashes)
    missing_stories = [s for s in stories if s.id not in cached]

    if missing_stories:
        texts = [s.text_content for s in missing_stories]
        computed = embedder.encode(texts)
        for s, vec in zip(missing_stories, computed):
            db.upsert_embedding(s.id, model_version, story_hashes[s.id], vec)
            cached[s.id] = vec

    return np.array([cached[story_id] for story_id in ids], dtype=np.float32)


# Ranking

# Normalization constants for metadata features
_LOG_POINTS_SCALE = 8.0  # log1p(~3000) ≈ 8
_AGE_DAYS_SCALE = 30.0  # cap at 30 days
_LOG_COMMENTS_SCALE = 7.0  # log1p(~1000) ≈ 6.9
_LOG_TEXTLEN_SCALE = 12.0  # log1p(~100000) ≈ 11.5
_LOG_QUALITY_SCALE = 8.0  # log1p(score/(age_hours+1)) rarely exceeds 8
_LOG_VELOCITY_SCALE = 8.0  # log1p(score/max(age_h, 0.1)) rarely exceeds 8


def _augment_features(
    embeddings: NDArray[np.float32],
    scores: list[int] | np.ndarray,
    age_seconds: list[float] | np.ndarray,
    comment_counts: np.ndarray | None = None,
    text_lengths: np.ndarray | None = None,
    hn_quality: np.ndarray | None = None,
    score_velocity: np.ndarray | None = None,
    comment_velocity: np.ndarray | None = None,
    sim_to_upvoted: np.ndarray | None = None,
    sim_to_downvoted: np.ndarray | None = None,
    closest_upvoted: np.ndarray | None = None,
    closest_downvoted: np.ndarray | None = None,
    comment_score_ratio: np.ndarray | None = None,
) -> NDArray[np.float32]:
    n = len(scores)
    n_meta = 1
    for f in (comment_counts, text_lengths, hn_quality, comment_score_ratio):
        if f is not None:
            n_meta += 1
    if score_velocity is not None and comment_velocity is not None:
        n_meta += 2
    if sim_to_upvoted is not None:
        n_meta += 4

    meta = np.zeros((n, n_meta), dtype=np.float32)
    col = 0

    # log points
    meta[:, col] = (
        np.clip(np.log1p(np.maximum(scores, 0)), 0, _LOG_POINTS_SCALE)
        / _LOG_POINTS_SCALE
    )
    col += 1

    if comment_counts is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(comment_counts, 0)), 0, _LOG_COMMENTS_SCALE)
            / _LOG_COMMENTS_SCALE
        )
        col += 1

    if text_lengths is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(text_lengths, 0)), 0, _LOG_TEXTLEN_SCALE)
            / _LOG_TEXTLEN_SCALE
        )
        col += 1

    if hn_quality is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(hn_quality, 0)), 0, _LOG_QUALITY_SCALE)
            / _LOG_QUALITY_SCALE
        )
        col += 1

    if comment_score_ratio is not None:
        meta[:, col] = comment_score_ratio
        col += 1

    if score_velocity is not None and comment_velocity is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(score_velocity, 0)), 0, _LOG_VELOCITY_SCALE)
            / _LOG_VELOCITY_SCALE
        )
        col += 1
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(comment_velocity, 0)), 0, _LOG_VELOCITY_SCALE)
            / _LOG_VELOCITY_SCALE
        )
        col += 1

    if sim_to_upvoted is not None:
        assert (
            sim_to_downvoted is not None
            and closest_upvoted is not None
            and closest_downvoted is not None
        )
        meta[:, col] = (np.clip(sim_to_upvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(sim_to_downvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(closest_upvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(closest_downvoted, -1, 1) + 1) / 2
        col += 1

    return np.concatenate([embeddings, meta], axis=1)


def rank_stories(
    candidates: list[Story],
    candidate_embeddings: NDArray[np.float32],
    db: Database,
    config: Config,
    embedder: Embedder,
    user_id: int | None = None,
) -> list[RankedStory]:
    if not candidates:
        return []

    now = time.time()
    scores = None
    probs = None
    feedback_stories, feedback_labels, vote_times = db.get_feedback_for_training(user_id=user_id)

    # Multiclass SVM: 0=down, 1=neutral, 2=up
    unique_classes = set(feedback_labels)
    if len(feedback_labels) >= 10 and len(unique_classes) >= 2:
        try:
            cand_scores = np.array([s.score for s in candidates])
            cand_ages = np.array([now - s.time for s in candidates])
            cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
            fb_embeddings = get_or_compute_embeddings(feedback_stories, embedder, db)

            # Personalization: mean/closest per class from ALL real feedback
            fb_labels_arr = np.array(feedback_labels)
            up_mask = fb_labels_arr == 2
            down_mask = fb_labels_arr == 0
            fb_up_embs = fb_embeddings[up_mask]
            fb_down_embs = fb_embeddings[down_mask]

            mean_up = (
                fb_up_embs.mean(axis=0)
                if up_mask.any()
                else np.zeros(384, dtype=np.float32)
            )
            mean_down = (
                fb_down_embs.mean(axis=0)
                if down_mask.any()
                else np.zeros(384, dtype=np.float32)
            )

            n_up = int(up_mask.sum())
            fb_sim_to_up = fb_embeddings @ mean_up
            if n_up > 0:
                if n_up > 1:
                    fb_sim_to_up[up_mask] = (n_up * fb_sim_to_up[up_mask] - 1.0) / (
                        n_up - 1
                    )
                else:
                    fb_sim_to_up[up_mask] = 0.0

                sim_up_mat = fb_embeddings @ fb_up_embs.T
                if n_up > 1:
                    train_up_positions = np.where(up_mask)[0]
                    for idx, tp in enumerate(train_up_positions):
                        sim_up_mat[tp, idx] = -1.0
                fb_closest_up = np.max(sim_up_mat, axis=1)
            else:
                fb_closest_up = np.zeros(len(fb_embeddings), dtype=np.float32)

            n_down = int(down_mask.sum())
            fb_sim_to_down = fb_embeddings @ mean_down
            if n_down > 0:
                if n_down > 1:
                    fb_sim_to_down[down_mask] = (
                        n_down * fb_sim_to_down[down_mask] - 1.0
                    ) / (n_down - 1)
                else:
                    fb_sim_to_down[down_mask] = 0.0

                sim_down_mat = fb_embeddings @ fb_down_embs.T
                if n_down > 1:
                    train_down_positions = np.where(down_mask)[0]
                    for idx, tp in enumerate(train_down_positions):
                        sim_down_mat[tp, idx] = -1.0
                fb_closest_down = np.max(sim_down_mat, axis=1)
            else:
                fb_closest_down = np.zeros(len(fb_embeddings), dtype=np.float32)

            cand_sim_to_up = candidate_embeddings @ mean_up
            cand_sim_to_down = candidate_embeddings @ mean_down
            cand_closest_up = (
                np.max(candidate_embeddings @ fb_up_embs.T, axis=1)
                if up_mask.any()
                else np.zeros(len(candidates))
            )
            cand_closest_down = (
                np.max(candidate_embeddings @ fb_down_embs.T, axis=1)
                if down_mask.any()
                else np.zeros(len(candidates))
            )

            # Augment training features: age_at_vote = vote_time - story_time
            fb_scores = np.array([s.score for s in feedback_stories])
            fb_ages = np.array(
                [vt - s.time for vt, s in zip(vote_times, feedback_stories)]
            )
            fb_comment_counts = np.array(
                [s.comment_count or 0 for s in feedback_stories]
            )
            fb_text_lengths = np.array([len(s.text_content) for s in feedback_stories])
            fb_quality = fb_scores / (np.maximum(fb_ages / 3600.0, 0) + 1)
            fb_age_hours = fb_ages / 3600.0
            fb_safe_h = np.maximum(fb_age_hours, 0.1)
            fb_score_vel = fb_scores / fb_safe_h
            fb_comment_vel = fb_comment_counts / fb_safe_h

            fb_csr_ratio = fb_comment_counts / np.maximum(fb_scores, 1)
            fb_csr = np.clip(np.log1p(fb_csr_ratio), 0, 3.0) / 3.0

            fb_features = _augment_features(
                fb_embeddings,
                fb_scores,
                fb_ages,
                comment_counts=fb_comment_counts,
                text_lengths=fb_text_lengths,
                hn_quality=fb_quality,
                score_velocity=fb_score_vel,
                comment_velocity=fb_comment_vel,
                sim_to_upvoted=fb_sim_to_up,
                sim_to_downvoted=fb_sim_to_down,
                closest_upvoted=fb_closest_up,
                closest_downvoted=fb_closest_down,
                comment_score_ratio=fb_csr,
            )

            # Ensure all three classes (0, 1, 2) are present
            missing = {0, 1, 2} - set(feedback_labels)
            if missing:
                fb_features = np.concatenate(
                    [
                        fb_features,
                        np.zeros(
                            (len(missing), fb_features.shape[1]), dtype=np.float32
                        ),
                    ],
                    axis=0,
                )
                labels = list(feedback_labels) + list(missing)
            else:
                labels = list(feedback_labels)

            # Compute balanced weights for real feedback; 1e-6 for dummies
            counts = Counter(feedback_labels)
            n_classes = len(counts)
            n_real = len(feedback_labels)
            weights = [n_real / (n_classes * counts[lbl]) for lbl in feedback_labels]
            weights.extend([1e-6] * len(missing))
            sample_weights = np.array(weights, dtype=np.float64)

            emb_dim = candidate_embeddings.shape[1]
            scaler = StandardScaler()
            fb_features_meta_scaled = np.clip(scaler.fit_transform(fb_features[:, emb_dim:]), -2.5, 2.5)

            fb_features_scaled = np.hstack(
                [fb_features[:, :emb_dim], fb_features_meta_scaled]
            )

            svm = SVC(
                C=config.model.svm_c,
                kernel=config.model.svm_kernel,
                gamma=config.model.svm_gamma,
                random_state=0,
                decision_function_shape="ovr",
                probability=False,
            )
            svm.fit(fb_features_scaled, labels, sample_weight=sample_weights)

            # Augment candidate features: age_now = now - story_time
            cand_text_lengths = np.array([len(s.text_content) for s in candidates])
            cand_quality = cand_scores / (np.maximum(cand_ages / 3600.0, 0) + 1)
            cand_age_hours = cand_ages / 3600.0
            cand_safe_h = np.maximum(cand_age_hours, 0.1)
            cand_score_vel = cand_scores / cand_safe_h
            cand_comment_vel = cand_comment_counts / cand_safe_h

            cand_csr_ratio = cand_comment_counts / np.maximum(cand_scores, 1)
            cand_csr = np.clip(np.log1p(cand_csr_ratio), 0, 3.0) / 3.0

            cand_features = _augment_features(
                candidate_embeddings,
                cand_scores,
                cand_ages,
                comment_counts=cand_comment_counts,
                text_lengths=cand_text_lengths,
                hn_quality=cand_quality,
                score_velocity=cand_score_vel,
                comment_velocity=cand_comment_vel,
                sim_to_upvoted=cand_sim_to_up,
                sim_to_downvoted=cand_sim_to_down,
                closest_upvoted=cand_closest_up,
                closest_downvoted=cand_closest_down,
                comment_score_ratio=cand_csr,
            )
            cand_features_meta_scaled = np.clip(scaler.transform(cand_features[:, emb_dim:]), -2.5, 2.5)
            cand_features_scaled = np.hstack(
                [cand_features[:, :emb_dim], cand_features_meta_scaled]
            )

            df_cand = svm.decision_function(cand_features_scaled)
            e_x = np.exp(df_cand - np.max(df_cand, axis=1, keepdims=True))
            probs = e_x / e_x.sum(axis=1, keepdims=True)
            class_order = list(svm.classes_)
            idx_up = class_order.index(2)
            idx_neutral = class_order.index(1)
            scores = (
                probs[:, idx_up] + config.model.neutral_weight * probs[:, idx_neutral]
            )
        except Exception as e:
            logging.error(f"Failed to fit feedback SVM: {e}")

    # Three-tier fallback based on feedback count
    n_feedback = len(feedback_labels)

    if scores is None:
        if n_feedback == 0:
            # Tier 1: No votes → HN gravity (frontpage-like)
            scores = np.array([
                (s.score if s.source == "hn" else s.score * 2) /
                max(((now - s.time) / 3600.0 + 2.0) ** 1.8, 0.1)
                for s in candidates
            ], dtype=np.float32)
            if scores.max() > 0:
                scores = scores / scores.max()
        else:
            # Tier 2: At least 1 vote → similarity to upvotes minus downvotes
            fb_embs = get_or_compute_embeddings(feedback_stories, embedder, db)
            fb_labels_arr = np.array(feedback_labels)
            up_mask = fb_labels_arr == 2
            down_mask = fb_labels_arr == 0

            if up_mask.any() or down_mask.any():
                up_emb = fb_embs[up_mask].mean(axis=0) if up_mask.any() else np.zeros(384, dtype=np.float32)
                down_emb = fb_embs[down_mask].mean(axis=0) if down_mask.any() else np.zeros(384, dtype=np.float32)

                sim_up = candidate_embeddings @ up_emb
                sim_down = candidate_embeddings @ down_emb
                scores = sim_up - sim_down
                # Shift to 0-1 range
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                scores = np.full(len(candidates), 0.5, dtype=np.float32)

    ranked = []
    if probs is not None:
        try:
            idx_down = class_order.index(0)
            idx_neutral = class_order.index(1)
            idx_up = class_order.index(2)
            for idx, (s, score) in enumerate(zip(candidates, scores)):
                ranked.append(
                    RankedStory(
                        story=s,
                        score=float(score),
                        best_match_title="",
                        prob_down=float(probs[idx, idx_down]),
                        prob_neutral=float(probs[idx, idx_neutral]),
                        prob_up=float(probs[idx, idx_up]),
                    )
                )
        except (ValueError, IndexError, NameError) as e:
            logging.error(f"Error mapping probability class indices: {e}")
            ranked = []

    if not ranked:
        for s, score in zip(candidates, scores):
            ranked.append(
                RankedStory(
                    story=s,
                    score=float(score),
                    best_match_title="",
                )
            )

    if len(feedback_labels) == 0:
        return sorted(ranked, key=lambda x: x.story.score, reverse=True)
    else:
        return sorted(ranked, key=lambda x: x.score, reverse=True)


# MMR
def mmr_filter(
    ranked: list[RankedStory],
    embeddings_map: dict[int, NDArray[np.float32]],
    threshold: float = 0.85,
    limit: int = 40,
) -> list[RankedStory]:
    selected = []
    discarded = set()

    for idx, item in enumerate(ranked):
        if item.story.id in discarded:
            continue

        emb = embeddings_map.get(item.story.id)
        selected.append(item)

        if emb is not None:
            for other in ranked[idx + 1 :]:
                if other.story.id in discarded:
                    continue
                other_emb = embeddings_map.get(other.story.id)
                if other_emb is not None:
                    sim = float(np.dot(emb, other_emb))
                    if sim > threshold:
                        discarded.add(other.story.id)

        if len(selected) >= limit:
            break

    # Sort selected items back to their original relative order in ranked
    selected.sort(key=lambda x: ranked.index(x))
    return selected


# HTML Render
def time_ago_filter(seconds: int) -> str:
    diff = int(time.time()) - seconds
    if diff < 0:
        return "now"
    if diff < 60:
        return f"{diff}s ago"
    minutes = diff // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def generate_dashboard(
    ranked: list[RankedStory],
    output_path: Path,
    username: str,
    timestamp: str,
    server_port: int,
    db: Database,
    user_id: int | None = None,
) -> None:
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["time_ago"] = time_ago_filter

    pico_css_path = Path("templates/pico.min.css")
    pico_css = (
        pico_css_path.read_text(encoding="utf-8") if pico_css_path.exists() else ""
    )

    # Map user feedback in database for active UI state highlighting
    all_fb = db.get_all_feedback(user_id=user_id)
    fb_map = {f.story_id: f.action for f in all_fb}

    template = env.get_template("index.html")
    html_content = template.render(
        username=username,
        timestamp=timestamp,
        stories=ranked,
        server_port=server_port,
        pico_css=pico_css,
        fb_map=fb_map,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")


def rerank_candidates(
    db: Database,
    config: Config,
    embedder: Embedder,
    candidates: list[Story],
    cand_embeddings: NDArray[np.float32] = None,
    user_id: int | None = None,
) -> list[RankedStory]:
    if not candidates:
        return []

    if cand_embeddings is None:
        cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    ranked = rank_stories(
        candidates,
        cand_embeddings,
        db,
        config,
        embedder,
        user_id=user_id,
    )

    embeddings_map = {s.id: vec for s, vec in zip(candidates, cand_embeddings)}

    num_uncertain = 5 if config.count >= 10 else 0
    limit = max(1, config.count - num_uncertain)
    final = mmr_filter(
        ranked,
        embeddings_map,
        threshold=config.model.diversity_threshold,
        limit=limit,
    )

    selected_ids = {item.story.id for item in final}
    remaining = [r for r in ranked if r.story.id not in selected_ids]

    # Calculate parameters for remaining discovery passes
    feedback_stories, feedback_labels, _ = db.get_feedback_for_training(user_id=user_id)
    fb_labels_arr = np.array(feedback_labels)
    up_mask = fb_labels_arr == 2
    down_mask = fb_labels_arr == 0
    fb_embeddings = get_or_compute_embeddings(feedback_stories, embedder, db)
    fb_up_embs = fb_embeddings[up_mask]
    fb_down_embs = fb_embeddings[down_mask]

    cand_closest_up = (
        np.max(cand_embeddings @ fb_up_embs.T, axis=1)
        if up_mask.any()
        else np.zeros(len(candidates))
    )
    cand_closest_down = (
        np.max(cand_embeddings @ fb_down_embs.T, axis=1)
        if down_mask.any()
        else np.zeros(len(candidates))
    )
    cand_max_sim = np.maximum(cand_closest_up, cand_closest_down)
    sim_threshold = np.percentile(cand_max_sim, 15) if len(cand_max_sim) else 0.0
    similar_threshold = np.percentile(cand_closest_up, 90) if len(cand_closest_up) else 0.0

    cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
    cand_scores = np.array([s.score for s in candidates])
    now_ts = time.time()
    cand_velocities = np.array(
        [s.score / max((now_ts - s.time) / 3600.0, 0.1) for s in candidates]
    )
    hot_threshold = np.percentile(cand_velocities, 98) if len(cand_velocities) else 0

    story_id_to_idx = {s.id: idx for idx, s in enumerate(candidates)}

    # Determine uncertainty IDs among remaining candidates
    def get_entropy(r: RankedStory) -> float:
        ent = 0.0
        for p in (r.prob_down, r.prob_neutral, r.prob_up):
            if p is not None and p > 1e-9:
                ent -= p * np.log2(p)
        return ent

    if remaining and num_uncertain > 0:
        uncertain_candidates = [r for r in remaining if r.prob_down is not None]
        uncertain_candidates.sort(key=get_entropy, reverse=True)
        uncertain_ids = {r.story.id for r in uncertain_candidates[:num_uncertain]}
    else:
        uncertain_ids = set()

    # Assign badges to all candidates in ranked
    discussion_threshold = (
        np.percentile(cand_comment_counts, 93) if len(cand_comment_counts) else 0
    )
    engagement_threshold = np.percentile(cand_scores, 95) if len(cand_scores) else 0

    cand_entropies = [get_entropy(r) for r in ranked if r.prob_down is not None]
    uncertainty_threshold = np.percentile(cand_entropies, 90) if cand_entropies else 0.0

    ranked_decorated = []
    for r in ranked:
        idx = story_id_to_idx[r.story.id]
        entropy = get_entropy(r)
        is_uncertain = bool(
            r.prob_down is not None
            and entropy >= 1.2
            and (r.story.id in uncertain_ids or entropy >= uncertainty_threshold)
        )
        is_novel = bool(cand_max_sim[idx] <= sim_threshold and r.score > 0.1)
        is_similar = bool(cand_closest_up[idx] >= similar_threshold)
        is_discussion_rich = bool(
            cand_comment_counts[idx] >= discussion_threshold
            and cand_comment_counts[idx] > 0
        )
        is_high_engagement = bool(cand_scores[idx] >= engagement_threshold)
        is_hot = bool(
            cand_velocities[idx] >= hot_threshold and cand_velocities[idx] > 0
        )
        ranked_decorated.append(
            replace(
                r,
                is_uncertain=is_uncertain,
                is_novel=is_novel,
                is_similar=is_similar,
                is_discussion_rich=is_discussion_rich,
                is_high_engagement=is_high_engagement,
                is_hot=is_hot,
            )
        )

    # Reconstruct final from the decorated candidates
    final_ids = {item.story.id for item in final}
    final = [r for r in ranked_decorated if r.story.id in final_ids]

    # And remaining_decorated contains the decorated candidates not in final
    remaining_decorated = [r for r in ranked_decorated if r.story.id not in final_ids]

    # 1. Surface uncertainty items
    uncertain_items = [r for r in remaining_decorated if r.story.id in uncertain_ids]
    final.extend(uncertain_items)
    selected_ids = final_ids | {item.story.id for item in uncertain_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 2. Surface up to 5 novel stories
    novel_pool = [r for r in remaining_decorated if r.is_novel]
    novel_pool.sort(key=lambda r: r.score, reverse=True)
    novel_items = novel_pool[:5]
    final.extend(novel_items)
    selected_ids |= {item.story.id for item in novel_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 3. Surface up to 5 most similar stories
    similar_pool = [r for r in remaining_decorated if r.is_similar]
    similar_pool.sort(
        key=lambda r: cand_closest_up[story_id_to_idx[r.story.id]], reverse=True
    )
    similar_items = similar_pool[:5]
    final.extend(similar_items)
    selected_ids |= {item.story.id for item in similar_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 4. Surface up to 5 discussion-rich stories
    discussion_pool = [r for r in remaining_decorated if r.is_discussion_rich]
    discussion_pool.sort(key=lambda r: r.story.comment_count or 0, reverse=True)
    discussion_items = discussion_pool[:5]
    final.extend(discussion_items)
    selected_ids |= {item.story.id for item in discussion_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 5. Surface up to 5 high-engagement stories
    engagement_pool = [r for r in remaining_decorated if r.is_high_engagement]
    engagement_pool.sort(key=lambda r: r.score, reverse=True)
    engagement_items = engagement_pool[:5]
    final.extend(engagement_items)
    selected_ids |= {item.story.id for item in engagement_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 6. Surface up to 5 hot stories
    hot_pool = [r for r in remaining_decorated if r.is_hot]
    hot_pool.sort(
        key=lambda r: cand_velocities[story_id_to_idx[r.story.id]], reverse=True
    )
    hot_items = hot_pool[:5]
    final.extend(hot_items)
    selected_ids |= {item.story.id for item in hot_items}

    final.sort(key=lambda r: r.score, reverse=True)
    return final


# Orchestrator
async def run_pipeline(config: Config) -> None:
    db = Database(config.db_path)
    embedder = Embedder(config.onnx_model_dir)

    # Exclude stories that already have user feedback (voted on via dashboard)
    feedback_records = db.get_all_feedback()
    feedback_ids = {f.story_id for f in feedback_records}
    feedback_urls = {f.url for f in feedback_records if f.url}

    exclude_ids = feedback_ids
    candidates, n_fetched = await fetch_candidates(
        config, exclude_ids, feedback_urls, db, embedder
    )
    logging.info(f"Fetched {n_fetched} candidates (excluded {len(exclude_ids)})")

    t0 = time.perf_counter()
    cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logging.info(f"Embedded {cand_embeddings.shape[0]} candidates in {elapsed_ms}ms")

    ranked = rank_stories(
        candidates,
        cand_embeddings,
        db,
        config,
        embedder,
    )
    top_score = max((r.score for r in ranked), default=0.0)
    logging.info(f"Ranked {len(ranked)} stories, top_score={top_score:.3f}")

    # -- Proactive Fetching (Top Recommendations & Popularity/Velocity Triggers) --
    now_ts = time.time()
    top_candidates = []
    for pos, r in enumerate(ranked):
        s = r.story
        if not s.url or s.article_body:
            continue

        # Calculate velocity metrics
        age_hours = (now_ts - s.time) / 3600.0
        safe_hours = max(age_hours, 0.1)
        score_vel = s.score / safe_hours
        comment_vel = (s.comment_count or 0) / safe_hours

        # Trigger 1: Highly ranked recommendation (Top 40)
        is_top_rank = pos < 40

        # Trigger 2: High virality recommendation (Top 150 with high score or velocity)
        is_viral = (pos < 150) and (
            s.score > 150 or score_vel > 30.0 or comment_vel > 20.0
        )

        if is_top_rank or is_viral:
            top_candidates.append(s)
            if len(top_candidates) >= 50:  # Cap at 50 parallel fetches per run
                break

    if top_candidates:
        logging.info(
            f"Proactively fetching article bodies for top {len(top_candidates)} candidates..."
        )
        from server import _fetch_article_body

        async def fetch_and_update(story: Story):
            try:
                body = await _fetch_article_body(story.url)
                if body:
                    new_text = compose_story_text(
                        story.title,
                        story.self_text,
                        story.top_comments,
                        body[:15000],
                    )
                    updated = replace(
                        story,
                        article_body=body[:15000],
                        text_content=new_text,
                    )
                    db.upsert_story(updated)
                    return story.id, updated
            except Exception as e:
                logging.warning(f"Proactive fetch failed for story {story.id}: {e!r}")
            return story.id, None

        tasks = [fetch_and_update(s) for s in top_candidates]
        results = await asyncio.gather(*tasks)
        updated_map = {sid: updated for sid, updated in results if updated}

        if updated_map:
            for idx, s in enumerate(candidates):
                if s.id in updated_map:
                    candidates[idx] = updated_map[s.id]

            # Update cache embeddings in the SQLite DB
            model_version = "all-MiniLM-L6-v2|mean|norm|512"
            for sid, updated in updated_map.items():
                new_vec = embedder.encode([updated.text_content])[0]
                import hashlib
                new_hash = hashlib.sha256(updated.text_content.encode("utf-8")).hexdigest()
                db.upsert_embedding(sid, model_version, new_hash, new_vec)

            # Reload candidates' embeddings (loads updated vectors + cached vectors)
            cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

            # Re-run SVM classification
            ranked = rank_stories(
                candidates,
                cand_embeddings,
                db,
                config,
                embedder,
            )
            top_score = max((r.score for r in ranked), default=0.0)
            logging.info(
                f"Re-ranked {len(ranked)} stories after proactive fetches, "
                f"top_score={top_score:.3f}"
            )

    final = rerank_candidates(
        db=db,
        config=config,
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=cand_embeddings,
    )

    logging.info(f"Filtering with MMR: {len(final)} of {len(ranked)} stories kept")
    pu_vals = [r.prob_up for r in final if r.prob_up is not None]
    pu_lo = min(pu_vals) if pu_vals else 0.0
    pu_hi = max(pu_vals) if pu_vals else 0.0
    logging.info(
        f"Generating dashboard: {len(final)} stories, prob_up range {pu_lo:.3f}–{pu_hi:.3f}"
    )
    generate_dashboard(
        final,
        Path(config.output),
        config.username,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        config.server_port,
        db,
    )

    pruned = db.prune_stories(max_age_days=config.days * 2)
    logging.info(f"Pruned {pruned} old stories")
    db.close()
    logging.info("Done.")


def fast_rerank_and_render(
    db: Database,
    config: Config,
    embedder: Embedder,
) -> None:
    now_ts = int(time.time())
    cutoff_ts = now_ts - (config.days * 86400)
    rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories WHERE time >= ? AND id NOT IN (SELECT story_id FROM feedback)",
        (cutoff_ts,)
    )
    candidates = [Database._row_to_story(row) for row in rows]

    if not candidates:
        generate_dashboard(
            [],
            Path(config.output),
            config.username,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            config.server_port,
            db,
        )
        return

    # Candidate Pruning: Sort by standard HN gravity formula and take top 1000
    # to keep evaluation time under 300ms.
    candidates.sort(
        key=lambda s: (s.score if s.source == "hn" else 100) / (((time.time() - s.time) / 3600.0 + 2.0) ** 1.8),
        reverse=True,
    )
    candidates = candidates[:1000]

    cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    final = rerank_candidates(
        db=db,
        config=config,
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=cand_embeddings,
    )

    generate_dashboard(
        final,
        Path(config.output),
        config.username,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        config.server_port,
        db,
    )


def generate_dashboard_bytes(
    ranked: list[RankedStory],
    config: Config,
    username: str,
    db: Database,
    user_id: int | None = None,
) -> bytes:
    """Render dashboard to bytes without writing to disk."""
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["time_ago"] = time_ago_filter

    pico_css_path = Path("templates/pico.min.css")
    pico_css = pico_css_path.read_text(encoding="utf-8") if pico_css_path.exists() else ""

    all_fb = db.get_all_feedback(user_id=user_id)
    fb_map = {f.story_id: f.action for f in all_fb}

    template = env.get_template("index.html")
    html_content = template.render(
        username=username,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        stories=ranked,
        server_port=config.server_port,
        pico_css=pico_css,
        fb_map=fb_map,
    )
    return html_content.encode("utf-8")


def fast_rerank_for_user(
    db: Database,
    config: Config,
    embedder: Embedder,
    user_id: int,
) -> list[RankedStory]:
    """Fast rerank for a specific user. Called on each dashboard request."""
    now_ts = int(time.time())
    cutoff_ts = now_ts - (config.days * 86400)

    rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories WHERE time >= ? AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)",
        (cutoff_ts, user_id),
    )
    candidates = [Database._row_to_story(row) for row in rows]
    if not candidates:
        return []

    candidates.sort(
        key=lambda s: (s.score if s.source == "hn" else 100) / (((time.time() - s.time) / 3600.0 + 2.0) ** 1.8),
        reverse=True,
    )
    candidates = candidates[:1000]

    cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    return rerank_candidates(
        db=db, config=config, embedder=embedder,
        candidates=candidates, cand_embeddings=cand_embeddings,
        user_id=user_id,
    )


async def fetch_candidates_only(config: Config, db: Database) -> None:
    """Fetch new candidates into shared DB. No per-user rendering."""
    feedback_records = db.get_all_feedback()
    feedback_ids = {f.story_id for f in feedback_records}
    feedback_urls = {f.url for f in feedback_records if f.url}

    candidates, n_fetched = await fetch_candidates(
        config, feedback_ids, feedback_urls, db, None
    )
    logging.info(f"Regen: fetched {n_fetched} candidates")

