from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from database import Database, Story
from transformers import AutoTokenizer


@dataclass(frozen=True)
class ModelConfig:
    svm_c: float = 0.3
    svm_gamma: float | str = 0.1
    svm_kernel: str = "rbf"
    diversity_threshold: float = 0.55


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
                diversity_threshold=model_cfg.get("diversity_threshold", 0.55),
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
    comments: list[str] | None = None,
) -> str:
    clean_title = clean_text(title)
    clean_self = clean_text(self_text)[:6000]
    clean_comments = " ".join(clean_text(c) for c in (comments or []) if c)[:6000]

    parts = []
    if clean_title:
        parts.append(f"{clean_title}.")
    if clean_self:
        parts.append(clean_self)
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
            return None
        return story

    url = f"https://hn.algolia.com/api/v1/items/{sid}"
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            db.upsert_story(_empty_story(sid))
            return None

        item = resp.json()
        if not item or item.get("type") != "story":
            db.upsert_story(_empty_story(sid))
            return None

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

        text_content = compose_story_text(
            title=title,
            self_text=story_text,
            comments=[c["text"] for c in selected],
        )

        if not text_content:
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
            return None

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
        )

        db.upsert_story(story)
        return story
    except Exception as e:
        logging.error(f"Error fetching story {sid}: {e}")
        return None


async def fetch_stories_by_id(
    ids: list[int], db: Database, client: httpx.AsyncClient | None = None
) -> list[Story]:
    if not ids:
        return []

    stories = db.get_stories(ids)
    found_ids = {s.id for s in stories}
    missing_ids = [sid for sid in ids if sid not in found_ids]

    if not missing_ids:
        return stories

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
                stories.append(s)
    finally:
        if created_client:
            await client.aclose()

    return stories


async def fetch_candidates(
    config: Config,
    exclude_ids: set[int],
    exclude_urls: set[str],
    db: Database,
) -> list[Story]:
    candidate_ids = set()
    now_ts = int(time.time())
    cutoff_ts = now_ts - (config.days * 86400)
    live_start_ts = now_ts - (7 * 86400)

    # 1. Archive Window from DB
    cursor = db.conn.execute(
        """
        SELECT id, title, url, score, time, text_content, source, comment_count, discussion_url
        FROM stories WHERE source = 'hn' AND time >= ? AND time < ?
        """,
        (cutoff_ts, live_start_ts),
    )
    for row in cursor.fetchall():
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
            target_hits = 150
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
                    page += 1
                    if len(hits) < 100:
                        break
                except Exception:
                    break

            for oid in day_ids:
                if oid not in exclude_ids:
                    candidate_ids.add(oid)

        candidates = await fetch_stories_by_id(list(candidate_ids), db, client)

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

    return deduped_candidates


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
        self.max_tokens = 256

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

    ids = [s.id for s in stories]
    model_version = "all-MiniLM-L6-v2|mean|norm|256"

    cached = db.get_embeddings_batch(ids, model_version)
    missing_stories = [s for s in stories if s.id not in cached]

    if missing_stories:
        texts = [s.text_content for s in missing_stories]
        computed = embedder.encode(texts)
        for s, vec in zip(missing_stories, computed):
            db.upsert_embedding(s.id, model_version, vec)
            cached[s.id] = vec

    return np.array([cached[story_id] for story_id in ids], dtype=np.float32)


# Ranking

# Normalization constants for metadata features
_LOG_POINTS_SCALE = 8.0  # log1p(~3000) ≈ 8
_AGE_DAYS_SCALE = 30.0  # cap at 30 days
_LOG_COMMENTS_SCALE = 7.0  # log1p(~1000) ≈ 6.9
_LOG_TEXTLEN_SCALE = 12.0  # log1p(~100000) ≈ 11.5
_LOG_QUALITY_SCALE = 8.0  # log1p(score/(age_hours+1)) rarely exceeds 8


def _augment_features(
    embeddings: NDArray[np.float32],
    scores: list[int] | np.ndarray,
    age_seconds: list[float] | np.ndarray,
    comment_counts: np.ndarray | None = None,
    text_lengths: np.ndarray | None = None,
    hn_quality: np.ndarray | None = None,
    sim_to_upvoted: np.ndarray | None = None,
    sim_to_downvoted: np.ndarray | None = None,
    closest_upvoted: np.ndarray | None = None,
    closest_downvoted: np.ndarray | None = None,
) -> NDArray[np.float32]:
    n = len(scores)
    n_meta = 2
    for f in (comment_counts, text_lengths, hn_quality):
        if f is not None:
            n_meta += 1
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

    # age days
    age_days = np.maximum(age_seconds, 0) / 86400.0
    meta[:, col] = np.clip(age_days, 0, _AGE_DAYS_SCALE) / _AGE_DAYS_SCALE
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
) -> list[RankedStory]:
    if not candidates:
        return []

    now = time.time()
    scores = None
    feedback_stories, feedback_labels, vote_times = db.get_feedback_for_training()

    # Multiclass SVM: 0=down, 1=neutral, 2=up
    if len(feedback_labels) >= 1:
        try:
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

            fb_sim_to_up = fb_embeddings @ mean_up
            fb_sim_to_down = fb_embeddings @ mean_down
            fb_closest_up = (
                np.max(fb_embeddings @ fb_up_embs.T, axis=1)
                if up_mask.any()
                else np.zeros(len(fb_embeddings))
            )
            fb_closest_down = (
                np.max(fb_embeddings @ fb_down_embs.T, axis=1)
                if down_mask.any()
                else np.zeros(len(fb_embeddings))
            )

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

            fb_features = _augment_features(
                fb_embeddings,
                fb_scores,
                fb_ages,
                comment_counts=fb_comment_counts,
                text_lengths=fb_text_lengths,
                hn_quality=fb_quality,
                sim_to_upvoted=fb_sim_to_up,
                sim_to_downvoted=fb_sim_to_down,
                closest_upvoted=fb_closest_up,
                closest_downvoted=fb_closest_down,
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

            svm = SVC(
                C=config.model.svm_c,
                kernel=config.model.svm_kernel,
                gamma=config.model.svm_gamma,
                random_state=0,
                decision_function_shape="ovr",
            )
            svm.fit(fb_features, labels, sample_weight=sample_weights)
            n_train = len(fb_features)
            calibrated = CalibratedClassifierCV(
                svm, cv=[(list(range(n_train)), list(range(n_train)))], method="sigmoid"
            )
            calibrated.fit(fb_features, labels, sample_weight=sample_weights)

            # Augment candidate features: age_now = now - story_time
            cand_scores = np.array([s.score for s in candidates])
            cand_ages = np.array([now - s.time for s in candidates])
            cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
            cand_text_lengths = np.array([len(s.text_content) for s in candidates])
            cand_quality = cand_scores / (np.maximum(cand_ages / 3600.0, 0) + 1)

            cand_features = _augment_features(
                candidate_embeddings,
                cand_scores,
                cand_ages,
                comment_counts=cand_comment_counts,
                text_lengths=cand_text_lengths,
                hn_quality=cand_quality,
                sim_to_upvoted=cand_sim_to_up,
                sim_to_downvoted=cand_sim_to_down,
                closest_upvoted=cand_closest_up,
                closest_downvoted=cand_closest_down,
            )

            probs = calibrated.predict_proba(cand_features)
            class_order = list(calibrated.classes_)
            idx_up = class_order.index(2)
            idx_neutral = class_order.index(1)
            scores = probs[:, idx_up] + 0.5 * probs[:, idx_neutral]
        except Exception as e:
            logging.error(f"Failed to fit feedback SVM: {e}")

    if scores is None:
        scores = np.full(len(candidates), 0.5, dtype=np.float32)

    ranked = []
    for s, score in zip(candidates, scores):
        ranked.append(RankedStory(story=s, score=float(score), best_match_title=""))

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
        if emb is None:
            selected.append(item)
            if len(selected) >= limit:
                break
            continue

        # Find all remaining unselected items that are similar to this one
        similar_group = [item]
        for other in ranked[idx + 1 :]:
            if other.story.id in discarded:
                continue
            other_emb = embeddings_map.get(other.story.id)
            if other_emb is not None:
                sim = float(np.dot(emb, other_emb))
                if sim > threshold:
                    similar_group.append(other)

        # Select the one with the best engagement if it is significantly higher
        def get_engagement(r: RankedStory) -> int:
            pts = r.story.score or 0
            coms = r.story.comment_count or 0
            return pts + coms

        rep = item
        item_eng = get_engagement(item)
        for other in similar_group:
            other_eng = get_engagement(other)
            if other_eng > item_eng * 2.0 + 30:
                rep = other
                item_eng = other_eng

        selected.append(rep)

        # Discard everything in the group
        for other in similar_group:
            discarded.add(other.story.id)

        # Also discard anything similar to the chosen representative
        rep_emb = embeddings_map.get(rep.story.id)
        if rep_emb is not None:
            for other in ranked[idx:]:
                if other.story.id in discarded:
                    continue
                other_emb = embeddings_map.get(other.story.id)
                if other_emb is not None:
                    sim = float(np.dot(rep_emb, other_emb))
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
) -> None:
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["time_ago"] = time_ago_filter

    pico_css_path = Path("templates/pico.min.css")
    pico_css = (
        pico_css_path.read_text(encoding="utf-8") if pico_css_path.exists() else ""
    )

    # Map user feedback in database for active UI state highlighting
    all_fb = db.get_all_feedback()
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


# Orchestrator
async def run_pipeline(config: Config) -> None:
    db = Database(config.db_path)
    embedder = Embedder(config.onnx_model_dir)

    # Use cached signals for exclusion (no scraping)
    signals = db.get_user_signals()
    signal_ids = (
        signals.get("favorite", set())
        | signals.get("upvote", set())
        | signals.get("hidden", set())
    )

    # Exclude stories that already have user feedback (voted on via dashboard)
    feedback_records = db.get_all_feedback()
    feedback_ids = {f.story_id for f in feedback_records}
    feedback_urls = {f.url for f in feedback_records if f.url}

    logging.info("Fetching candidates...")
    exclude_ids = signal_ids | feedback_ids
    candidates = await fetch_candidates(config, exclude_ids, feedback_urls, db)

    logging.info("Computing embeddings for candidates...")
    cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)
    embeddings_map = {s.id: vec for s, vec in zip(candidates, cand_embeddings)}

    logging.info("Ranking candidates...")
    ranked = rank_stories(
        candidates,
        cand_embeddings,
        db,
        config,
        embedder,
    )

    logging.info("Filtering with MMR...")
    final = mmr_filter(
        ranked,
        embeddings_map,
        threshold=config.model.diversity_threshold,
        limit=config.count,
    )

    logging.info("Generating dashboard...")
    generate_dashboard(
        final,
        Path(config.output),
        config.username,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        config.server_port,
        db,
    )

    logging.info("Pruning old stories from DB...")
    db.prune_stories(max_age_days=config.days * 2)
    db.close()
    logging.info("Done.")
