#!/usr/bin/env -S uv run
"""Retrieve HN archive stories from the ClickHouse MiniLM vector dataset.

This is an explicit experiment, not part of normal dashboard generation. It
uses the same embedding model as the public VectorSearch dataset:
sentence-transformers/all-MiniLM-L6-v2.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx
import numpy as np

from api.client import USER_CACHE_DIR_PATH
from api.fetching import fetch_story
from api.feedback import load_feedback
from api.models import Story

VECTOR_DATASET_REPO = "labofsahil/hackernews-vector-search-dataset"
VECTOR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROFILE_CACHE_VERSION = 1
PROFILE_CACHE_PATH = Path(".cache/vector_archive/upvote_profile_minilm.json")


@dataclass(frozen=True)
class ArchiveHit:
    similarity: float
    id: int
    title: str
    time: str
    score: int
    comments: int
    by: str
    url: str | None
    text: str

    @property
    def hn_url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.id}"


def _as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    out: set[int] = set()
    for item in value:
        if isinstance(item, int):
            out.add(item)
        elif isinstance(item, str) and item.isdigit():
            out.add(int(item))
    return out


def load_cached_user_ids(username: str, cache_dir: Path = USER_CACHE_DIR_PATH) -> dict[str, set[int]]:
    path = cache_dir / f"{username}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No cached user data found at {path}")

    raw = json.loads(path.read_text())
    ids = raw.get("ids", raw)
    if not isinstance(ids, dict):
        raise ValueError(f"Cached user data has unexpected shape: {path}")

    return {
        "pos": _as_int_set(ids.get("pos")),
        "upvoted": _as_int_set(ids.get("upvoted")),
        "favorites": _as_int_set(ids.get("favorites")),
        "hidden": _as_int_set(ids.get("hidden")),
    }


def load_hn_dashboard_feedback_ids() -> tuple[set[int], set[int]]:
    up: set[int] = set()
    down: set[int] = set()
    for record in load_feedback().values():
        if record.source != "hn" or record.id <= 0:
            continue
        if record.action == "up":
            up.add(record.id)
        elif record.action == "down":
            down.add(record.id)
    return up, down


def story_cache_text(story_id: int) -> str | None:
    cache_path = Path(".cache/stories") / f"{story_id}.json"
    if not cache_path.is_file():
        return None
    try:
        raw = json.loads(cache_path.read_text())
    except Exception:
        return None
    story = raw.get("story")
    if not isinstance(story, dict):
        return None
    text = str(story.get("text_content") or story.get("title") or "").strip()
    return text or None


async def load_upvote_stories(
    upvote_ids: Iterable[int],
    *,
    cache_only: bool,
    max_signals: int,
) -> list[Story]:
    selected_ids = list(upvote_ids)
    selected_ids.sort(reverse=True)
    if max_signals > 0:
        selected_ids = selected_ids[:max_signals]

    stories: list[Story] = []
    missing: list[int] = []
    for story_id in selected_ids:
        text = story_cache_text(story_id)
        if text:
            stories.append(
                Story(
                    id=story_id,
                    title=text.splitlines()[0][:160],
                    url=None,
                    score=0,
                    time=0,
                    text_content=text,
                    source="hn",
                )
            )
        else:
            missing.append(story_id)

    if cache_only or not missing:
        return stories

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [fetch_story(client, story_id) for story_id in missing]
        for task in asyncio.as_completed(tasks):
            story = await task
            if story is not None and story.text_content:
                stories.append(story)

    return stories


def embed_profile(texts: Sequence[str], *, model_name: str = VECTOR_MODEL_NAME) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for MiniLM profile embeddings. "
            "Run `uv sync` after updating dependencies."
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise RuntimeError("No embeddings produced for upvote profile")

    profile = arr.mean(axis=0)
    norm = float(np.linalg.norm(profile))
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("Upvote profile vector has zero/invalid norm")
    return (profile / norm).astype(np.float32)


def save_profile_cache(path: Path, username: str, upvote_ids: Sequence[int], profile: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PROFILE_CACHE_VERSION,
        "username": username,
        "model": VECTOR_MODEL_NAME,
        "upvote_ids": list(upvote_ids),
        "profile": profile.astype(float).tolist(),
        "updated_at": time.time(),
    }
    path.write_text(json.dumps(payload))


def load_profile_cache(path: Path, username: str, upvote_ids: Sequence[int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return None
    if raw.get("version") != PROFILE_CACHE_VERSION:
        return None
    if raw.get("username") != username or raw.get("model") != VECTOR_MODEL_NAME:
        return None
    if raw.get("upvote_ids") != list(upvote_ids):
        return None
    profile = np.asarray(raw.get("profile"), dtype=np.float32)
    if profile.ndim != 1 or profile.shape[0] != 384:
        return None
    return profile


def vector_dot(profile: np.ndarray, vector: Any) -> float:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != profile.shape[0]:
        return float("-inf")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not math.isfinite(norm):
        return float("-inf")
    return float(np.dot(profile, arr / norm))


def metadata_url(metadata: Any) -> str | None:
    raw = parse_metadata(metadata)
    if raw is None:
        return None
    url = str(raw.get("url") or "").strip()
    return url or None


def parse_metadata(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, str) or not metadata:
        return None
    try:
        raw = json.loads(metadata)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def metadata_comment_count(metadata: Any) -> int:
    raw = parse_metadata(metadata)
    if raw is None:
        return 0
    try:
        return int(raw.get("descendants") or 0)
    except (TypeError, ValueError):
        return 0


def archive_parquet_paths(
    limit_shards: int | None = None,
    *,
    newest: bool = False,
) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
        from huggingface_hub import hf_hub_url
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to list archive shards") from exc

    files = [
        path
        for path in list_repo_files(VECTOR_DATASET_REPO, repo_type="dataset")
        if path.endswith(".parquet")
    ]
    files.sort()
    if limit_shards is not None and limit_shards > 0:
        files = files[-limit_shards:] if newest else files[:limit_shards]
    return [
        hf_hub_url(VECTOR_DATASET_REPO, filename=path, repo_type="dataset")
        for path in files
    ]


def iter_archive_rows(
    parquet_paths: Sequence[str],
    *,
    profile: np.ndarray | None = None,
    exclude_ids: set[int],
    min_score: int,
    min_comments: int,
    batch_size: int,
    per_shard_limit: int | None = None,
    max_retries: int = 4,
):
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required to scan the VectorSearch Parquet archive. "
            "Run `uv sync` after updating dependencies."
        ) from exc

    conn = duckdb.connect()
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")

    exclude_list = sorted(exclude_ids)
    for path in parquet_paths:
        similarity_expr = "vector"
        params: list[Any] = [path]
        if profile is not None:
            similarity_expr = "list_cosine_similarity(vector, ?::FLOAT[]) AS similarity"
            params = [profile.astype(float).tolist(), path]

        query = """
            SELECT id, title, time, post_score, "by", metadata, text, {similarity_expr}
            FROM read_parquet(?)
            WHERE type = 1
              AND coalesce(dead, 0) = 0
              AND coalesce(deleted, 0) = 0
              AND coalesce(post_score, 0) >= ?
              AND title IS NOT NULL
              AND title != ''
        """.format(similarity_expr=similarity_expr)
        params.append(min_score)
        if exclude_list:
            query += " AND id NOT IN (SELECT * FROM UNNEST(?))"
            params.append(exclude_list)
        if profile is not None:
            query += " ORDER BY similarity DESC"
            if per_shard_limit is not None and per_shard_limit > 0:
                query += " LIMIT ?"
                params.append(per_shard_limit)

        result = None
        for attempt in range(max_retries + 1):
            try:
                result = conn.execute(query, params)
                break
            except Exception as exc:
                if "429" not in str(exc) or attempt >= max_retries:
                    raise
                delay = min(120.0, 10.0 * (2**attempt))
                print(
                    f"[archive-vector] rate limited reading {path}; "
                    f"retrying in {delay:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
        if result is None:
            continue
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            yield from rows


def top_archive_hits(
    rows: Iterable[tuple[Any, ...]],
    *,
    profile: np.ndarray,
    top_k: int,
    min_comments: int,
) -> list[ArchiveHit]:
    heap: list[tuple[float, int, ArchiveHit]] = []
    counter = 0
    for row in rows:
        story_id, title, story_time, score, by, metadata, text, score_or_vector = row
        comments = metadata_comment_count(metadata)
        if comments < min_comments:
            continue
        if isinstance(score_or_vector, int | float):
            similarity = float(score_or_vector)
        else:
            similarity = vector_dot(profile, score_or_vector)
        if not math.isfinite(similarity):
            continue
        hit = ArchiveHit(
            similarity=similarity,
            id=int(story_id),
            title=str(title or ""),
            time=str(story_time or ""),
            score=int(score or 0),
            comments=comments,
            by=str(by or ""),
            url=metadata_url(metadata),
            text=str(text or ""),
        )
        item = (similarity, counter, hit)
        counter += 1
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif similarity > heap[0][0]:
            heapq.heapreplace(heap, item)

    return [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the public HN MiniLM VectorSearch archive using cached HN upvotes."
    )
    parser.add_argument("username", help="HN username with cached signals")
    parser.add_argument("--top-k", type=int, default=10, help="Number of archive hits")
    parser.add_argument(
        "--max-signals",
        type=int,
        default=0,
        help="Use only the N most recent upvoted IDs for the profile (0 = all)",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use locally cached upvote story text; do not fetch missing stories",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=10,
        help="Minimum archive story score/post_score to consider",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=1,
        help="Minimum archive story descendant/comment count to consider",
    )
    parser.add_argument(
        "--refresh-profile",
        action="store_true",
        help="Ignore the cached averaged MiniLM profile vector",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Scan only the first N Parquet shards for a faster smoke test",
    )
    parser.add_argument(
        "--newest-shards",
        action="store_true",
        help="When --limit-shards is set, scan the newest shards instead of the oldest",
    )
    parser.add_argument(
        "--parquet",
        action="append",
        default=None,
        help="Explicit Parquet path/URL to scan; repeatable. Defaults to HF dataset shards.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--json-output", type=Path)
    return parser


async def run(args: argparse.Namespace) -> list[ArchiveHit]:
    user_ids = load_cached_user_ids(args.username)
    upvote_ids = sorted(user_ids["upvoted"], reverse=True)
    if not upvote_ids:
        raise RuntimeError(f"No cached HN upvotes found for @{args.username}")
    if args.max_signals > 0:
        upvote_ids = upvote_ids[: args.max_signals]

    profile = None
    if not args.refresh_profile:
        profile = load_profile_cache(PROFILE_CACHE_PATH, args.username, upvote_ids)

    if profile is None:
        stories = await load_upvote_stories(
            upvote_ids,
            cache_only=args.cache_only,
            max_signals=0,
        )
        texts = [story.text_content for story in stories if story.text_content]
        if not texts:
            raise RuntimeError("No upvote story text available for profile embedding")
        profile = embed_profile(texts)
        save_profile_cache(PROFILE_CACHE_PATH, args.username, upvote_ids, profile)

    dashboard_up, dashboard_down = load_hn_dashboard_feedback_ids()
    exclude_ids = (
        user_ids["upvoted"]
        | user_ids["favorites"]
        | user_ids["hidden"]
        | dashboard_up
        | dashboard_down
    )

    parquet_paths = args.parquet or archive_parquet_paths(
        args.limit_shards,
        newest=args.newest_shards,
    )
    rows = iter_archive_rows(
        parquet_paths,
        profile=profile,
        exclude_ids=exclude_ids,
        min_score=args.min_score,
        min_comments=args.min_comments,
        batch_size=args.batch_size,
        per_shard_limit=max(args.top_k * 5, 50),
    )
    return top_archive_hits(
        rows,
        profile=profile,
        top_k=args.top_k,
        min_comments=args.min_comments,
    )


def print_hits(hits: Sequence[ArchiveHit]) -> None:
    for rank, hit in enumerate(hits, start=1):
        preview = " ".join(hit.text.split())[:220]
        print(f"{rank}. {hit.title}")
        print(
            f"   similarity={hit.similarity:.4f} score={hit.score} "
            f"comments={hit.comments} by={hit.by} time={hit.time}"
        )
        print(f"   HN: {hit.hn_url}")
        if hit.url:
            print(f"   URL: {hit.url}")
        if preview:
            print(f"   {preview}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    hits = asyncio.run(run(args))
    print_hits(hits)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                [
                    {
                        "rank": i,
                        "id": hit.id,
                        "title": hit.title,
                        "time": hit.time,
                        "score": hit.score,
                        "comments": hit.comments,
                        "by": hit.by,
                        "url": hit.url,
                        "hn_url": hit.hn_url,
                        "similarity": hit.similarity,
                        "preview": " ".join(hit.text.split())[:500],
                    }
                    for i, hit in enumerate(hits, start=1)
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
