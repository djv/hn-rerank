#!/usr/bin/env -S uv run
"""Benchmark embedding models on cached HN story text.

Usage:
    uv run python scripts/benchmark_embedding_models.py
    uv run python scripts/benchmark_embedding_models.py --max-stories 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# Ensure the project root is on sys.path so 'api' module is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    raise SystemExit("sentence_transformers is required. Run: uv sync")

import numpy as np  # noqa: E402
from sklearn.metrics import pairwise_distances  # noqa: E402

from api.model_metadata import (  # noqa: E402
    BGE_BASE_OFFICIAL_SPEC,
    BGE_SMALL_CLS_QUERY_ALL_SPEC,
    E5_BASE_V2_SPEC,
    GTE_BASE_V15_SPEC,
    EmbeddingModelSpec,
)
from api.models import Story  # noqa: E402

BASELINE_SPEC = BGE_SMALL_CLS_QUERY_ALL_SPEC

MODEL_TIMEOUT_SECONDS = 600


@contextmanager
def model_timeout(seconds: int) -> Iterator[None]:
    def handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"model operation timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


CANDIDATE_SPECS: list[tuple[str, EmbeddingModelSpec]] = [
    ("bge-small-en-v1.5", BASELINE_SPEC),
    (
        "all-MiniLM-L6-v2",
        EmbeddingModelSpec(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            pooling="mean",
            normalize=True,
            text_mode="plain",
        ),
    ),
    (
        "multi-qa-MiniLM-L6-cos-v1",
        EmbeddingModelSpec(
            model_id="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            pooling="mean",
            normalize=True,
            text_mode="plain",
        ),
    ),
    (
        "gte-small",
        EmbeddingModelSpec(
            model_id="thenlper/gte-small",
            pooling="mean",
            normalize=True,
            text_mode="plain",
            trust_remote_code=True,
        ),
    ),
    (
        "e5-small-v2",
        EmbeddingModelSpec(
            model_id="intfloat/e5-small-v2",
            pooling="mean",
            normalize=True,
            text_mode="query_prefix_all",
            query_prefix="query: ",
        ),
    ),
    ("bge-base-en-v1.5", BGE_BASE_OFFICIAL_SPEC),
    (
        "bge-large-en-v1.5",
        EmbeddingModelSpec(
            model_id="BAAI/bge-large-en-v1.5",
            pooling="cls",
            normalize=True,
            text_mode="query_prefix_all",
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
    ),
    ("gte-base-en-v1.5", GTE_BASE_V15_SPEC),
    (
        "gte-large-en-v1.5",
        EmbeddingModelSpec(
            model_id="Alibaba-NLP/gte-large-en-v1.5",
            pooling="cls",
            normalize=True,
            text_mode="plain",
            trust_remote_code=True,
        ),
    ),
    (
        "nomic-embed-text-v1.5",
        EmbeddingModelSpec(
            model_id="nomic-ai/nomic-embed-text-v1.5",
            pooling="mean",
            normalize=True,
            text_mode="plain",
            trust_remote_code=True,
        ),
    ),
    ("e5-base-v2", E5_BASE_V2_SPEC),
]


def load_cached_stories(max_stories: int) -> list[str]:
    cache_dir = Path(".cache/stories")
    if not cache_dir.is_dir():
        raise SystemExit(f"cache directory not found: {cache_dir}")

    paths = sorted(cache_dir.glob("*.json"))
    logger.info("Found %d cached story files", len(paths))

    texts: list[str] = []
    for p in paths:
        try:
            raw = json.loads(p.read_bytes())
            story_dict = raw.get("story") or raw
            story = Story.from_dict(story_dict)
            text = (story.text_content or "").strip()
            if len(text) >= 30:
                texts.append(text)
        except Exception:
            continue

    if not texts:
        raise SystemExit("No cached stories with text_content >= 30 chars found.")

    logger.info("Loaded %d stories with text_content >= 30 chars", len(texts))
    rng = random.Random(42)
    sample = rng.sample(texts, min(max_stories, len(texts)))
    logger.info("Sampling %d stories for benchmark", len(sample))
    return sample


def prepare_texts(texts: list[str], spec: EmbeddingModelSpec) -> list[str]:
    return [spec.prepare_text(t, is_query=False) for t in texts]


def compute_spread(embeddings: np.ndarray) -> float:
    dist = pairwise_distances(embeddings, metric="cosine")
    tri = dist[np.triu_indices_from(dist, k=1)]
    p10, p90 = float(np.percentile(tri, 10)), float(np.percentile(tri, 90))
    return p90 - p10


def compute_neighbor_overlap(
    baseline: np.ndarray, candidate: np.ndarray, k: int = 5
) -> float:
    n = baseline.shape[0]
    bl_dist = pairwise_distances(baseline, metric="cosine")
    cand_dist = pairwise_distances(candidate, metric="cosine")

    total_overlap = 0.0
    for i in range(n):
        bl_neighbors = set(np.argsort(bl_dist[i])[1 : k + 1])
        cand_neighbors = set(np.argsort(cand_dist[i])[1 : k + 1])
        total_overlap += len(bl_neighbors & cand_neighbors) / k
    return total_overlap / n


def error_result(label: str, exc: BaseException) -> dict[str, Any]:
    return {
        "label": label,
        "dim": 0,
        "stories_per_sec": 0.0,
        "spread": 0.0,
        "nn_overlap": 0.0,
        "error": str(exc),
    }


def main(max_stories: int = 100) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    texts = load_cached_stories(max_stories)
    n = len(texts)

    results: list[dict[str, Any]] = []
    baseline_embeddings: np.ndarray | None = None

    for label, spec in CANDIDATE_SPECS:
        logger.info("Loading %s (%s)...", label, spec.model_id)
        try:
            with model_timeout(MODEL_TIMEOUT_SECONDS):
                model = SentenceTransformer(
                    spec.model_id,
                    trust_remote_code=spec.trust_remote_code,
                )
        except TimeoutError as e:
            logger.error("Timed out loading %s: %s", label, e)
            results.append(error_result(label, e))
            continue
        except Exception as e:
            logger.error("Failed to load %s: %s", label, e)
            results.append(error_result(label, e))
            continue

        prepared = prepare_texts(texts, spec)
        dim = model.get_sentence_embedding_dimension()

        logger.info("Encoding %d texts with %s...", n, label)
        t0 = time.perf_counter()
        try:
            with model_timeout(MODEL_TIMEOUT_SECONDS):
                embeddings: np.ndarray = model.encode(
                    prepared,
                    show_progress_bar=True,
                    normalize_embeddings=spec.normalize,
                )
        except TimeoutError as e:
            logger.error("Timed out encoding with %s: %s", label, e)
            results.append(error_result(label, e))
            continue
        elapsed = time.perf_counter() - t0
        sps = n / elapsed if elapsed > 0 else 0.0

        spread = compute_spread(embeddings)

        nn_overlap = 0.0
        if baseline_embeddings is not None:
            nn_overlap = compute_neighbor_overlap(baseline_embeddings, embeddings)
        else:
            baseline_embeddings = embeddings

        results.append(
            {
                "label": label,
                "dim": dim,
                "stories_per_sec": sps,
                "spread": spread,
                "nn_overlap": nn_overlap,
            }
        )

        logger.info(
            "%s: %d dim, %.0f stories/s, spread=%.3f, nn_overlap=%.3f",
            label,
            dim,
            sps,
            spread,
            nn_overlap,
        )

    print()
    print(
        f"{'Model':<24} {'Dim':>5} {'stories/s':>10} {'spread':>8} {'nn_overlap':>11}"
    )
    print("-" * 62)

    best_models: list[tuple[float, str]] = []
    for r in results:
        if r.get("error"):
            print(f"{r['label']:<24} {'FAILED':>5}  {r['error']}")
            continue
        print(
            f"{r['label']:<24} {r['dim']:>5} {r['stories_per_sec']:>10.0f} {r['spread']:>8.3f} {r['nn_overlap']:>11.3f}"
        )
        score = r["spread"] * np.log1p(r["stories_per_sec"])
        best_models.append((score, r["label"]))

    print()
    best_models.sort(reverse=True)
    logger.info("Top pick: %s (spread × log(speed) tradeoff)", best_models[0][1])
    logger.info("Runner-up: %s", best_models[1][1] if len(best_models) > 1 else "n/a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models on cached HN stories"
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=100,
        help="Maximum number of cached stories to use (default: 100)",
    )
    args = parser.parse_args()
    main(max_stories=args.max_stories)
