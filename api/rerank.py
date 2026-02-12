from __future__ import annotations
import asyncio
import os
import hashlib
import json
import logging
import math
import threading
import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

def _ensure_joblib_settings() -> None:
    # Disable joblib multiprocessing in this environment to avoid SemLock
    # permission warnings; joblib falls back to serial either way.
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    tmp = os.environ.get("JOBLIB_TEMP_FOLDER") or os.environ.get("LOKY_TEMP_FOLDER")
    if not tmp:
        tmp = str(Path(__file__).resolve().parents[1] / ".cache" / "joblib")
        os.environ["JOBLIB_TEMP_FOLDER"] = tmp
        os.environ["LOKY_TEMP_FOLDER"] = tmp
    Path(tmp).mkdir(parents=True, exist_ok=True)


_ensure_joblib_settings()

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase  # noqa: E402
from aiolimiter import AsyncLimiter  # noqa: E402
from tenacity import (  # noqa: E402
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from api.cache_utils import atomic_write_json  # noqa: E402
from api.constants import (  # noqa: E402
    ADAPTIVE_HN_THRESHOLD_OLD,
    ADAPTIVE_HN_THRESHOLD_YOUNG,
    ADAPTIVE_HN_WEIGHT_MAX,
    ADAPTIVE_HN_WEIGHT_MIN,
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_CACHE_MAX_FILES,
    EMBEDDING_MIN_CLIP,
    EMBEDDING_MODEL_VERSION,
    CLUSTER_EMBEDDING_CACHE_DIR,
    CLUSTER_EMBEDDING_MODEL_DIR,
    CLUSTER_EMBEDDING_MODEL_VERSION,
    CLUSTER_ALGORITHM,
    CLUSTER_AGGLOMERATIVE_LINKAGE,
    CLUSTER_AGGLOMERATIVE_METRIC,
    CLUSTER_REFINE_ITERS,
    CLUSTER_SPECTRAL_NEIGHBORS,
    FRESHNESS_HALF_LIFE_HOURS,
    FRESHNESS_MAX_BOOST,
    HN_SCORE_NORMALIZATION_CAP,
    KNN_NEIGHBORS,
    LLM_CLUSTER_MAX_TOKENS,
    LLM_CLUSTER_NAME_MAX_WORDS,
    LLM_CLUSTER_TITLE_SAMPLES,
    LLM_CLUSTER_TITLE_MAX_CHARS,
    LLM_CLUSTER_MAX_RETRIES,
    LLM_CLUSTER_MAX_ROUNDS,
    LLM_CLUSTER_MAX_TOTAL_SECONDS,
    LLM_429_COOLDOWN_BASE,
    LLM_429_COOLDOWN_MAX,
    LLM_CLUSTER_NAME_MODEL_FALLBACK,
    LLM_CLUSTER_NAME_MODEL_PRIMARY,
    LLM_CLUSTER_NAME_PROMPT_VERSION,
    CLUSTER_OUTLIER_SIMILARITY_THRESHOLD,
    LLM_HTTP_CONNECT_TIMEOUT,
    LLM_HTTP_READ_TIMEOUT,
    LLM_HTTP_WRITE_TIMEOUT,
    LLM_HTTP_POOL_TIMEOUT,
    LLM_HTTP_USER_AGENT,
    LLM_MIN_REQUEST_INTERVAL,
    LLM_TEMPERATURE,
    LLM_TLDR_BATCH_SIZE,
    LLM_TLDR_MAX_TOKENS,
    LLM_TLDR_MODEL,
    MAX_CLUSTERS,
    MAX_CLUSTER_FRACTION,
    MAX_CLUSTER_SIZE,
    MIN_CLUSTERS,
    MIN_SAMPLES_PER_CLUSTER,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_DIVERSITY_LAMBDA_CLASSIFIER,
    RANKING_HN_WEIGHT,
    RANKING_MAX_RESULTS,
    RANKING_NEGATIVE_WEIGHT,
    RATE_LIMIT_ERROR_BACKOFF_BASE,
    RATE_LIMIT_ERROR_BACKOFF_MAX,
    KNN_SIGMOID_K,
    KNN_MAXSIM_WEIGHT,
    CLASSIFIER_K_FEAT,
    CLASSIFIER_NEG_SAMPLE_WEIGHT,
    SIMILARITY_MIN,
    TEXT_CONTENT_MAX_TOKENS,
)
from api.llm_utils import build_payload  # noqa: E402
from api.models import RankResult, Story, StoryDict, StoryForTldr  # noqa: E402

logger = logging.getLogger(__name__)


class GroqQuotaError(RuntimeError):
    """Raised when Groq returns a non-retryable quota error (e.g., TPD)."""


class GroqRetryableError(RuntimeError):
    """Raised for retryable Groq errors."""

    def __init__(
        self, message: str, cooldown: float | None = None, is_rate_limit: bool = False
    ) -> None:
        super().__init__(message)
        self.cooldown = cooldown
        self.is_rate_limit = is_rate_limit


type ClusterItem = tuple[StoryDict, float]


# Global singleton for the model
_model: ONNXEmbeddingModel | None = None
_cluster_model: ONNXEmbeddingModel | None = None
_model_init_lock = threading.Lock()
_cluster_model_init_lock = threading.Lock()

CACHE_DIR: Path = Path(EMBEDDING_CACHE_DIR)
CLUSTER_CACHE_DIR: Path = Path(CLUSTER_EMBEDDING_CACHE_DIR)
for _cache_dir in (CACHE_DIR, CLUSTER_CACHE_DIR):
    _cache_dir.mkdir(parents=True, exist_ok=True)


def _evict_old_embedding_cache_files(
    cache_dir: Path, max_files: int = EMBEDDING_CACHE_MAX_FILES
) -> None:
    # Racy by nature under multi-process sweeps; files can disappear between
    # glob/stat/unlink. Keep eviction best-effort and ignore transient misses.
    candidates = list(cache_dir.glob("*.npz")) + list(cache_dir.glob("*.npy"))
    existing: list[tuple[float, Path]] = []
    for p in candidates:
        try:
            existing.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue

    if len(existing) <= max_files:
        return

    existing.sort(key=lambda t: t[0])
    for _, f in existing[: len(existing) - max_files]:
        try:
            f.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            pass


class ONNXEmbeddingModel:
    def __init__(self, model_dir: str = "onnx_model") -> None:
        self.model_dir: str = model_dir
        if not Path(f"{model_dir}/model.onnx").exists():
            raise FileNotFoundError(
                f"Model not found in {model_dir}. Please run setup_model.py."
            )

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_dir
        )

        providers = ["CPUExecutionProvider"]

        self.session: ort.InferenceSession = ort.InferenceSession(
            f"{model_dir}/model.onnx", providers=providers
        )
        self.model_id: str = "bge-base-en-v1.5"
        self._lock = threading.Lock()

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> NDArray[np.float32]:
        all_embeddings: list[NDArray[np.float32]] = []
        total_items: int = len(texts)

        for i in range(0, total_items, batch_size):
            if progress_callback:
                progress_callback(i, total_items)

            batch: list[str] = texts[i : i + batch_size]
            with self._lock:
                inputs: BatchEncoding = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )
                # Copy arrays while tokenizer internals are exclusively held.
                attention_mask = cast(
                    NDArray[np.int64], inputs["attention_mask"].astype(np.int64, copy=True)
                )

                input_names: list[str] = [node.name for node in self.session.get_inputs()]
                ort_inputs: dict[str, NDArray[np.int64]] = {
                    k: v.astype(np.int64) for k, v in inputs.items() if k in input_names
                }

                outputs = self.session.run(None, ort_inputs)
                last_hidden_state: NDArray[np.float32] = cast(
                    NDArray[np.float32], outputs[0]
                )

            # Mean Pooling
            mask_expanded: NDArray[np.float64] = np.expand_dims(
                attention_mask, -1
            ).astype(float)
            sum_embeddings: NDArray[np.float32] = np.sum(
                last_hidden_state * mask_expanded, axis=1
            )
            sum_mask: NDArray[np.float64] = np.clip(
                mask_expanded.sum(axis=1), a_min=EMBEDDING_MIN_CLIP, a_max=None
            )
            batch_embeddings: NDArray[np.float32] = sum_embeddings / sum_mask

            if normalize_embeddings:
                norm: NDArray[np.float64] = np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True
                )
                batch_embeddings = batch_embeddings / np.clip(
                    norm, a_min=EMBEDDING_MIN_CLIP, a_max=None
                )

            all_embeddings.append(batch_embeddings)

        if progress_callback:
            progress_callback(total_items, total_items)

        return (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.array([], dtype=np.float32)
        )


def init_model() -> ONNXEmbeddingModel:
    global _model
    if _model is None:
        with _model_init_lock:
            if _model is None:
                _model = ONNXEmbeddingModel()
    assert _model is not None
    return _model


def get_model() -> ONNXEmbeddingModel:
    return init_model()

def init_cluster_model() -> ONNXEmbeddingModel:
    global _cluster_model
    if _cluster_model is None:
        with _cluster_model_init_lock:
            if _cluster_model is None:
                _cluster_model = ONNXEmbeddingModel(model_dir=CLUSTER_EMBEDDING_MODEL_DIR)
    assert _cluster_model is not None
    return _cluster_model


def _get_embeddings_with_model(
    texts: list[str],
    model: ONNXEmbeddingModel,
    cache_dir: Path,
    cache_version: str,
    is_query: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    if not texts:
        return np.array([], dtype=np.float32)

    # BGE-style prefix for queries only
    prefix: str = (
        "Represent this sentence for searching relevant passages: " if is_query else ""
    )
    truncated_count = 0
    processed_texts: list[str] = []

    def _truncate_to_token_budget(text: str) -> tuple[str, bool]:
        encoded = model.tokenizer(
            text,
            truncation=True,
            max_length=TEXT_CONTENT_MAX_TOKENS,
            return_tensors=None,
        )
        input_ids = encoded.get("input_ids", [])
        if isinstance(input_ids, list) and input_ids:
            token_ids = input_ids[0] if isinstance(input_ids[0], list) else input_ids
        else:
            return text, False

        truncated = len(token_ids) >= TEXT_CONTENT_MAX_TOKENS
        if not truncated:
            return text, False
        return model.tokenizer.decode(token_ids, skip_special_tokens=True), True
    for t in texts:
        text = f"{prefix}{t}"
        truncated_text, was_truncated = _truncate_to_token_budget(text)
        if was_truncated:
            truncated_count += 1
        processed_texts.append(truncated_text)
    if truncated_count:
        logger.info(
            "Embedding input truncated for %d/%d texts (max %d tokens).",
            truncated_count,
            len(texts),
            TEXT_CONTENT_MAX_TOKENS,
        )

    vectors: list[NDArray[np.float32] | None] = []
    to_compute_indices: list[int] = []

    expected_dim: int = 768  # bge-base-en-v1.5

    for idx, text in enumerate(processed_texts):
        # Include model version in hash to invalidate cache on model change
        h: str = hashlib.sha256(
            f"{cache_version}:{model.model_id}:{text}".encode()
        ).hexdigest()
        cache_path_npz: Path = cache_dir / f"{h}.npz"
        cache_path_npy: Path = cache_dir / f"{h}.npy"

        vec: NDArray[np.float32] | None = None
        if cache_path_npz.exists():
            try:
                data = np.load(cache_path_npz)
                vec = data["embedding"]
            except Exception as e:
                logger.debug(f"Failed to load embedding cache {cache_path_npz}: {e}")
        elif cache_path_npy.exists():
            try:
                vec = np.load(cache_path_npy)
            except Exception as e:
                logger.debug(f"Failed to load embedding cache {cache_path_npy}: {e}")

        if vec is not None and vec.shape == (expected_dim,):
            vectors.append(vec)
        else:
            vectors.append(None)
            to_compute_indices.append(idx)

    # Report progress for cached items immediately
    cached_count: int = len(texts) - len(to_compute_indices)
    if progress_callback and cached_count > 0:
        progress_callback(cached_count, len(texts))

    if to_compute_indices:
        # Wrap the original callback to add the offset from cached items
        def wrapped_callback(curr: int, total: int) -> None:
            if progress_callback:
                progress_callback(cached_count + curr, len(texts))

        computed: NDArray[np.float32] = model.encode(
            [processed_texts[i] for i in to_compute_indices],
            progress_callback=wrapped_callback,
        )
        for i, original_idx in enumerate(to_compute_indices):
            vec_res: NDArray[np.float32] = computed[i]
            vectors[original_idx] = vec_res
            h_res: str = hashlib.sha256(
                f"{cache_version}:{model.model_id}:{processed_texts[original_idx]}".encode()
            ).hexdigest()
            # Use compressed format for cache efficiency
            np.savez_compressed(cache_dir / f"{h_res}.npz", embedding=vec_res)
        _evict_old_embedding_cache_files(cache_dir)

    if not vectors or all(v is None for v in vectors):
        return np.zeros((0, expected_dim), dtype=np.float32)

    return np.stack([v for v in vectors if v is not None]).astype(np.float32)


def get_embeddings(
    texts: list[str],
    is_query: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    model: ONNXEmbeddingModel = get_model()
    return _get_embeddings_with_model(
        texts,
        model=model,
        cache_dir=CACHE_DIR,
        cache_version=EMBEDDING_MODEL_VERSION,
        is_query=is_query,
        progress_callback=progress_callback,
    )


def get_cluster_embeddings(
    texts: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    model: ONNXEmbeddingModel = init_cluster_model()
    return _get_embeddings_with_model(
        texts,
        model=model,
        cache_dir=CLUSTER_CACHE_DIR,
        cache_version=CLUSTER_EMBEDDING_MODEL_VERSION,
        is_query=False,
        progress_callback=progress_callback,
    )


def cluster_interests(
    embeddings: NDArray[np.float32],
    weights: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """
    Cluster user interest embeddings into K centroids.
    Returns centroids array of shape (n_clusters, embedding_dim).
    """
    centroids, _ = cluster_interests_with_labels(embeddings, weights)
    return centroids


def cluster_interests_with_labels(
    embeddings: NDArray[np.float32],
    weights: NDArray[np.float32] | None = None,
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Cluster user interest embeddings using a cosine-aware algorithm.
    Uses fixed k (default 30); LLM naming handles semantic coherence.
    Returns (centroids, labels) where:
      - centroids: shape (n_clusters, embedding_dim)
      - labels: shape (n_samples,) cluster assignment per sample
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

    n_samples = len(embeddings)
    if n_samples == 0:
        return embeddings, np.array([], dtype=np.int32)

    if n_samples < MIN_SAMPLES_PER_CLUSTER * 2:
        # Not enough for meaningful clustering
        labels = np.zeros(n_samples, dtype=np.int32)
        if weights is not None:
            centroid = np.average(embeddings, axis=0, weights=weights).reshape(1, -1)
        else:
            centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        return centroid.astype(np.float32), labels

    # Normalize embeddings for cosine-like behavior
    normalized = _normalize_embeddings(embeddings)

    # Use fixed number of clusters as requested (max MAX_CLUSTERS)
    # n_clusters is capped by n_samples / MIN_SAMPLES_PER_CLUSTER and MAX_CLUSTERS
    effective_n_clusters = min(n_clusters, MAX_CLUSTERS, n_samples // MIN_SAMPLES_PER_CLUSTER)
    effective_n_clusters = max(effective_n_clusters, MIN_CLUSTERS)

    if CLUSTER_ALGORITHM == "spectral":
        neighbors = min(CLUSTER_SPECTRAL_NEIGHBORS, max(2, n_samples - 1))
        clustering = SpectralClustering(
            n_clusters=effective_n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
            random_state=0,
        )
        labels = clustering.fit_predict(normalized).astype(np.int32)
    elif CLUSTER_ALGORITHM == "agglomerative":
        clustering = AgglomerativeClustering(
            n_clusters=effective_n_clusters,
            metric=CLUSTER_AGGLOMERATIVE_METRIC,
            linkage=CLUSTER_AGGLOMERATIVE_LINKAGE,
        )
        labels = clustering.fit_predict(normalized).astype(np.int32)
    else:
        kmeans = KMeans(
            n_clusters=effective_n_clusters,
            n_init=10,
            random_state=0,
        )
        labels = kmeans.fit_predict(normalized).astype(np.int32)

    labels = _refine_cluster_assignments(
        normalized, labels, CLUSTER_REFINE_ITERS
    )
    labels = _merge_small_clusters(embeddings, labels, MIN_SAMPLES_PER_CLUSTER)

    max_size = max(
        MIN_SAMPLES_PER_CLUSTER,
        min(
            MAX_CLUSTER_SIZE,
            int(math.ceil(n_samples * MAX_CLUSTER_FRACTION)),
        ),
    )
    max_clusters = min(MAX_CLUSTERS, n_samples // MIN_SAMPLES_PER_CLUSTER)
    labels = _split_large_clusters(
        embeddings,
        labels,
        min_size=MIN_SAMPLES_PER_CLUSTER,
        max_size=max_size,
        max_clusters=max_clusters,
    )

    centroids = _centroids_from_labels(embeddings, labels, weights)
    centroids, labels, _ = split_outlier_clusters(
        embeddings, labels, CLUSTER_OUTLIER_SIMILARITY_THRESHOLD, weights=weights
    )
    return centroids, labels


def _centroids_from_labels(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int32],
    weights: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Compute centroids for each cluster label."""
    unique_labels = sorted(set(labels))
    centroids = []
    for lbl in unique_labels:
        mask = labels == lbl
        if weights is not None:
            cluster_weights = weights[mask]
            centroid = np.average(embeddings[mask], axis=0, weights=cluster_weights)
        else:
            centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids, dtype=np.float32)


def _normalize_embeddings(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return embeddings / norms


def split_outlier_clusters(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int32],
    min_similarity: float,
    weights: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], int]:
    """Split low-similarity items into singleton clusters.

    Returns (new_centroids, new_labels, outlier_count).
    """
    if len(labels) == 0:
        return embeddings, labels, 0
    if min_similarity <= SIMILARITY_MIN:
        centroids = _centroids_from_labels(embeddings, labels)
        return centroids, labels, 0

    centroids = _centroids_from_labels(embeddings, labels, weights)
    if len(centroids) == 0:
        return centroids, labels, 0

    emb_norm = _normalize_embeddings(embeddings)
    cent_norm = _normalize_embeddings(centroids)
    sims = np.sum(emb_norm * cent_norm[labels], axis=1)
    outlier_idx = np.where(sims < min_similarity)[0]
    if len(outlier_idx) == 0:
        return centroids, labels, 0

    new_labels = labels.copy()
    next_label = int(np.max(labels)) + 1
    for idx in outlier_idx:
        new_labels[idx] = next_label
        next_label += 1

    remap = {old: i for i, old in enumerate(sorted(set(new_labels)))}
    new_labels = np.array([remap[int(lbl)] for lbl in new_labels], dtype=np.int32)
    new_centroids = _centroids_from_labels(embeddings, new_labels, weights)
    logger.info(
        "Split %d low-similarity items into singleton clusters (min %.2f).",
        len(outlier_idx),
        min_similarity,
    )
    return new_centroids, new_labels, len(outlier_idx)


def _refine_cluster_assignments(
    embeddings_norm: NDArray[np.float32],
    labels: NDArray[np.int32],
    iters: int,
) -> NDArray[np.int32]:
    """Iteratively reassign samples to the nearest centroid (cosine space)."""
    if iters <= 0 or len(labels) == 0:
        remap = {old: i for i, old in enumerate(sorted(set(labels)))}
        return np.array([remap[int(lbl)] for lbl in labels], dtype=np.int32)

    current = labels.astype(np.int32, copy=True)
    for _ in range(iters):
        unique_labels = sorted(set(current))
        if len(unique_labels) <= 1:
            break

        centroids = []
        for lbl in unique_labels:
            mask = current == lbl
            centroid = embeddings_norm[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                centroid = centroid / norm
            centroids.append(centroid)
        centroids = np.array(centroids, dtype=np.float32)

        sims = embeddings_norm @ centroids.T
        best_idx = np.argmax(sims, axis=1)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        curr_idx = np.array([label_to_idx[int(lbl)] for lbl in current], dtype=np.int32)

        best_sim = sims[np.arange(len(current)), best_idx]
        curr_sim = sims[np.arange(len(current)), curr_idx]
        move = best_sim > curr_sim
        if not np.any(move):
            break

        next_idx = np.where(move, best_idx, curr_idx)
        next_labels = np.array([unique_labels[int(i)] for i in next_idx], dtype=np.int32)
        if np.array_equal(next_labels, current):
            break
        current = next_labels

    remap = {old: i for i, old in enumerate(sorted(set(current)))}
    return np.array([remap[int(lbl)] for lbl in current], dtype=np.int32)


def _merge_small_clusters(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int32],
    min_size: int,
) -> NDArray[np.int32]:
    """Merge clusters smaller than min_size into the nearest larger cluster."""
    if min_size <= 1:
        return labels

    unique_labels, counts = np.unique(labels, return_counts=True)
    small = {int(lbl) for lbl, cnt in zip(unique_labels, counts) if cnt < min_size}
    if not small:
        return labels

    large = [int(lbl) for lbl, cnt in zip(unique_labels, counts) if cnt >= min_size]
    if not large:
        return np.zeros(len(labels), dtype=np.int32)

    centroids = _centroids_from_labels(embeddings, labels)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    centroids_norm = centroids / norms

    new_labels = labels.copy()
    for lbl in small:
        idxs = np.where(labels == lbl)[0]
        if len(idxs) == 0:
            continue

        cluster_emb = embeddings[idxs]
        c_norm = np.linalg.norm(cluster_emb, axis=1, keepdims=True)
        c_norm = np.maximum(c_norm, 1e-9)
        cluster_emb_norm = cluster_emb / c_norm

        large_indices = [label for label in large]
        sims = cluster_emb_norm @ centroids_norm[large_indices].T
        target = large_indices[int(np.argmax(np.mean(sims, axis=0)))]
        new_labels[idxs] = target

    remap = {old: i for i, old in enumerate(sorted(set(new_labels)))}
    return np.array([remap[int(lbl)] for lbl in new_labels], dtype=np.int32)


def _reassign_small_subclusters(
    embeddings_norm: NDArray[np.float32],
    labels: NDArray[np.int32],
    min_size: int,
) -> NDArray[np.int32]:
    """Reassign samples from tiny subclusters to nearest large centroid."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    small = {int(lbl) for lbl, cnt in zip(unique_labels, counts) if cnt < min_size}
    if not small:
        return labels

    large = [int(lbl) for lbl, cnt in zip(unique_labels, counts) if cnt >= min_size]
    if not large:
        return labels

    centroids = []
    for lbl in unique_labels:
        mask = labels == lbl
        centroids.append(embeddings_norm[mask].mean(axis=0))
    centroids = np.array(centroids, dtype=np.float32)

    # Normalize centroids to ensure cosine similarity.
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    centroids_norm = centroids / norms

    new_labels = labels.copy()
    large_indices = [int(np.where(unique_labels == lbl)[0][0]) for lbl in large]

    for idx, lbl in enumerate(labels):
        if int(lbl) in small:
            sims = embeddings_norm[idx] @ centroids_norm[large_indices].T
            target_idx = large_indices[int(np.argmax(sims))]
            new_labels[idx] = int(unique_labels[target_idx])

    remap = {old: i for i, old in enumerate(sorted(set(new_labels)))}
    return np.array([remap[int(lbl)] for lbl in new_labels], dtype=np.int32)


def _split_large_clusters(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int32],
    min_size: int,
    max_size: int,
    max_clusters: int,
) -> NDArray[np.int32]:
    """Split clusters larger than max_size, without exceeding max_clusters."""
    from sklearn.cluster import KMeans
    if max_clusters <= 0 or max_size <= 0:
        return labels

    while True:
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_count = len(unique_labels)
        if cluster_count >= max_clusters:
            return labels

        oversized = {
            int(lbl): int(cnt)
            for lbl, cnt in zip(unique_labels, counts)
            if int(cnt) > max_size
        }
        if not oversized:
            return labels

        # Split the largest cluster first.
        target_label = max(oversized, key=lambda label: oversized[label])
        target_size = oversized[target_label]
        needed = max(2, int(math.ceil(target_size / max_size)))
        available = max_clusters - cluster_count
        split_k = min(available + 1, needed)
        if split_k <= 1:
            return labels

        idxs = np.where(labels == target_label)[0]
        sub_emb = embeddings[idxs]

        norms = np.linalg.norm(sub_emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        sub_norm = sub_emb / norms

        kmeans = KMeans(n_clusters=split_k, n_init=10, random_state=0)
        sub_labels = kmeans.fit_predict(sub_norm).astype(np.int32)
        sub_labels = _reassign_small_subclusters(sub_norm, sub_labels, min_size)

        sub_unique = sorted(set(sub_labels))
        new_label_start = int(labels.max()) + 1
        mapping: dict[int, int] = {}
        for i, sub_lbl in enumerate(sub_unique):
            mapping[int(sub_lbl)] = (
                target_label if i == 0 else new_label_start + (i - 1)
            )

        labels[idxs] = np.array(
            [mapping[int(sub_lbl)] for sub_lbl in sub_labels], dtype=np.int32
        )

        # If we failed to increase the number of clusters, stop to avoid loops.
        if len(set(labels)) <= cluster_count:
            return labels


_LLM_LIMITER: AsyncLimiter = AsyncLimiter(
    1, max(1.0, float(LLM_MIN_REQUEST_INTERVAL))
)


def _parse_retry_after(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    now = datetime.now(dt.tzinfo)
    delta = (dt - now).total_seconds()
    return max(0.0, delta)


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if not header:
        return None
    return _parse_retry_after(header)


def _extract_groq_error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text.strip()
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str):
                return msg.strip()
    return resp.text.strip()


_RATE_LIMIT_WAIT = wait_random_exponential(
    min=RATE_LIMIT_ERROR_BACKOFF_BASE, max=RATE_LIMIT_ERROR_BACKOFF_MAX
)
_RATE_LIMIT_429_WAIT = wait_random_exponential(
    min=LLM_429_COOLDOWN_BASE, max=LLM_429_COOLDOWN_MAX
)


def _retry_wait(retry_state: RetryCallState) -> float:
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    if isinstance(exc, GroqRetryableError):
        if exc.cooldown is not None:
            return exc.cooldown
        if exc.is_rate_limit:
            return _RATE_LIMIT_429_WAIT(retry_state)
    return _RATE_LIMIT_WAIT(retry_state)


CLUSTER_NAME_CACHE_PATH = Path(".cache/cluster_names.json")


def _cluster_name_cache_key(story_ids: Sequence[str], model: str) -> str:
    key_src = ",".join(story_ids)
    key_src += f"|model={model}|prompt={LLM_CLUSTER_NAME_PROMPT_VERSION}"
    return hashlib.sha256(key_src.encode()).hexdigest()


_INVALID_CLUSTER_NAMES = {
    "",
    "misc",
    "not provided",
    "n/a",
    "none",
    "unknown",
}
_CLUSTER_NAME_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "ask",
    "askhn",
    "by",
    "for",
    "from",
    "hn",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "show",
    "tell",
    "the",
    "to",
    "vs",
    "with",
    "what",
    "when",
    "where",
    "why",
}


def _is_valid_cluster_name(name: str) -> bool:
    normalized = name.strip()
    if not normalized:
        return False
    if normalized.lower() in _INVALID_CLUSTER_NAMES:
        return False
    words = normalized.split()
    cleaned = [
        re.sub(r"[^A-Za-z0-9]+", "", word)
        for word in words
        if re.sub(r"[^A-Za-z0-9]+", "", word)
    ]
    if len(cleaned) < 2:
        return False
    return 2 <= len(cleaned) <= LLM_CLUSTER_NAME_MAX_WORDS


def _title_to_label(title: str) -> str:
    # Clean HN prefixes
    for prefix in ["Show HN:", "Ask HN:", "Tell HN:"]:
        if title.startswith(prefix):
            title = title[len(prefix) :].strip()
    tokens = re.findall(r"[A-Za-z0-9+#]+", title)
    filtered = [t for t in tokens if t.lower() not in _CLUSTER_NAME_STOPWORDS]
    base = filtered if len(filtered) >= 2 else tokens

    if len(base) >= 3:
        words = base[:3]
    elif len(base) == 2:
        words = base
    elif len(base) == 1:
        words = tokens[:2] if len(tokens) >= 2 else [base[0], "Topic"]
    else:
        words = ["Misc", "Topic"]

    return " ".join(words[:3])


def _cluster_token_frequencies(items: Sequence[ClusterItem]) -> list[str]:
    from collections import Counter

    counter: Counter[str] = Counter()
    sorted_items = sorted(items, key=lambda x: -x[1])[:20]
    for s, _ in sorted_items:
        title = str(s.get("title", "")).strip()
        if not title:
            continue
        tokens = re.findall(r"[A-Za-z0-9+#]+", title)
        for t in tokens:
            tl = t.lower()
            if tl in _CLUSTER_NAME_STOPWORDS or len(tl) < 2:
                continue
            counter[tl] += 1

    if not counter:
        return []

    def _sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
        token, freq = item
        return (freq, len(token), token)

    top = [t for t, _ in sorted(counter.items(), key=_sort_key, reverse=True)[:3]]
    return top


def _format_label_tokens(tokens: list[str]) -> str:
    if not tokens:
        return "Misc Topic"
    upper = {"ai", "llm", "api", "gpu", "sql", "k8s", "wasm", "ios", "mac"}
    formatted: list[str] = []
    for t in tokens:
        if t in upper:
            formatted.append(t.upper())
        elif t.isupper():
            formatted.append(t)
        else:
            formatted.append(t.capitalize())
    return " ".join(formatted[:3])


def _fallback_cluster_name(items: Sequence[ClusterItem]) -> str:
    tokens = _cluster_token_frequencies(items)
    if tokens:
        return _format_label_tokens(tokens)
    sorted_items = sorted(items, key=lambda x: -x[1])
    for s, _ in sorted_items:
        title = str(s.get("title", "")).strip()
        if title:
            return _title_to_label(title)
    return "Misc Topic"


def _label_coverage(label: str, items: Sequence[ClusterItem]) -> float:
    if not label or not items:
        return 0.0
    label_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9+#]+", label)]
    if not label_tokens:
        return 0.0
    match_count = 0
    total = 0
    for s, _ in items:
        title = str(s.get("title", "")).strip().lower()
        if not title:
            continue
        total += 1
        if any(tok in title for tok in label_tokens):
            match_count += 1
    return match_count / max(total, 1)


def _cluster_token_set(items: Sequence[ClusterItem]) -> set[str]:
    tokens: set[str] = set()
    for s, _ in items:
        title = str(s.get("title", "")).strip()
        if not title:
            continue
        for t in re.findall(r"[A-Za-z0-9+#]+", title):
            if len(t) < 2:
                continue
            tokens.add(t.lower())
    return tokens


def _name_has_overlap(name: str, items: Sequence[ClusterItem]) -> bool:
    name_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9+#]+", name)]
    if not name_tokens:
        return False
    title_tokens = _cluster_token_set(items)
    if not title_tokens:
        return False
    return any(tok in title_tokens for tok in name_tokens)


def _normalize_cluster_name(
    name: str,
    items: Sequence[ClusterItem],
) -> str:
    cleaned = " ".join(name.strip().split()[:LLM_CLUSTER_NAME_MAX_WORDS])
    cleaned = cleaned.rstrip(" ,&/").rstrip()
    if cleaned.endswith(" and") or cleaned.endswith(" or"):
        cleaned = cleaned.rsplit(" ", 1)[0]
    if _is_valid_cluster_name(cleaned):
        return cleaned
    return _fallback_cluster_name(items)


def _finalize_cluster_name(raw_name: str) -> str | None:
    """Normalize cluster name formatting without altering semantics."""
    cleaned = raw_name.strip()
    if not cleaned or "\n" in cleaned:
        return None
    if any(ch in cleaned for ch in ("{", "}", "[", "]")):
        return None
    cleaned = " ".join(cleaned.split()[:LLM_CLUSTER_NAME_MAX_WORDS])
    cleaned = cleaned.rstrip(" ,&/").rstrip()
    if cleaned.endswith(" and") or cleaned.endswith(" or"):
        cleaned = cleaned.rsplit(" ", 1)[0].rstrip()
    return cleaned or None


def _load_cluster_name_cache() -> dict[str, str]:
    if CLUSTER_NAME_CACHE_PATH.exists():
        try:
            cache = json.loads(CLUSTER_NAME_CACHE_PATH.read_text())
            if isinstance(cache, dict):
                return {
                    str(key): str(val)
                    for key, val in cache.items()
                    if isinstance(val, str) and val.strip()
                }
        except Exception as e:
            logger.warning(f"Failed to load cluster name cache: {e}")
    return {}


def _save_cluster_name_cache(cache: dict[str, str]) -> None:
    CLUSTER_NAME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CLUSTER_NAME_CACHE_PATH, cache)


def _safe_json_loads(text: str) -> dict[str, object]:
    """Safely load JSON, handling potential markdown blocks."""
    if not text:
        return {}

    def _strip_code_fence(src: str) -> str:
        cleaned = src.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _extract_json_substring(src: str) -> str | None:
        first_obj = src.find("{")
        first_arr = src.find("[")
        if first_obj == -1 and first_arr == -1:
            return None
        if first_arr == -1 or (first_obj != -1 and first_obj < first_arr):
            open_ch, close_ch, start = "{", "}", first_obj
        else:
            open_ch, close_ch, start = "[", "]", first_arr

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(src)):
            ch = src[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return src[start : idx + 1]
        return None

    clean_text = _strip_code_fence(text)
    candidates = [clean_text]
    extracted = _extract_json_substring(clean_text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    import ast

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode failed, trying fallback: {e}")
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
        except Exception:
            continue
    return {}


async def _generate_with_retry(
    model: str = LLM_TLDR_MODEL,
    contents: object | None = None,
    config: dict[str, object] | None = None,
    max_retries: int = 3,
) -> str | None:
    """Call Groq API with exponential backoff retry logic using httpx."""
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, skipping LLM call")
        return None

    payload = build_payload(model=model, contents=contents, config=config)

    timeout = httpx.Timeout(
        connect=LLM_HTTP_CONNECT_TIMEOUT,
        read=LLM_HTTP_READ_TIMEOUT,
        write=LLM_HTTP_WRITE_TIMEOUT,
        pool=LLM_HTTP_POOL_TIMEOUT,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries),
                retry=retry_if_exception_type(GroqRetryableError),
                wait=_retry_wait,
                reraise=True,
            ):
                with attempt:
                    try:
                        async with _LLM_LIMITER:
                            resp = await client.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers={
                                    "Authorization": f"Bearer {api_key}",
                                    "Content-Type": "application/json",
                                    "User-Agent": LLM_HTTP_USER_AGENT,
                                },
                                json=payload,
                            )
                    except httpx.HTTPError as e:
                        raise GroqRetryableError(str(e)) from e

                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]

                    if resp.status_code == 429:
                        error_msg = _extract_groq_error_message(resp)
                        msg_lower = error_msg.lower()
                        if "tokens per day" in msg_lower or " tpd" in msg_lower:
                            raise GroqQuotaError(error_msg)
                        if "requests per day" in msg_lower or " rpd" in msg_lower:
                            raise GroqQuotaError(error_msg)
                        retry_after = _retry_after_seconds(resp)
                        raise GroqRetryableError(
                            error_msg,
                            cooldown=retry_after,
                            is_rate_limit=True,
                        )

                    if resp.status_code in {408, 500, 502, 503, 504}:
                        raise GroqRetryableError(
                            f"Groq API error {resp.status_code}"
                        )

                    error_msg = _extract_groq_error_message(resp)
                    logger.error(
                        "Groq API error %d: %s", resp.status_code, error_msg or resp.text
                    )
                    return None
        except GroqQuotaError:
            raise
        except GroqRetryableError as e:
            logger.error("Groq API call failed after %d retries: %s", max_retries, e)
            return None

    return None


async def generate_batch_cluster_names(
    clusters: dict[int, list[ClusterItem]],
    progress_callback: Callable[[int, int], None] | None = None,
    debug_path: Path | None = None,
) -> dict[int, str]:
    """Generate names for clusters one at a time to improve reliability."""
    if not clusters:
        return {}

    cache = _load_cluster_name_cache()
    results: dict[int, str] = {}
    to_generate: dict[int, list[ClusterItem]] = {}
    primary_model = LLM_CLUSTER_NAME_MODEL_PRIMARY
    fallback_model = LLM_CLUSTER_NAME_MODEL_FALLBACK
    active_model = primary_model
    fallback_triggered = False

    for cid, items in clusters.items():
        # Generate cache key based on sorted story IDs
        story_ids = sorted([str(s.get("id", s.get("objectID", ""))) for s, _ in items])
        primary_key = _cluster_name_cache_key(story_ids, primary_model)
        cached_val = cache.get(primary_key)
        if not cached_val and fallback_model:
            fallback_key = _cluster_name_cache_key(story_ids, fallback_model)
            cached_val = cache.get(fallback_key)
        if cached_val and cached_val.strip():
            results[cid] = cached_val
        else:
            to_generate[cid] = items

    if not to_generate:
        if progress_callback:
            progress_callback(len(clusters), len(clusters))
        return {cid: results.get(cid, "") for cid in clusters}

    cid_list = list(to_generate.keys())
    total_batches = len(cid_list)
    max_rounds = LLM_CLUSTER_MAX_ROUNDS
    request_count = 0
    start_time = time.time()
    deadline = start_time + LLM_CLUSTER_MAX_TOTAL_SECONDS
    missing_overall: set[int] = set()

    debug_records: list[dict[str, object]] = []

    payloads: dict[int, list[str]] = {}
    for cid, items in to_generate.items():
        # Use top titles only to keep prompts compact and reduce latency.
        sorted_items = sorted(items, key=lambda x: -x[1])[:LLM_CLUSTER_TITLE_SAMPLES]
        cluster_titles: list[str] = []
        for s, _ in sorted_items:
            title = str(s.get("title", "")).strip()
            if not title:
                continue
            title = " ".join(title.split())
            if len(title) > LLM_CLUSTER_TITLE_MAX_CHARS:
                title = title[:LLM_CLUSTER_TITLE_MAX_CHARS].rstrip()
            cluster_titles.append(title)

        payloads[cid] = cluster_titles

    def _flush_debug() -> None:
        if debug_path is None:
            return
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug_records, indent=2))

    def _remap_batch_results(
        batch_results: dict[str, object],
        pending_ids: Sequence[int],
        context: str,
        batch_index: int,
        attempt: int,
        model: str,
    ) -> dict[str, object]:
        if not batch_results or not pending_ids:
            return batch_results
        pending_set = set(pending_ids)
        numeric_keys: list[str] = []
        for k in batch_results:
            if isinstance(k, str) and k.strip().lstrip("-").isdigit():
                numeric_keys.append(k)
        if not numeric_keys:
            return batch_results
        if any(int(k) in pending_set for k in numeric_keys):
            return batch_results
        # No overlap: model ignored requested IDs. Remap by order.
        ordered_items = sorted(
            ((int(k), v) for k, v in batch_results.items() if k in numeric_keys),
            key=lambda x: x[0],
        )
        if len(pending_ids) == 1:
            return {str(pending_ids[0]): ordered_items[0][1]}
        if len(ordered_items) != len(pending_ids):
            return batch_results
        remapped: dict[str, object] = {}
        for cid, (_, value) in zip(sorted(pending_ids), ordered_items):
            remapped[str(cid)] = value
        if debug_path is not None:
            debug_records.append(
                {
                    "event": "remap_keys",
                    "context": context,
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "model": model,
                    "pending_ids": sorted(pending_ids),
                    "original_keys": numeric_keys,
                }
            )
            _flush_debug()
        return remapped

    for batch_index, cid in enumerate(cid_list, start=1):
        batch_cids = [cid]
        pending: set[int] = {cid}

        for attempt in range(1, max_rounds + 1):
            if not pending:
                break
            if time.time() > deadline:
                debug_records.append(
                    {
                        "event": "timeout",
                        "batch_index": batch_index,
                        "pending": sorted(pending),
                        "elapsed": time.time() - start_time,
                    }
                )
                _flush_debug()
                raise RuntimeError(
                    "Groq cluster naming timed out after "
                    f"{LLM_CLUSTER_MAX_TOTAL_SECONDS:.0f}s. "
                    "Try rerunning with fewer clusters/candidates or wait for quota reset."
                )

            batch_prompts: list[str] = []
            for cid in sorted(pending):
                titles = payloads.get(cid, [])
                title_lines = "\n".join(f"- {t}" for t in titles) if titles else "- (no titles)"
                batch_prompts.append(f"Titles:\n{title_lines}")

            full_prompt = f"""
Provide a concise label between two and six words that describes the cluster stories.

Rules:
- Prefer the most specific recurring technical topic in the titles.
- Use 1-2 topics only if the cluster clearly spans two themes.
- Avoid umbrella terms unless the titles are genuinely diverse.
- Use key terms from the titles where possible.
- Use normal spacing and Title Case (e.g., "Large Language Models", not "LargeLanguageModels" or "Large_Language_Models").
- Return ONLY the label text. Do not return JSON, bullets, or extra commentary.

Group:
{chr(10).join(batch_prompts)}
"""

            try:
                pending_before = sorted(pending)
                logger.info(
                    "Groq cluster naming batch %d/%d attempt %d starting (%d pending).",
                    batch_index,
                    total_batches,
                    attempt,
                    len(pending),
                )
                request_count += 1
                t0 = time.time()
                attempt_model = active_model
                text = await _generate_with_retry(
                    model=attempt_model,
                    contents=full_prompt,
                    config={
                        "temperature": LLM_TEMPERATURE,
                        "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                    },
                    max_retries=LLM_CLUSTER_MAX_RETRIES,
                )
                duration = time.time() - t0

                returned = 0
                batch_results: dict[str, object] | None = None
                if (
                    text is None
                    and attempt_model == primary_model
                    and fallback_model
                    and not fallback_triggered
                ):
                    fallback_triggered = True
                    active_model = fallback_model
                    if debug_path is not None:
                        debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "empty_response",
                                "batch_index": batch_index,
                                "attempt": attempt,
                                "from_model": primary_model,
                                "to_model": fallback_model,
                            }
                        )
                        _flush_debug()
                if text:
                    logger.debug("Groq cluster naming raw response: %s", text)
                    if len(pending) == 1:
                        only_cid = next(iter(pending))
                        final_name = _finalize_cluster_name(text)
                        if final_name:
                            results[only_cid] = final_name
                            items = to_generate[only_cid]
                            story_ids = sorted(
                                [
                                    str(s.get("id", s.get("objectID", "")))
                                    for s, _ in items
                                ]
                            )
                            cache_key = _cluster_name_cache_key(
                                story_ids, attempt_model
                            )
                            cache[cache_key] = final_name
                            returned += 1
                pending = {cid for cid in pending if cid not in results}
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "batch",
                            "batch_index": batch_index,
                            "attempt": attempt,
                            "model": attempt_model,
                            "batch_cids": batch_cids,
                            "pending_before": pending_before,
                            "payloads": {
                                str(cid): payloads.get(cid, {})
                                for cid in sorted(pending_before)
                            },
                            "prompt": full_prompt,
                            "response": text,
                            "parsed": batch_results if text else None,
                            "returned": returned,
                            "pending_after": sorted(pending),
                        }
                    )
                    _flush_debug()
                logger.info(
                    "Groq cluster naming batch %d/%d attempt %d: %d/%d names in %.2fs (pending %d).",
                    batch_index,
                    total_batches,
                    attempt,
                    returned,
                    len(batch_cids),
                    duration,
                    len(pending),
                )

                if pending and attempt < max_rounds:
                    await asyncio.sleep(min(2**attempt, 8))
            except GroqQuotaError as e:
                if (
                    not fallback_triggered
                    and fallback_model
                    and active_model == primary_model
                ):
                    fallback_triggered = True
                    active_model = fallback_model
                    if debug_path is not None:
                        debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "quota_error",
                                "batch_index": batch_index,
                                "attempt": attempt,
                                "batch_cids": batch_cids,
                                "pending_before": sorted(pending),
                                "error": str(e),
                                "from_model": primary_model,
                                "to_model": fallback_model,
                            }
                        )
                        _flush_debug()
                    if attempt < max_rounds:
                        await asyncio.sleep(min(2**attempt, 8))
                        continue
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "batch",
                            "batch_index": batch_index,
                            "attempt": attempt,
                            "batch_cids": batch_cids,
                            "pending_before": sorted(pending),
                            "error": str(e),
                            "model": active_model,
                        }
                    )
                    _flush_debug()
                raise RuntimeError(f"Groq quota exceeded: {e}") from e
            except Exception as e:
                logger.warning(f"Cluster naming batch failed: {e}")
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "batch",
                            "batch_index": batch_index,
                            "attempt": attempt,
                            "batch_cids": batch_cids,
                            "pending_before": sorted(pending),
                            "error": repr(e),
                        }
                    )
                    _flush_debug()
                if attempt < max_rounds:
                    await asyncio.sleep(min(2**attempt, 8))

        if pending:
            missing_overall.update(pending)

        if progress_callback:
            progress_callback(len(results), len(clusters))

    _save_cluster_name_cache(cache)

    if missing_overall:
        # Final rescue pass: retry missing clusters in small batches.
        missing_list = sorted(missing_overall)
        for rescue_index, cid in enumerate(missing_list, start=1):
            batch_missing = [cid]
            if time.time() > deadline:
                debug_records.append(
                    {
                        "event": "timeout",
                        "batch_index": "rescue",
                        "pending": batch_missing,
                        "elapsed": time.time() - start_time,
                    }
                )
                _flush_debug()
                raise RuntimeError(
                    "Groq cluster naming timed out after "
                    f"{LLM_CLUSTER_MAX_TOTAL_SECONDS:.0f}s. "
                    "Try rerunning with fewer clusters/candidates or wait for quota reset."
                )

            rescue_prompts = []
            for cid in batch_missing:
                titles = payloads.get(cid, [])
                title_lines = "\n".join(f"- {t}" for t in titles) if titles else "- (no titles)"
                rescue_prompts.append(f"Titles:\n{title_lines}")

            rescue_prompt = f"""
Provide a concise label between two and six words that describes the cluster stories.

Rules:
- Prefer specific technical topics from the titles.
- Generic labels are allowed only if the titles are genuinely diverse.
- Use normal spacing and Title Case (e.g., "Large Language Models", not "LargeLanguageModels" or "Large_Language_Models").
- Return ONLY the label text. Do not return JSON, bullets, or extra commentary.

Group:
{chr(10).join(rescue_prompts)}
"""

            try:
                rescue_model = active_model
                request_count += 1
                t0 = time.time()
                try:
                    text = await _generate_with_retry(
                        model=rescue_model,
                        contents=rescue_prompt,
                        config={
                            "temperature": LLM_TEMPERATURE,
                            "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                        },
                        max_retries=1,
                    )
                except GroqQuotaError as e:
                    if (
                        not fallback_triggered
                        and fallback_model
                        and active_model == primary_model
                    ):
                        fallback_triggered = True
                        active_model = fallback_model
                        rescue_model = active_model
                        if debug_path is not None:
                            debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "quota_error_rescue",
                                "batch_index": rescue_index,
                                "batch_cids": batch_missing,
                                "error": str(e),
                                    "from_model": primary_model,
                                    "to_model": fallback_model,
                                }
                            )
                            _flush_debug()
                        request_count += 1
                        t0 = time.time()
                        text = await _generate_with_retry(
                            model=rescue_model,
                            contents=rescue_prompt,
                            config={
                                "temperature": LLM_TEMPERATURE,
                                "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                            },
                            max_retries=1,
                        )
                    else:
                        raise
                duration = time.time() - t0
                if (
                    text is None
                    and rescue_model == primary_model
                    and fallback_model
                    and not fallback_triggered
                ):
                    fallback_triggered = True
                    active_model = fallback_model
                    rescue_model = active_model
                    if debug_path is not None:
                        debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "empty_response_rescue",
                                "batch_index": rescue_index,
                                "batch_cids": batch_missing,
                                "from_model": primary_model,
                                "to_model": fallback_model,
                            }
                        )
                        _flush_debug()
                    request_count += 1
                    t0 = time.time()
                    text = await _generate_with_retry(
                        model=rescue_model,
                        contents=rescue_prompt,
                        config={
                            "temperature": LLM_TEMPERATURE,
                            "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                        },
                        max_retries=1,
                    )
                    duration = time.time() - t0

                returned = 0
                batch_results: dict[str, object] | None = None
                if text and len(batch_missing) == 1:
                    cid = batch_missing[0]
                    final_name = _finalize_cluster_name(text)
                    if final_name:
                        results[cid] = final_name
                        items = to_generate[cid]
                        story_ids = sorted(
                            [
                                str(s.get("id", s.get("objectID", "")))
                                for s, _ in items
                            ]
                        )
                        cache_key = _cluster_name_cache_key(
                            story_ids, rescue_model
                        )
                        cache[cache_key] = final_name
                        returned += 1

                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "rescue_batch",
                            "batch_index": rescue_index,
                            "model": rescue_model,
                            "batch_cids": batch_missing,
                            "payloads": {
                                str(cid): payloads.get(cid, {})
                                for cid in batch_missing
                            },
                            "prompt": rescue_prompt,
                            "response": text,
                            "parsed": batch_results if text else None,
                            "returned": returned,
                            "duration": duration,
                        }
                    )
                    _flush_debug()
            except GroqQuotaError as e:
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "rescue_batch",
                            "batch_index": rescue_index,
                            "batch_cids": batch_missing,
                            "error": str(e),
                        }
                    )
                    _flush_debug()
                raise RuntimeError(f"Groq quota exceeded: {e}") from e
            except Exception as e:
                logger.warning(f"Cluster naming rescue batch failed: {e}")

    _flush_debug()

    missing = [cid for cid in clusters if not results.get(cid)]
    if missing:
        elapsed = time.time() - start_time
        raise RuntimeError(
            "Groq cluster naming failed for "
            f"{len(missing)} clusters after {request_count} requests "
            f"({elapsed:.1f}s). Missing IDs: {sorted(missing)}. "
            "Likely rate-limited or incomplete JSON. "
            "Try rerunning, lowering --clusters, or waiting for quota reset."
        )

    return {cid: results.get(cid, "") for cid in clusters}


TLDR_CACHE_PATH = Path(".cache/tldrs.json")


def _load_tldr_cache() -> dict[str, str]:
    """Load TL;DR cache from disk."""
    if TLDR_CACHE_PATH.exists():
        try:
            return json.loads(TLDR_CACHE_PATH.read_text())
        except Exception as e:
            logger.warning(f"Failed to load TLDR cache: {e}")
    return {}


def _save_tldr_cache(cache: dict[str, str]) -> None:
    """Save TL;DR cache to disk."""
    TLDR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(TLDR_CACHE_PATH, cache)


async def generate_batch_tldrs(
    stories: Sequence[StoryForTldr],
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, str]:
    """Generate TL;DRs for multiple stories in batches to save API quota."""
    if not stories:
        return {}

    cache = _load_tldr_cache()
    results: dict[int, str] = {}
    to_generate: list[StoryForTldr] = []

    for s in stories:
        sid = int(s["id"])
        cached_val = cache.get(str(sid))
        if cached_val and len(cached_val.strip()) > 0:
            results[sid] = cached_val
        else:
            to_generate.append(s)

    if not to_generate:
        if progress_callback:
            progress_callback(len(stories), len(stories))
        return {
            int(s["id"]): results.get(int(s["id"]), cache.get(str(s["id"]), ""))
            for s in stories
        }

    total_to_gen = len(to_generate)
    completed_initial = len(stories) - total_to_gen

    for i in range(0, total_to_gen, LLM_TLDR_BATCH_SIZE):
        original_batch = to_generate[i : i + LLM_TLDR_BATCH_SIZE]
        pending_stories: dict[int, StoryForTldr] = {
            int(s["id"]): s for s in original_batch
        }

        # Try up to 2 times to get all summaries for this batch
        for attempt in range(2):
            if not pending_stories:
                break

            current_batch = list(pending_stories.values())
            stories_formatted = []
            for s in current_batch:
                title = s.get("title", "Untitled")
                comments = s.get("comments", [])
                context = f"ID: {s['id']}\nTitle: {title}"
                if comments:
                    context += "\nComments:\n" + "\n".join(
                        f"- {c[:300]}" for c in comments[:4]
                    )
                stories_formatted.append(context)

            batch_context = "\n\n---\n\n".join(stories_formatted)

            prompt = f"""
Summarize the discussion and technical insights in 2-3 sentences.
CRITICAL: Do NOT repeat the title. The user sees it. Do NOT say "This story is about...".

Focus on:
- Technical implementation details, trade-offs, or benchmarks mentioned in comments
- Significant debates, criticisms, or comparisons to other tools

BAD: "PostgreSQL 17 Released. It features..." (Repeats title)
GOOD: "Praised for incremental backups and optimized vacuuming, though some users warn about update conflicts on legacy systems."

Return JSON with story IDs as keys: {{ "12345": "Summary here." }}

Stories:
{batch_context}

JSON:"""

            try:
                text = await _generate_with_retry(
                    model=LLM_TLDR_MODEL,
                    contents=prompt,
                    config={
                        "temperature": LLM_TEMPERATURE,
                        "max_output_tokens": LLM_TLDR_MAX_TOKENS,
                        "response_mime_type": "application/json",
                    },
                )

                if text:
                    batch_results = _safe_json_loads(text)
                    for sid_str, tldr in batch_results.items():
                        try:
                            sid = int(sid_str)
                            if not isinstance(tldr, str):
                                logger.debug(
                                    f"TLDR value for {sid_str} was not a string"
                                )
                                continue
                            tldr_clean = tldr.strip().strip('"').strip("'")
                            if sid in pending_stories:
                                results[sid] = tldr_clean
                                cache[str(sid)] = tldr_clean
                                del pending_stories[sid]
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Failed to parse TLDR for {sid_str}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"TLDR batch generation failed (attempt {attempt+1}): {e}")

        if progress_callback:
            progress_callback(completed_initial + i + len(original_batch), len(stories))

    _save_tldr_cache(cache)
    return {
        int(s["id"]): results.get(int(s["id"]), cache.get(str(s["id"]), ""))
        for s in stories
    }


def rank_stories(
    stories: list[Story],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None = None,
    positive_weights: NDArray[np.float32] | None = None,
    hn_weight: float = RANKING_HN_WEIGHT,
    neg_weight: float = RANKING_NEGATIVE_WEIGHT,
    diversity_lambda: float = RANKING_DIVERSITY_LAMBDA,
    use_classifier: bool = False,
    use_contrastive: bool = False,
    knn_k: int = KNN_NEIGHBORS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[RankResult]:
    """
    Rank candidate stories by relevance to user interests.

    Uses multi-interest clustering and MMR diversity reranking.
    Optionally uses a Logistic Regression classifier if sufficient data exists.
    """
    if not stories:
        return []

    cand_texts: list[str] = [s.text_content for s in stories]
    cand_emb: NDArray[np.float32] = get_embeddings(
        cand_texts, progress_callback=progress_callback
    )

    semantic_scores: NDArray[np.float32]
    max_sim_scores: NDArray[np.float32]
    best_fav_indices: NDArray[np.int64]
    raw_knn_scores: NDArray[np.float32]  # For display (before sigmoid)

    # Check if we can/should use classifier
    classifier_success = False
    if (
        use_classifier
        and positive_embeddings is not None
        and negative_embeddings is not None
        and len(positive_embeddings) >= 5
        and len(negative_embeddings) >= 5
    ):
        try:
            # Prepare training data
            X_pos = positive_embeddings
            X_neg = negative_embeddings
            y_pos = np.ones(len(X_pos))
            y_neg = np.zeros(len(X_neg))

            X_train = np.vstack([X_pos, X_neg])
            y_train = np.concatenate([y_pos, y_neg])

            # Calculate cluster-balanced weights for positives
            # Use inverse log weighting (1/log(1+N)) for even softer dampening.
            # This respects that large clusters represent confirmed, deep interest
            # while still preventing them from totally drowning out small niches.
            centroids, labels = cluster_interests_with_labels(X_pos, positive_weights)
            unique_labels, counts = np.unique(labels, return_counts=True)
            # Use log1p for soft dampening: 1/log(1+count)
            weight_map = {
                lbl: 1.0 / np.log1p(count) for lbl, count in zip(unique_labels, counts)
            }

            # Base weights from clustering
            pos_sample_weights = np.array([weight_map[label] for label in labels])

            # Normalize so sum(weights) == n_samples (maintains scale with negatives)
            norm_factor = len(X_pos) / np.sum(pos_sample_weights)
            pos_sample_weights *= norm_factor

            # Apply recency weights if available (multiplicative)
            if positive_weights is not None:
                pos_sample_weights *= positive_weights

            # Scale negative-class weight to tune classifier decision boundary.
            neg_sample_weights = (
                np.ones(len(X_neg), dtype=np.float32) * CLASSIFIER_NEG_SAMPLE_WEIGHT
            )

            sample_weights = np.concatenate([pos_sample_weights, neg_sample_weights])

            # --- Feature augmentation: append derived similarity features ---
            # Compute 3 features consistently for train and candidates to avoid
            # train/test distribution mismatch (no hardcoded zeros).
            k_feat = min(len(X_pos), CLASSIFIER_K_FEAT)
            k_neg_feat = min(len(X_neg), CLASSIFIER_K_FEAT)

            def _derived_features(
                embs: NDArray[np.float32],
                pos_ref: NDArray[np.float32],
                neg_ref: NDArray[np.float32],
                centroid_ref: NDArray[np.float32],
                k_pos: int,
                k_neg: int,
                exclude_self_pos: bool = False,
                exclude_self_neg: bool = False,
            ) -> NDArray[np.float32]:
                """Compute [max_sim_centroid, knn_pos_median, knn_neg_median]."""
                # 1. Max cosine to any cluster centroid
                sim_c = cosine_similarity(embs, centroid_ref)
                f_centroid = np.max(sim_c, axis=1)

                # 2. Median top-k cosine to positive embeddings
                sim_p = cosine_similarity(embs, pos_ref)
                if exclude_self_pos:
                    np.fill_diagonal(sim_p, -1.0)  # exclude self-match
                if k_pos > 0:
                    f_knn_pos = np.median(
                        np.partition(sim_p, -k_pos, axis=1)[:, -k_pos:], axis=1
                    )
                else:
                    f_knn_pos = np.zeros(len(embs))

                # 3. Median top-k cosine to negative embeddings
                sim_n = cosine_similarity(embs, neg_ref)
                if exclude_self_neg:
                    np.fill_diagonal(sim_n, -1.0)
                if k_neg > 0:
                    f_knn_neg = np.median(
                        np.partition(sim_n, -k_neg, axis=1)[:, -k_neg:], axis=1
                    )
                else:
                    f_knn_neg = np.zeros(len(embs))

                return np.column_stack([f_centroid, f_knn_pos, f_knn_neg]).astype(np.float32)

            pos_derived = _derived_features(
                X_pos, X_pos, X_neg, centroids, k_feat, k_neg_feat,
                exclude_self_pos=True,
            )
            neg_derived = _derived_features(
                X_neg, X_pos, X_neg, centroids, k_feat, k_neg_feat,
                exclude_self_neg=True,
            )
            train_derived = np.vstack([pos_derived, neg_derived])
            X_train = np.hstack([X_train, train_derived])

            cand_derived = _derived_features(
                cand_emb, X_pos, X_neg, centroids, k_feat, k_neg_feat,
            )
            cand_features = np.hstack([cand_emb, cand_derived])

            clf = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=3,
                class_weight="balanced",
                l1_ratios=(0.0,),
                solver="liblinear",
                scoring="f1",
                use_legacy_attributes=False,
            )
            clf.fit(X_train, y_train, sample_weight=sample_weights)

            # Predict probabilities (class 1 = positive interest)
            probs = clf.predict_proba(cand_features)[:, 1]
            semantic_scores = probs.astype(np.float32)

            # Post-classifier negative penalty: classifier doesn't see hidden stories
            # after training, so apply explicit penalty for similarity to hidden items
            if negative_embeddings is not None and len(negative_embeddings) > 0:
                sim_neg = cosine_similarity(negative_embeddings, cand_emb)
                k_neg = min(len(negative_embeddings), knn_k)
                knn_neg = np.median(np.partition(sim_neg, -k_neg, axis=0)[-k_neg:, :], axis=0)
                semantic_scores = semantic_scores - neg_weight * knn_neg

            # We still need max_sim_scores for the UI "Similar to..."
            sim_pos_ui = cosine_similarity(positive_embeddings, cand_emb)
            max_sim_scores = np.max(sim_pos_ui, axis=0)
            best_fav_indices = np.argmax(sim_pos_ui, axis=0)

            # Compute k-NN scores for display
            k = min(len(positive_embeddings), knn_k)
            if k > 0:
                top_k_sims = np.partition(sim_pos_ui, -k, axis=0)[-k:, :]
                raw_knn_scores = np.median(top_k_sims, axis=0).astype(np.float32)
            else:
                raw_knn_scores = np.zeros(len(stories), dtype=np.float32)

            classifier_success = True
            # Classifier probabilities are sharp (often >0.9 for dominant clusters).
            # We increase diversity penalty to ensure we skip to the next cluster.
            diversity_lambda = max(
                diversity_lambda, RANKING_DIVERSITY_LAMBDA_CLASSIFIER
            )
        except Exception as e:
            # Fallback to heuristic on error
            logger.warning(f"Classifier training failed, using heuristic: {e}")

    if not classifier_success:
        if positive_embeddings is None or len(positive_embeddings) == 0:
            # If no positive signals, use HN scores primarily
            semantic_scores = np.zeros(len(stories), dtype=np.float32)
            max_sim_scores = np.zeros(len(stories), dtype=np.float32)
            best_fav_indices = np.full(len(stories), -1, dtype=np.int64)
            raw_knn_scores = np.zeros(len(stories), dtype=np.float32)
        else:
            # 1. Candidate similarity to positive history (for display + reasons)
            # Calculate full similarity matrix: (n_history, n_candidates)
            sim_matrix: NDArray[np.float32] = cosine_similarity(
                positive_embeddings, cand_emb
            )

            # Find top K neighbors for each candidate (along axis 0)
            k = min(len(positive_embeddings), knn_k)
            if k > 0:
                # np.partition moves the top K elements to the end
                # We take the last k rows (which are the largest)
                top_k_sims = np.partition(sim_matrix, -k, axis=0)[-k:, :]
                # Use median for outlier robustness (single bad match won't skew score)
                knn_scores = np.median(top_k_sims, axis=0)
            else:
                knn_scores = np.zeros(len(stories), dtype=np.float32)

            # Store raw k-NN scores for display before sigmoid
            raw_knn_scores = knn_scores.astype(np.float32)

            # For display score and best_fav_index, use the single best match
            # This preserves interpretable "match to specific story" display
            max_sim_scores = np.max(sim_matrix, axis=0)
            best_fav_indices = np.argmax(sim_matrix, axis=0)

            # 2. Cluster-max semantic scoring
            cluster_centroids = cluster_interests(positive_embeddings, positive_weights)
            if len(cluster_centroids) > 0:
                cluster_sim = cosine_similarity(cluster_centroids, cand_emb)
                cluster_max_scores = np.max(cluster_sim, axis=0).astype(np.float32)
            else:
                cluster_max_scores = np.zeros(len(stories), dtype=np.float32)

            # Blend cluster-max and k-NN scores, then z-score normalize
            # This replaces the fixed sigmoid that saturated all k-NN scores to ~1.0
            blended = (
                KNN_MAXSIM_WEIGHT * cluster_max_scores
                + (1.0 - KNN_MAXSIM_WEIGHT) * knn_scores
            )
            mu = blended.mean()
            sigma = blended.std() + 1e-9
            z = (blended - mu) / sigma
            # Gentle sigmoid (k=KNN_SIGMOID_K) maps z-scores to [0,1]
            semantic_scores = 1.0 / (1.0 + np.exp(-KNN_SIGMOID_K * z))
            semantic_scores = semantic_scores.astype(np.float32)

        # 3. Negative Signal (Penalty) - Only applies in heuristic mode
        # Use k-NN for negatives: penalize consistent "not interested" patterns
        if negative_embeddings is not None and len(negative_embeddings) > 0:
            sim_neg_matrix: NDArray[np.float32] = cosine_similarity(
                negative_embeddings, cand_emb
            )
            # k-NN for negative signals (same k as positive)
            k_neg = min(len(negative_embeddings), knn_k)
            if k_neg > 0:
                top_k_neg = np.partition(sim_neg_matrix, -k_neg, axis=0)[-k_neg:, :]
                knn_neg: NDArray[np.float32] = np.median(top_k_neg, axis=0)
            else:
                knn_neg = np.zeros(len(stories), dtype=np.float32)
            # Apply penalty: contrastive (only when neg > pos) or always
            if use_contrastive:
                should_penalize = knn_neg > raw_knn_scores
                semantic_scores -= neg_weight * knn_neg * should_penalize
            else:
                semantic_scores -= neg_weight * knn_neg

    # 4. HN Gravity Score (Log-scaled)
    # We use a log scale so that high-point stories punch through without dominating
    points: NDArray[np.float32] = np.array(
        [float(s.score) for s in stories], dtype=np.float32
    )
    hn_scores = np.log1p(points) / np.log1p(
        max(points.max(), HN_SCORE_NORMALIZATION_CAP)
    )

    # 5. Compute story ages for adaptive weighting
    now = time.time()
    ages_hours: NDArray[np.float64] = np.array(
        [(now - s.time) / 3600.0 for s in stories]
    )
    rss_mask: NDArray[np.bool_] = np.array([s.id < 0 for s in stories], dtype=bool)

    # 6. Adaptive HN weight based on age (only if hn_weight > 0)
    # hn_weight=0 means pure semantic mode (for testing invariants)
    freshness_boost: NDArray[np.float32] = np.zeros(len(stories), dtype=np.float32)
    if hn_weight > 0:
        # Young stories (<6h): trust semantic more (low HN weight)
        # Old stories (>48h): trust HN score more (higher HN weight)
        young_hn_weight = min(ADAPTIVE_HN_WEIGHT_MIN, ADAPTIVE_HN_WEIGHT_MAX)
        old_hn_weight = max(ADAPTIVE_HN_WEIGHT_MIN, ADAPTIVE_HN_WEIGHT_MAX)
        adaptive_t = np.clip(
            (ages_hours - ADAPTIVE_HN_THRESHOLD_YOUNG)
            / (ADAPTIVE_HN_THRESHOLD_OLD - ADAPTIVE_HN_THRESHOLD_YOUNG),
            0.0,
            1.0,
        )
        hn_weights: NDArray[np.float64] = (
            young_hn_weight + adaptive_t * (old_hn_weight - young_hn_weight)
        )

        # 7. Hybrid Score (per-story adaptive weighting)
        hybrid_scores: NDArray[np.float32] = (
            (1 - hn_weights) * semantic_scores + hn_weights * hn_scores
        ).astype(np.float32)

        # 8. Freshness boost (exponential decay)
        # Newer stories get a boost; score halves every FRESHNESS_HALF_LIFE_HOURS
        freshness: NDArray[np.float64] = np.power(
            2.0, -ages_hours / FRESHNESS_HALF_LIFE_HOURS
        )
        freshness = np.clip(freshness, 0.0, 1.0)
        freshness_boost = (FRESHNESS_MAX_BOOST * freshness).astype(np.float32)
        hybrid_scores = hybrid_scores + freshness_boost

        # RSS stories use semantic score only (no HN score or freshness)
        if np.any(rss_mask):
            hybrid_scores[rss_mask] = semantic_scores[rss_mask]
            freshness_boost[rss_mask] = 0.0
    else:
        # Pure semantic mode (hn_weight=0): no HN score, no freshness
        hybrid_scores = semantic_scores.astype(np.float32)

    # 9. Diversity (MMR)
    results: list[RankResult] = []
    selected_mask: NDArray[np.bool_] = np.zeros(len(cand_emb), dtype=bool)
    cand_sim: NDArray[np.float32] = cosine_similarity(cand_emb, cand_emb)

    for _ in range(min(len(stories), RANKING_MAX_RESULTS)):
        unselected: NDArray[np.int64] = np.where(~selected_mask)[0]
        if not len(unselected):
            break

        mmr_scores: NDArray[np.float32]
        if np.any(selected_mask):
            redundancy: NDArray[np.float32] = np.max(
                cand_sim[unselected][:, selected_mask], axis=1
            )
            mmr_scores = hybrid_scores[unselected] - diversity_lambda * redundancy
        else:
            mmr_scores = hybrid_scores[unselected]

        best_idx: int = int(unselected[np.argmax(mmr_scores)])

        results.append(
            RankResult(
                index=best_idx,
                hybrid_score=float(hybrid_scores[best_idx]),
                best_fav_index=int(best_fav_indices[best_idx]),
                max_sim_score=float(max_sim_scores[best_idx]),
                knn_score=float(raw_knn_scores[best_idx]),
                semantic_score=float(semantic_scores[best_idx]),
                hn_score=float(hn_scores[best_idx]),
                freshness_boost=float(freshness_boost[best_idx]),
            )
        )
        selected_mask[best_idx] = True

    return results
