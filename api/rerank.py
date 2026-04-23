from __future__ import annotations
import os
import hashlib
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

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

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase

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
    FRESHNESS_ENABLED,
    FRESHNESS_HALF_LIFE_HOURS,
    FRESHNESS_MAX_BOOST,
    HN_SCORE_NORMALIZATION_CAP,
    KNN_NEIGHBORS,
    CLUSTER_OUTLIER_SIMILARITY_THRESHOLD,
    MAX_CLUSTERS,
    MAX_CLUSTER_FRACTION,
    MAX_CLUSTER_SIZE,
    MIN_CLUSTERS,
    MIN_SAMPLES_PER_CLUSTER,
    POSITIVE_RECENCY_HALF_LIFE_DAYS,
    RANKING_HN_WEIGHT,
    RANKING_MAX_RESULTS,
    RANKING_NEGATIVE_WEIGHT,
    KNN_SIGMOID_K,
    KNN_MAXSIM_WEIGHT,
    CLASSIFIER_K_FEAT,
    CLASSIFIER_LOCAL_HIDDEN_PENALTY_K,
    CLASSIFIER_LOCAL_HIDDEN_PENALTY_WEIGHT,
    CLASSIFIER_NEG_SAMPLE_WEIGHT,
    CLASSIFIER_CV_SCORING,
    CLASSIFIER_USE_BALANCED_CLASS_WEIGHT,
    CLASSIFIER_USE_CENTROID_FEATURE,
    CLASSIFIER_USE_LOCAL_HIDDEN_PENALTY,
    CLASSIFIER_USE_POS_KNN_FEATURE,
    CLASSIFIER_USE_NEG_KNN_FEATURE,
    SIMILARITY_MIN,
    TEXT_CONTENT_MAX_TOKENS,
)
from api.models import RankResult, Story, StoryDict  # noqa: E402

logger = logging.getLogger(__name__)


def _set_rank_diagnostics(
    diagnostics: dict[str, object] | None,
    **kwargs: object,
) -> None:
    if diagnostics is None:
        return
    diagnostics.update(kwargs)


def compute_recency_weights(
    timestamps: Sequence[int | float],
    *,
    half_life_seconds: float = POSITIVE_RECENCY_HALF_LIFE_DAYS * 24 * 3600,
    now: float | None = None,
) -> NDArray[np.float32] | None:
    """Return exponential recency weights aligned to the input timestamp order."""
    if not timestamps:
        return None
    current_time = time.time() if now is None else now
    times = np.array(timestamps, dtype=np.float32)
    ages = current_time - times
    return np.exp(-ages * np.log(2) / half_life_seconds).astype(np.float32)


def _combine_classifier_features(
    *,
    centroid_feature: NDArray[np.float32],
    pos_knn_feature: NDArray[np.float32],
    neg_knn_feature: NDArray[np.float32],
) -> NDArray[np.float32]:
    columns: list[NDArray[np.float32]] = []
    if CLASSIFIER_USE_CENTROID_FEATURE:
        columns.append(centroid_feature.reshape(-1, 1))
    if CLASSIFIER_USE_POS_KNN_FEATURE:
        columns.append(pos_knn_feature.reshape(-1, 1))
    if CLASSIFIER_USE_NEG_KNN_FEATURE:
        columns.append(neg_knn_feature.reshape(-1, 1))
    if not columns:
        return np.zeros((len(centroid_feature), 0), dtype=np.float32)
    return np.hstack(columns).astype(np.float32)






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

        from transformers import AutoTokenizer

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_dir
        )

        providers = ["CPUExecutionProvider"]

        self.session: ort.InferenceSession = ort.InferenceSession(
            f"{model_dir}/model.onnx", providers=providers
        )
        self.model_id: str = "bge-base-en-v1.5"
        # Hugging Face fast tokenizers mutate internal padding/truncation state per call,
        # so tokenizer access must be serialized across threads.
        self._tokenizer_lock = threading.Lock()
        self._input_names: list[str] = [node.name for node in self.session.get_inputs()]

    def truncate_to_token_budget(
        self, text: str, max_tokens: int
    ) -> tuple[str, bool]:
        """Truncate text to token budget using tokenizer in a thread-safe way."""
        with self._tokenizer_lock:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=max_tokens,
                return_tensors=None,
            )
            input_ids = encoded.get("input_ids", [])
            if isinstance(input_ids, list) and input_ids:
                token_ids = input_ids[0] if isinstance(input_ids[0], list) else input_ids
            else:
                return text, False

            truncated = len(token_ids) >= max_tokens
            if not truncated:
                return text, False
            truncated_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return truncated_text, True

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
            with self._tokenizer_lock:
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

                ort_inputs: dict[str, NDArray[np.int64]] = {
                    k: v.astype(np.int64) for k, v in inputs.items() if k in self._input_names
                }

            # InferenceSession.run is thread-safe; keep it outside tokenizer lock
            # so embedding inference can run in parallel across callers.
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

    for t in texts:
        text = f"{prefix}{t}"
        truncated_text, was_truncated = model.truncate_to_token_budget(
            text, TEXT_CONTENT_MAX_TOKENS
        )
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
                with np.load(cache_path_npz) as data:
                    vec = cast(NDArray[np.float32], data["embedding"])
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


























































def rank_stories(
    stories: list[Story],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None = None,
    positive_weights: NDArray[np.float32] | None = None,
    hn_weight: float = RANKING_HN_WEIGHT,
    neg_weight: float = RANKING_NEGATIVE_WEIGHT,
    diversity_lambda: float = 0.0,
    use_classifier: bool = False,
    use_contrastive: bool = False,
    knn_k: int = KNN_NEIGHBORS,
    progress_callback: Callable[[int, int], None] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> list[RankResult]:
    """
    Rank candidate stories by relevance to user interests.

    Uses semantic ranking with optional classifier scoring when sufficient
    positive and hidden history is available.
    """
    if not stories:
        return []

    cand_texts: list[str] = [s.text_content for s in stories]
    cand_emb: NDArray[np.float32] = get_embeddings(
        cand_texts, progress_callback=progress_callback
    )
    positive_count = 0 if positive_embeddings is None else len(positive_embeddings)
    negative_count = 0 if negative_embeddings is None else len(negative_embeddings)
    _set_rank_diagnostics(
        diagnostics,
        classifier_requested=bool(use_classifier),
        classifier_used=False,
        classifier_failure_reason=None,
        positive_count=int(positive_count),
        negative_count=int(negative_count),
        base_feature_dim=int(cand_emb.shape[1]),
        derived_feature_dim=0,
        local_hidden_penalty_applied=False,
        local_hidden_penalty_mean=0.0,
        local_hidden_penalty_max=0.0,
    )

    semantic_scores: NDArray[np.float32]
    max_sim_scores: NDArray[np.float32]
    best_fav_indices: NDArray[np.int64]
    raw_knn_scores: NDArray[np.float32]  # For display (before sigmoid)

    # Check if we can/should use classifier
    classifier_success = False
    classifier_requested_and_eligible = (
        use_classifier
        and positive_embeddings is not None
        and negative_embeddings is not None
        and len(positive_embeddings) >= 5
        and len(negative_embeddings) >= 5
    )
    if classifier_requested_and_eligible:
        try:
            # Prepare training data
            X_pos = cast(NDArray[np.float32], positive_embeddings)
            X_neg = cast(NDArray[np.float32], negative_embeddings)
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

                return _combine_classifier_features(
                    centroid_feature=f_centroid.astype(np.float32),
                    pos_knn_feature=f_knn_pos.astype(np.float32),
                    neg_knn_feature=f_knn_neg.astype(np.float32),
                )

            pos_derived = _derived_features(
                X_pos, X_pos, X_neg, centroids, k_feat, k_neg_feat,
                exclude_self_pos=True,
            )
            neg_derived = _derived_features(
                X_neg, X_pos, X_neg, centroids, k_feat, k_neg_feat,
                exclude_self_neg=True,
            )
            if pos_derived.shape[1] > 0:
                train_derived = np.vstack([pos_derived, neg_derived])
                X_train = np.hstack([X_train, train_derived])

            cand_derived = _derived_features(
                cand_emb, X_pos, X_neg, centroids, k_feat, k_neg_feat,
            )
            cand_features = (
                np.hstack([cand_emb, cand_derived])
                if cand_derived.shape[1] > 0
                else cand_emb
            )
            _set_rank_diagnostics(
                diagnostics,
                derived_feature_dim=int(cand_derived.shape[1]),
            )

            clf = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=3,
                class_weight=(
                    "balanced" if CLASSIFIER_USE_BALANCED_CLASS_WEIGHT else None
                ),
                l1_ratios=(0.0,),
                solver="liblinear",
                scoring=CLASSIFIER_CV_SCORING,
                use_legacy_attributes=False,
            )
            clf.fit(X_train, y_train, sample_weight=sample_weights)

            # Predict probabilities (class 1 = positive interest)
            probs = clf.predict_proba(cand_features)[:, 1]
            semantic_scores = probs.astype(np.float32)

            local_hidden_penalty = np.zeros(len(stories), dtype=np.float32)
            if (
                CLASSIFIER_USE_LOCAL_HIDDEN_PENALTY
                and CLASSIFIER_LOCAL_HIDDEN_PENALTY_WEIGHT > 0
                and len(X_neg) > 0
            ):
                penalty_k = min(len(X_neg), CLASSIFIER_LOCAL_HIDDEN_PENALTY_K)
                if penalty_k > 0:
                    hidden_sim = cosine_similarity(cand_emb, X_neg)
                    local_hidden_penalty = np.median(
                        np.partition(hidden_sim, -penalty_k, axis=1)[:, -penalty_k:],
                        axis=1,
                    ).astype(np.float32)
                    local_hidden_penalty = np.clip(local_hidden_penalty, 0.0, None)
                    local_hidden_penalty *= CLASSIFIER_LOCAL_HIDDEN_PENALTY_WEIGHT
                    semantic_scores = np.clip(
                        semantic_scores - local_hidden_penalty, 0.0, 1.0
                    ).astype(np.float32)
                    _set_rank_diagnostics(
                        diagnostics,
                        local_hidden_penalty_applied=True,
                        local_hidden_penalty_mean=float(np.mean(local_hidden_penalty)),
                        local_hidden_penalty_max=float(np.max(local_hidden_penalty)),
                    )

            # We still need max_sim_scores for the UI "Similar to..."
            sim_pos_ui = cosine_similarity(X_pos, cand_emb)
            max_sim_scores = np.max(sim_pos_ui, axis=0)
            best_fav_indices = np.argmax(sim_pos_ui, axis=0)

            # Compute k-NN scores for display
            k = min(len(X_pos), knn_k)
            if k > 0:
                top_k_sims = np.partition(sim_pos_ui, -k, axis=0)[-k:, :]
                raw_knn_scores = np.median(top_k_sims, axis=0).astype(np.float32)
            else:
                raw_knn_scores = np.zeros(len(stories), dtype=np.float32)

            classifier_success = True
            _set_rank_diagnostics(
                diagnostics,
                classifier_used=True,
                classifier_failure_reason=None,
            )
        except Exception as e:
            # Fallback to heuristic on error
            logger.warning(f"Classifier training failed, using heuristic: {e}")
            _set_rank_diagnostics(
                diagnostics,
                classifier_failure_reason=f"{type(e).__name__}: {e}",
            )
    elif use_classifier:
        _set_rank_diagnostics(
            diagnostics,
            classifier_failure_reason="insufficient_examples",
        )

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

    # 3. HN Gravity Score (Log-scaled)
    # We use a log scale so that high-point stories punch through without dominating
    points: NDArray[np.float32] = np.array(
        [float(s.score) for s in stories], dtype=np.float32
    )
    hn_scores = np.log1p(points) / np.log1p(
        max(points.max(), HN_SCORE_NORMALIZATION_CAP)
    )

    # 4. Compute story ages for adaptive weighting
    now = time.time()
    ages_hours: NDArray[np.float64] = np.array(
        [(now - s.time) / 3600.0 for s in stories]
    )

    # 5. Adaptive HN weight based on age (only if hn_weight > 0)
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

        # 6. Hybrid Score (per-story adaptive weighting)
        hybrid_scores: NDArray[np.float32] = (
            (1 - hn_weights) * semantic_scores + hn_weights * hn_scores
        ).astype(np.float32)

        # 7. Freshness boost (exponential decay)
        # Newer stories get a boost; score halves every FRESHNESS_HALF_LIFE_HOURS
        if FRESHNESS_ENABLED and FRESHNESS_MAX_BOOST > 0:
            freshness: NDArray[np.float64] = np.power(
                2.0, -ages_hours / FRESHNESS_HALF_LIFE_HOURS
            )
            freshness = np.clip(freshness, 0.0, 1.0)
            freshness_boost = (FRESHNESS_MAX_BOOST * freshness).astype(np.float32)
            hybrid_scores = hybrid_scores + freshness_boost
    else:
        # Pure semantic mode (hn_weight=0): no HN score, no freshness
        hybrid_scores = semantic_scores.astype(np.float32)

    ranked_indices = np.argsort(-hybrid_scores, kind="stable")[: min(len(stories), RANKING_MAX_RESULTS)]
    return [
        RankResult(
            index=int(best_idx),
            hybrid_score=float(hybrid_scores[best_idx]),
            best_fav_index=int(best_fav_indices[best_idx]),
            max_sim_score=float(max_sim_scores[best_idx]),
            knn_score=float(raw_knn_scores[best_idx]),
            semantic_score=float(semantic_scores[best_idx]),
            hn_score=float(hn_scores[best_idx]),
            freshness_boost=float(freshness_boost[best_idx]),
        )
        for best_idx in ranked_indices
    ]
