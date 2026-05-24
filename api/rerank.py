from __future__ import annotations
import os
import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast


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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase
    from api.learned_ranker import OrdinalThresholdModel

from api.constants import (  # noqa: E402
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_CACHE_MAX_FILES,
    EMBEDDING_MIN_CLIP,
    EMBEDDING_MODEL_VERSION,
    CLUSTER_EMBEDDING_CACHE_DIR,
    CLUSTER_EMBEDDING_MODEL_DIR,
    CLUSTER_EMBEDDING_MODEL_VERSION,
    SIMILARITY_MIN,
    TEXT_CONTENT_MAX_TOKENS,
)
from api.cache_utils import atomic_write_json, evict_old_cache_files  # noqa: E402
from api.models import RankResult, Story, StoryDict  # noqa: E402
from api.model_metadata import CURRENT_PRODUCTION_SPEC, load_model_spec  # noqa: E402
from api.config import (  # noqa: E402
    AppConfig,
    ClusteringConfig,
    ClassifierConfig,
)

logger = logging.getLogger(__name__)

RankProgressPhase = Literal[
    "embeddings",
    "scoring",
    "finalize",
    "complete",
]


class RankProgress(TypedDict):
    phase: RankProgressPhase
    current: int
    total: int
    label: str


RankProgressCallback = Callable[[RankProgress], None]


def _set_rank_diagnostics(
    diagnostics: dict[str, object] | None,
    **kwargs: object,
) -> None:
    if diagnostics is None:
        return
    diagnostics.update(kwargs)


def _combine_classifier_features(
    *,
    centroid_feature: NDArray[np.float32],
    pos_knn_feature: NDArray[np.float32],
    neg_knn_feature: NDArray[np.float32],
    config: ClassifierConfig,
    base_embeddings: NDArray[np.float32] | None = None,
    closest_pos_feature: NDArray[np.float32] | None = None,
    closest_neg_feature: NDArray[np.float32] | None = None,
    closest_centroid_feature: NDArray[np.float32] | None = None,
    knn_pos_n1: NDArray[np.float32] | None = None,
    knn_pos_n3: NDArray[np.float32] | None = None,
    knn_pos_n5: NDArray[np.float32] | None = None,
    knn_pos_n10: NDArray[np.float32] | None = None,
    knn_neg_n1: NDArray[np.float32] | None = None,
    knn_neg_n3: NDArray[np.float32] | None = None,
    knn_neg_n5: NDArray[np.float32] | None = None,
    knn_neg_n10: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    columns: list[NDArray[np.float32]] = []
    
    # In "full" mode, we include the original embeddings. 
    # In "bottleneck" or "similarity_only" mode, we only use derived features.
    if config.feature_mode == "full" and base_embeddings is not None:
        columns.append(base_embeddings)

    if config.use_centroid_feature:
        columns.append(centroid_feature.reshape(-1, 1))
    if config.use_pos_knn_feature:
        columns.append(pos_knn_feature.reshape(-1, 1))
    if config.use_neg_knn_feature:
        columns.append(neg_knn_feature.reshape(-1, 1))

    # Add new rich features if configured
    if getattr(config, "use_closest_pos_feature", False) and closest_pos_feature is not None:
        columns.append(closest_pos_feature.reshape(-1, 1))
    if getattr(config, "use_closest_neg_feature", False) and closest_neg_feature is not None:
        columns.append(closest_neg_feature.reshape(-1, 1))
    if getattr(config, "use_closest_centroid_feature", False) and closest_centroid_feature is not None:
        columns.append(closest_centroid_feature.reshape(-1, 1))

    if getattr(config, "use_knn_pos_n1_feature", False) and knn_pos_n1 is not None:
        columns.append(knn_pos_n1.reshape(-1, 1))
    if getattr(config, "use_knn_pos_n3_feature", False) and knn_pos_n3 is not None:
        columns.append(knn_pos_n3.reshape(-1, 1))
    if getattr(config, "use_knn_pos_n5_feature", False) and knn_pos_n5 is not None:
        columns.append(knn_pos_n5.reshape(-1, 1))
    if getattr(config, "use_knn_pos_n10_feature", False) and knn_pos_n10 is not None:
        columns.append(knn_pos_n10.reshape(-1, 1))

    if getattr(config, "use_knn_neg_n1_feature", False) and knn_neg_n1 is not None:
        columns.append(knn_neg_n1.reshape(-1, 1))
    if getattr(config, "use_knn_neg_n3_feature", False) and knn_neg_n3 is not None:
        columns.append(knn_neg_n3.reshape(-1, 1))
    if getattr(config, "use_knn_neg_n5_feature", False) and knn_neg_n5 is not None:
        columns.append(knn_neg_n5.reshape(-1, 1))
    if getattr(config, "use_knn_neg_n10_feature", False) and knn_neg_n10 is not None:
        columns.append(knn_neg_n10.reshape(-1, 1))

    if not columns:
        return np.zeros((len(centroid_feature), 0), dtype=np.float32)
    return np.hstack(columns).astype(np.float32)


def _classifier_metadata_features(
    stories: list[Story], 
    config: AppConfig,
    now: float,
    expected_len: int,
) -> NDArray[np.float32]:
    if not stories or len(stories) != expected_len:
        # Return zeros if metadata is missing or mismatched
        width = 0
        if config.classifier.use_log_points_feature:
            width += 1
        if config.classifier.use_log_comments_feature:
            width += 1
        if config.classifier.use_comment_ratio_feature:
            width += 1
        return np.zeros((expected_len, width), dtype=np.float32)

    columns: list[NDArray[np.float32]] = []
    
    if config.classifier.use_log_points_feature:
        score_normalization_cap = 1000.0
        normalizer = np.log1p(score_normalization_cap)
        points = np.array([max(float(s.score), 0.0) for s in stories], dtype=np.float32)
        columns.append((np.log1p(points) / normalizer).reshape(-1, 1))

    if config.classifier.use_log_comments_feature:
        comment_normalization_cap = 500.0
        normalizer = np.log1p(comment_normalization_cap)
        comments = np.array([max(float(s.comment_count if s.comment_count is not None else 0.0), 0.0) for s in stories], dtype=np.float32)
        columns.append((np.log1p(comments) / normalizer).reshape(-1, 1))

    if config.classifier.use_comment_ratio_feature:
        comments = np.array([max(float(s.comment_count if s.comment_count is not None else 0.0), 0.0) for s in stories], dtype=np.float32)
        points = np.array([max(float(s.score), 0.0) for s in stories], dtype=np.float32)
        ratio = np.log1p(comments) / (np.log1p(points) + 1.0)
        columns.append(ratio.reshape(-1, 1))

    if not columns:
        return np.zeros((expected_len, 0), dtype=np.float32)
        
    return np.hstack(columns).astype(np.float32)


def compute_classifier_similarity_features(
    embs: NDArray[np.float32],
    pos_ref: NDArray[np.float32],
    neg_ref: NDArray[np.float32],
    centroid_ref: NDArray[np.float32],
    config: ClassifierConfig,
    *,
    exclude_self_pos: bool = False,
    exclude_self_neg: bool = False,
) -> dict[str, NDArray[np.float32]]:
    """Compute the first-stage derived similarity features used by the classifier."""
    sim_c = (
        cosine_similarity(embs, centroid_ref)
        if centroid_ref.shape[0] > 0
        else np.zeros((len(embs), 0), dtype=np.float32)
    )
    f_centroid_max = (
        np.max(sim_c, axis=1) if sim_c.shape[1] > 0 else np.zeros(len(embs), dtype=np.float32)
    )

    sim_p = (
        cosine_similarity(embs, pos_ref)
        if pos_ref.shape[0] > 0
        else np.zeros((len(embs), 0), dtype=np.float32)
    )
    if exclude_self_pos and sim_p.shape[1] > 0:
        np.fill_diagonal(sim_p, -1.0)
    f_closest_pos = (
        np.max(sim_p, axis=1) if sim_p.shape[1] > 0 else np.zeros(len(embs), dtype=np.float32)
    )

    sim_n = (
        cosine_similarity(embs, neg_ref)
        if neg_ref.shape[0] > 0
        else np.zeros((len(embs), 0), dtype=np.float32)
    )
    if exclude_self_neg and sim_n.shape[1] > 0:
        np.fill_diagonal(sim_n, -1.0)
    f_closest_neg = (
        np.max(sim_n, axis=1) if sim_n.shape[1] > 0 else np.zeros(len(embs), dtype=np.float32)
    )

    def compute_knn_mean(sims: NDArray[np.float32], n: int) -> NDArray[np.float32]:
        num_ref = sims.shape[1]
        if num_ref == 0:
            return np.zeros(sims.shape[0], dtype=np.float32)
        effective_n = min(num_ref, n)
        top_n = np.partition(sims, -effective_n, axis=1)[:, -effective_n:]
        return np.mean(top_n, axis=1)

    knn_p1 = compute_knn_mean(sim_p, 1)
    knn_p3 = compute_knn_mean(sim_p, 3)
    knn_p5 = compute_knn_mean(sim_p, 5)
    knn_p10 = compute_knn_mean(sim_p, 10)

    knn_n1 = compute_knn_mean(sim_n, 1)
    knn_n3 = compute_knn_mean(sim_n, 3)
    knn_n5 = compute_knn_mean(sim_n, 5)
    knn_n10 = compute_knn_mean(sim_n, 10)

    k_pos = min(pos_ref.shape[0], config.k_feat)
    if k_pos > 0 and sim_p.shape[1] > 0:
        f_knn_pos = np.median(np.partition(sim_p, -k_pos, axis=1)[:, -k_pos:], axis=1)
    else:
        f_knn_pos = np.zeros(len(embs), dtype=np.float32)

    k_neg = min(neg_ref.shape[0], config.k_feat)
    if k_neg > 0 and sim_n.shape[1] > 0:
        f_knn_neg = np.median(np.partition(sim_n, -k_neg, axis=1)[:, -k_neg:], axis=1)
    else:
        f_knn_neg = np.zeros(len(embs), dtype=np.float32)

    return {
        "centroid_feature": f_centroid_max,
        "pos_knn_feature": f_knn_pos,
        "neg_knn_feature": f_knn_neg,
        "closest_pos": f_closest_pos,
        "closest_neg": f_closest_neg,
        "closest_centroid": f_centroid_max,
        "knn_pos_n1": knn_p1,
        "knn_pos_n3": knn_p3,
        "knn_pos_n5": knn_p5,
        "knn_pos_n10": knn_p10,
        "knn_neg_n1": knn_n1,
        "knn_neg_n3": knn_n3,
        "knn_neg_n5": knn_n5,
        "knn_neg_n10": knn_n10,
    }


def stack_classifier_similarity_features(
    derived_dict: dict[str, NDArray[np.float32]],
    config: ClassifierConfig,
    *,
    base_embeddings: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Assemble the similarity feature matrix using the runtime classifier toggles."""
    row_count = len(next(iter(derived_dict.values()))) if derived_dict else 0
    if base_embeddings is None:
        base_embeddings = np.zeros((row_count, 0), dtype=np.float32)
    return _combine_classifier_features(
        centroid_feature=derived_dict["centroid_feature"],
        pos_knn_feature=derived_dict["pos_knn_feature"],
        neg_knn_feature=derived_dict["neg_knn_feature"],
        config=config,
        base_embeddings=base_embeddings,
        closest_pos_feature=derived_dict["closest_pos"],
        closest_neg_feature=derived_dict["closest_neg"],
        closest_centroid_feature=derived_dict["closest_centroid"],
        knn_pos_n1=derived_dict["knn_pos_n1"],
        knn_pos_n3=derived_dict["knn_pos_n3"],
        knn_pos_n5=derived_dict["knn_pos_n5"],
        knn_pos_n10=derived_dict["knn_pos_n10"],
        knn_neg_n1=derived_dict["knn_neg_n1"],
        knn_neg_n3=derived_dict["knn_neg_n3"],
        knn_neg_n5=derived_dict["knn_neg_n5"],
        knn_neg_n10=derived_dict["knn_neg_n10"],
    )


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
        self.spec = load_model_spec(model_dir)

        from transformers import AutoTokenizer

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_dir
        )

        providers = ["CPUExecutionProvider"]

        self.session: ort.InferenceSession = ort.InferenceSession(
            f"{model_dir}/model.onnx", providers=providers
        )
        self.model_id: str = self.spec.cache_key
        self.embedding_dim = self._infer_embedding_dim(model_dir)
        # Hugging Face fast tokenizers mutate internal padding/truncation state per call,
        # so tokenizer access must be serialized across threads.
        self._tokenizer_lock = threading.Lock()
        self._input_names: list[str] = [node.name for node in self.session.get_inputs()]

    def _infer_embedding_dim(self, model_dir: str) -> int:
        output_shape = self.session.get_outputs()[0].shape
        if output_shape and isinstance(output_shape[-1], int):
            return int(output_shape[-1])
        try:
            raw = json.loads((Path(model_dir) / "config.json").read_text())
            hidden_size = raw.get("hidden_size")
            if isinstance(hidden_size, int):
                return hidden_size
        except Exception:
            pass
        return 768

    def truncate_to_token_budget(self, text: str, max_tokens: int) -> tuple[str, bool]:
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
                token_ids = (
                    input_ids[0] if isinstance(input_ids[0], list) else input_ids
                )
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
        normalize_embeddings: bool | None = None,
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
                    NDArray[np.int64],
                    inputs["attention_mask"].astype(np.int64, copy=True),
                )

                ort_inputs: dict[str, NDArray[np.int64]] = {
                    k: v.astype(np.int64)
                    for k, v in inputs.items()
                    if k in self._input_names
                }

            # InferenceSession.run is thread-safe; keep it outside tokenizer lock
            # so embedding inference can run in parallel across callers.
            outputs = self.session.run(None, ort_inputs)
            last_hidden_state: NDArray[np.float32] = cast(
                NDArray[np.float32], outputs[0]
            )

            spec = getattr(self, "spec", CURRENT_PRODUCTION_SPEC)
            if spec.pooling == "cls":
                batch_embeddings = last_hidden_state[:, 0, :]
            else:
                mask_expanded: NDArray[np.float64] = np.expand_dims(
                    attention_mask, -1
                ).astype(float)
                sum_embeddings: NDArray[np.float32] = np.sum(
                    last_hidden_state * mask_expanded, axis=1
                )
                sum_mask: NDArray[np.float64] = np.clip(
                    mask_expanded.sum(axis=1), a_min=EMBEDDING_MIN_CLIP, a_max=None
                )
                batch_embeddings = sum_embeddings / sum_mask

            should_normalize = (
                spec.normalize
                if normalize_embeddings is None
                else normalize_embeddings
            )
            if should_normalize:
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
                logger.info("Loading BERT embedding model (ONNX)...")
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
                logger.info("Loading cluster embedding model (ONNX)...")
                _cluster_model = ONNXEmbeddingModel(
                    model_dir=CLUSTER_EMBEDDING_MODEL_DIR
                )
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

    truncated_count = 0
    processed_texts: list[str] = []
    model_spec = getattr(model, "spec", None)

    for t in texts:
        text = model_spec.prepare_text(t, is_query=is_query) if model_spec else t
        truncated_text, was_truncated = model.truncate_to_token_budget(
            text,
            min(
                TEXT_CONTENT_MAX_TOKENS,
                model_spec.max_tokens if model_spec else TEXT_CONTENT_MAX_TOKENS,
            ),
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

    expected_dim = getattr(model, "embedding_dim", 768)

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

        if vec is not None and vec.ndim == 1:
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
    config: ClusteringConfig = ClusteringConfig(),
) -> NDArray[np.float32]:
    """
    Cluster user interest embeddings into K centroids.
    Returns centroids array of shape (n_clusters, embedding_dim).
    """
    centroids, _ = cluster_interests_with_labels(embeddings, config=config)
    return centroids


def cluster_interests_with_labels(
    embeddings: NDArray[np.float32],
    config: ClusteringConfig = ClusteringConfig(),
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Cluster user interest embeddings using a cosine-aware algorithm.
    Returns (centroids, labels) where:
      - centroids: shape (n_clusters, embedding_dim)
      - labels: shape (n_samples,) cluster assignment per sample
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

    n_samples = len(embeddings)
    if n_samples == 0:
        return embeddings, np.array([], dtype=np.int32)

    if n_samples < config.min_samples_per_cluster * 2:
        # Not enough for meaningful clustering
        labels = np.zeros(n_samples, dtype=np.int32)
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        return centroid.astype(np.float32), labels

    # Normalize embeddings for cosine-like behavior (unless metric is euclidean)
    if (
        config.metric == "euclidean"
        and config.algorithm == "agglomerative"
    ):
        normalized = embeddings
    else:
        normalized = _normalize_embeddings(embeddings)

    # Use fixed number of clusters or distance threshold
    effective_n_clusters = min(
        config.default_count,
        config.max_clusters,
        n_samples // max(1, config.min_samples_per_cluster),
    )
    effective_n_clusters = max(effective_n_clusters, config.min_clusters)

    if config.algorithm == "spectral":
        neighbors = min(config.spectral_neighbors, max(2, n_samples - 1))
        clustering = SpectralClustering(
            n_clusters=effective_n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
            random_state=0,
        )
        labels = clustering.fit_predict(normalized).astype(np.int32)
    elif config.algorithm == "agglomerative":
        # If threshold is provided, use auto-k
        use_n = effective_n_clusters if config.distance_threshold is None else None
        clustering = AgglomerativeClustering(
            n_clusters=use_n,
            distance_threshold=config.distance_threshold,
            metric=config.metric,
            linkage=config.linkage,
        )
        labels = clustering.fit_predict(normalized).astype(np.int32)
        if config.distance_threshold is not None:
            labels = _merge_closest_clusters_to_limit(
                embeddings,
                labels,
                max_clusters=effective_n_clusters,
            )
    else:
        kmeans = KMeans(
            n_clusters=effective_n_clusters,
            n_init=10,
            random_state=0,
        )
        labels = kmeans.fit_predict(normalized).astype(np.int32)

    # Bypass heuristic post-processing if using threshold-based agglomerative
    # (Ward linkage handles balance and coherence natively)
    if not (config.algorithm == "agglomerative" and config.distance_threshold is not None):
        labels = _refine_cluster_assignments(normalized, labels, config.refine_iters)
        labels = _merge_small_clusters(embeddings, labels, config.min_samples_per_cluster)

        max_size = max(
            config.min_samples_per_cluster,
            min(
                config.max_cluster_size,
                int(math.ceil(n_samples * config.max_cluster_fraction)),
            ),
        )
        max_clusters = min(config.max_clusters, n_samples // max(1, config.min_samples_per_cluster))
        labels = _split_large_clusters(
            embeddings,
            labels,
            min_size=config.min_samples_per_cluster,
            max_size=max_size,
            max_clusters=max_clusters,
        )

    centroids = _centroids_from_labels(embeddings, labels)

    if not (config.algorithm == "agglomerative" and config.distance_threshold is not None):
        centroids, labels, _ = split_outlier_clusters(
            embeddings, labels, config.outlier_similarity_threshold
        )
    return centroids, labels


def _centroids_from_labels(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Compute centroids for each cluster label."""
    unique_labels = sorted(set(labels))
    centroids = []
    for lbl in unique_labels:
        mask = labels == lbl
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids, dtype=np.float32)


def _relabel_consecutive(labels: NDArray[np.int32]) -> NDArray[np.int32]:
    remap = {old: i for i, old in enumerate(sorted(set(labels)))}
    return np.array([remap[int(lbl)] for lbl in labels], dtype=np.int32)


def _merge_closest_clusters_to_limit(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int32],
    max_clusters: int,
) -> NDArray[np.int32]:
    """Merge nearest centroid pairs until the cluster count is within the cap."""
    if max_clusters <= 0 or len(labels) == 0:
        return labels

    current = _relabel_consecutive(labels)
    while len(set(current)) > max_clusters:
        centroids = _centroids_from_labels(embeddings, current)
        if len(centroids) <= 1:
            break

        diff = centroids[:, None, :] - centroids[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(distances, np.inf)
        source_idx, target_idx = np.unravel_index(
            int(np.argmin(distances)),
            distances.shape,
        )
        source_label = int(source_idx)
        target_label = int(target_idx)
        if source_label == target_label:
            break

        current[current == source_label] = target_label
        current = _relabel_consecutive(current)

    return current


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
) -> tuple[NDArray[np.float32], NDArray[np.int32], int]:
    """Split low-similarity items into singleton clusters.

    Returns (new_centroids, new_labels, outlier_count).
    """
    if len(labels) == 0:
        return embeddings, labels, 0
    if min_similarity <= SIMILARITY_MIN:
        centroids = _centroids_from_labels(embeddings, labels)
        return centroids, labels, 0

    centroids = _centroids_from_labels(embeddings, labels)
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
    new_centroids = _centroids_from_labels(embeddings, new_labels)
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
        next_labels = np.array(
            [unique_labels[int(i)] for i in next_idx], dtype=np.int32
        )
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
    model_or_positive_embeddings: OrdinalThresholdModel | NDArray[np.float32] | None = None,
    positive_embeddings: NDArray[np.float32] | None = None,
    negative_embeddings: NDArray[np.float32] | None = None,
    config: AppConfig = AppConfig(),
    progress_callback: Callable[[int, int], None] | None = None,
    diagnostics: dict[str, object] | None = None,
    positive_stories: list[Story] | None = None,
    negative_stories: list[Story] | None = None,
    cluster_names: dict[int, str] | None = None,
    cluster_keywords: dict[int, str] | None = None,
) -> list[RankResult]:
    """
    Rank candidate stories with the feedback-trained single model.
    """
    if not stories:
        return []
    from api.learned_ranker import (
        DOWNVOTE_LABEL,
        UPVOTE_LABEL,
        _predict_ordinal_outputs,
        train_model_from_matrix,
    )

    model: OrdinalThresholdModel | None
    if (
        model_or_positive_embeddings is not None
        and hasattr(model_or_positive_embeddings, "at_least_neutral")
        and hasattr(model_or_positive_embeddings, "upvote")
    ):
        model = cast("OrdinalThresholdModel", model_or_positive_embeddings)
    else:
        model = None
        if model_or_positive_embeddings is not None:
            positive_embeddings = cast(NDArray[np.float32], model_or_positive_embeddings)

    def report_progress(
        phase: RankProgressPhase,
        current: int,
        total: int,
        label: str,
    ) -> None:
        if progress_callback:
            progress_callback(
                {
                    "phase": phase,
                    "current": current,
                    "total": max(total, 1),
                    "label": label,
                }
            )

    cand_texts: list[str] = [s.text_content for s in stories]

    def embedding_progress(curr: int, total: int) -> None:
        report_progress("embeddings", curr, total, "Embedding candidate stories")

    cand_emb: NDArray[np.float32] = get_embeddings(
        cand_texts, progress_callback=embedding_progress
    )
    report_progress("scoring", 0, 1, "Scoring candidates")

    embedding_dim = int(cand_emb.shape[1]) if cand_emb.ndim == 2 else 0
    X_pos = (
        positive_embeddings
        if positive_embeddings is not None
        else np.zeros((0, embedding_dim), dtype=np.float32)
    )
    X_neg = (
        negative_embeddings
        if negative_embeddings is not None
        else np.zeros((0, embedding_dim), dtype=np.float32)
    )
    positive_count = len(X_pos)
    negative_count = len(X_neg)
    _set_rank_diagnostics(
        diagnostics,
        positive_count=int(positive_count),
        negative_count=int(negative_count),
        base_feature_dim=int(cand_emb.shape[1]),
        derived_feature_dim=0,
        classifier_metadata_features_used=False,
        classifier_metadata_feature_dim=0,
        local_hidden_penalty_applied=False,
        local_hidden_penalty_mean=0.0,
        local_hidden_penalty_max=0.0,
    )

    # Display-only scores (always populated regardless of scoring path)
    semantic_scores: NDArray[np.float32] = np.zeros(len(stories), dtype=np.float32)
    max_sim_scores: NDArray[np.float32] = np.zeros(len(stories), dtype=np.float32)
    best_fav_indices: NDArray[np.int64] = np.full(len(stories), -1, dtype=np.int64)
    raw_knn_scores: NDArray[np.float32] = np.zeros(len(stories), dtype=np.float32)
    cluster_max_scores: NDArray[np.float32] = np.zeros(len(stories), dtype=np.float32)
    centroids: NDArray[np.float32]
    if len(X_pos) > 0:
        centroids, _ = cluster_interests_with_labels(X_pos, config=config.clustering)
    else:
        centroids = np.zeros((0, embedding_dim), dtype=np.float32)

    if len(X_pos) > 0:
        # Always compute display scores against positive embeddings
        sim_pos_ui = cosine_similarity(X_pos, cand_emb)  # (n_pos, n_cand)
        max_sim_scores = np.max(sim_pos_ui, axis=0)
        best_fav_indices = np.argmax(sim_pos_ui, axis=0)

        k = min(len(X_pos), config.semantic.knn_neighbors)
        if k > 0:
            top_k_sims = np.partition(sim_pos_ui, -k, axis=0)[-k:, :]
            raw_knn_scores = np.median(top_k_sims, axis=0).astype(np.float32)
    if len(centroids) > 0:
        cluster_sim = cosine_similarity(centroids, cand_emb)
        cluster_max_scores = np.max(cluster_sim, axis=0).astype(np.float32)

    cand_derived = compute_classifier_similarity_features(
        cand_emb,
        X_pos,
        X_neg,
        centroids,
        config.classifier,
    )
    cand_metadata = _classifier_metadata_features(
        stories, config, time.time(), len(cand_emb)
    )
    cand_features = stack_classifier_similarity_features(
        cand_derived,
        config.classifier,
        base_embeddings=cand_emb,
    )
    if cand_metadata.shape[1] > 0:
        cand_features = np.hstack([cand_features, cand_metadata])
    derived_width = stack_classifier_similarity_features(
        cand_derived,
        config.classifier,
        base_embeddings=np.zeros((len(cand_emb), 0), dtype=np.float32),
    ).shape[1]
    _set_rank_diagnostics(
        diagnostics,
        derived_feature_dim=int(derived_width),
        classifier_metadata_features_used=bool(cand_metadata.shape[1] > 0),
        classifier_metadata_feature_dim=int(cand_metadata.shape[1]),
    )
    if model is None and len(X_pos) >= config.classifier.min_positive_examples:
        if len(X_neg) >= config.classifier.min_negative_examples:
            pos_derived = compute_classifier_similarity_features(
                X_pos,
                X_pos,
                X_neg,
                centroids,
                config.classifier,
                exclude_self_pos=True,
            )
            neg_derived = compute_classifier_similarity_features(
                X_neg,
                X_pos,
                X_neg,
                centroids,
                config.classifier,
                exclude_self_neg=True,
            )
            pos_features = stack_classifier_similarity_features(
                pos_derived,
                config.classifier,
                base_embeddings=X_pos,
            )
            neg_features = stack_classifier_similarity_features(
                neg_derived,
                config.classifier,
                base_embeddings=X_neg,
            )
            pos_metadata = _classifier_metadata_features(
                positive_stories or [],
                config,
                time.time(),
                len(X_pos),
            )
            neg_metadata = _classifier_metadata_features(
                negative_stories or [],
                config,
                time.time(),
                len(X_neg),
            )
            if pos_metadata.shape[1] > 0:
                pos_features = np.hstack([pos_features, pos_metadata])
            if neg_metadata.shape[1] > 0:
                neg_features = np.hstack([neg_features, neg_metadata])
            train_rows = np.vstack([pos_features, neg_features]).astype(np.float32)
            train_labels = np.concatenate(
                [
                    np.full(len(X_pos), UPVOTE_LABEL, dtype=np.int64),
                    np.full(len(X_neg), DOWNVOTE_LABEL, dtype=np.int64),
                ]
            )
            model = train_model_from_matrix(train_rows, train_labels, config.single_model)
        else:
            semantic_scores = cluster_max_scores

    if model is None and not np.any(semantic_scores):
        semantic_scores = cluster_max_scores

    if model is not None:
        try:
            expected_n = model.at_least_neutral.steps[0][1].n_features_in_
        except AttributeError:
            expected_n = cand_features.shape[1]

        if expected_n != cand_features.shape[1]:
            # Build full single-model features (embeddings + derived + metadata)
            columns = [cand_emb.astype(np.float32)]
            derived_rows = stack_classifier_similarity_features(
                cand_derived,
                config.classifier,
                base_embeddings=np.zeros((len(stories), 0), dtype=np.float32),
            )
            if derived_rows.shape[1] > 0:
                columns.append(derived_rows.astype(np.float32))
            if cand_metadata.shape[1] > 0:
                columns.append(cand_metadata.astype(np.float32))
            full_features = np.hstack(columns).astype(np.float32)
            semantic_scores, _, _, _ = _predict_ordinal_outputs(model, full_features)
        else:
            semantic_scores, _, _, _ = _predict_ordinal_outputs(model, cand_features)

    report_progress("scoring", 1, 1, "Scored candidates")

    hybrid_scores = semantic_scores
    ranked_indices = np.argsort(-hybrid_scores, kind="stable")

    report_progress("finalize", 0, 1, "Finalizing ranked stories")
    results = [
        RankResult(
            index=int(best_idx),
            hybrid_score=float(hybrid_scores[best_idx]),
            best_fav_index=int(best_fav_indices[best_idx]),
            max_sim_score=float(max_sim_scores[best_idx]),
            knn_score=float(raw_knn_scores[best_idx]),
            max_cluster_score=float(cluster_max_scores[best_idx]),
            semantic_score=float(semantic_scores[best_idx]),
        )
        for best_idx in ranked_indices
    ]
    report_progress("finalize", 1, 1, "Finalized ranked stories")
    report_progress("complete", 1, 1, "Reranking complete")
    return results
