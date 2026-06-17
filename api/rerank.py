from __future__ import annotations
import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast


from api.env_setup import ensure_joblib_settings

ensure_joblib_settings()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase
    from api.ordinal_model import OrdinalThresholdModel

from api.constants import (  # noqa: E402
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_CACHE_MAX_FILES,
    EMBEDDING_MIN_CLIP,
    EMBEDDING_MODEL_VERSION,
    SIMILARITY_MIN,
    TEXT_CONTENT_MAX_TOKENS,
)
from api.models import RankResult, Story, StoryDict  # noqa: E402
from api.model_metadata import CURRENT_PRODUCTION_SPEC, load_model_spec  # noqa: E402
from api.config import (  # noqa: E402
    AppConfig,
    ClusteringConfig,
    ClassifierConfig,
)
from api.telemetry_features import extract_domain_with_fallback  # noqa: E402


SIMILARITY_FEATURES: frozenset[str] = frozenset(
    {
        "centroid",
        "pos_knn",
        "neg_knn",
        "pos_neg_ratio",
        "closest_pos",
        "closest_neg",
        "closest_margin",
        "embedding_magnitude",
    }
)

MetadataFeatureFn = Callable[[list[Story], float], NDArray[np.float32]]
METADATA_FEATURES: dict[str, MetadataFeatureFn] = {}


# Thread-local caches for metadata feature functions.
# Each rank_stories call has its own local_density and cluster_size arrays,
# eliminating races in parallel CV (_local_density_cache, _cluster_size_cache).
class _RankCache(threading.local):
    def __init__(self) -> None:
        self.local_density: NDArray[np.float32] | None = None
        self.cluster_size: NDArray[np.int32] | None = None
        self.domain_trust: dict[str, float] = {}
        self.story_age_at_vote_map: dict[int, float] = {}
        self.domain_recency_map: dict[str, float] = {}


_rank_cache = _RankCache()


def _populate_rank_cache_metadata(
    positive_stories: list[Story] | None,
    negative_stories: list[Story] | None,
    now: float,
) -> None:
    _rank_cache.domain_trust.clear()
    _rank_cache.story_age_at_vote_map.clear()
    _rank_cache.domain_recency_map.clear()

    counts: dict[str, list[int]] = {}

    for stories, idx in [(positive_stories, 0), (negative_stories, 1)]:
        if not stories:
            continue
        for s in stories:
            if (
                s.id
                and s.feedback_updated_at > 0
                and s.time > 0
                and s.feedback_updated_at > s.time
            ):
                _rank_cache.story_age_at_vote_map[s.id] = (
                    s.feedback_updated_at - s.time
                ) / 86400.0
            domain = extract_domain_with_fallback(s.url, is_hn=s.is_hn)
            if domain:
                c = counts.setdefault(domain, [0, 0])
                c[idx] += 1
                if s.feedback_updated_at > 0:
                    days = (now - s.feedback_updated_at) / 86400.0
                    if (
                        domain not in _rank_cache.domain_recency_map
                        or days < _rank_cache.domain_recency_map[domain]
                    ):
                        _rank_cache.domain_recency_map[domain] = max(days, 0.0)

    _rank_cache.domain_trust.update(
        {
            domain: (ups + 1) / (ups + downs + 2)
            for domain, (ups, downs) in counts.items()
        }
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


def stack_similarity_features(
    derived: dict[str, NDArray[np.float32]],
    config: ClassifierConfig,
) -> NDArray[np.float32]:
    enabled = [f for f in config.features if f in SIMILARITY_FEATURES]
    if not enabled:
        n = len(next(iter(derived.values()))) if derived else 0
        return np.zeros((n, 0), dtype=np.float32)
    return np.hstack([derived[f].reshape(-1, 1) for f in enabled]).astype(np.float32)


def _make_feature(
    extract_fn: Callable[[Story, float], float],
    norm_val: float = 1.0,
    log: bool = True,
) -> Callable[[list[Story], float], NDArray[np.float32]]:
    def _feature_fn(stories: list[Story], now: float) -> NDArray[np.float32]:
        vals = np.array([extract_fn(s, now) for s in stories], dtype=np.float32)
        if log:
            vals = np.log1p(vals)
            if norm_val != 1.0:
                vals /= np.log1p(norm_val)
        elif norm_val != 1.0:
            vals /= norm_val
        return vals.reshape(-1, 1)

    return _feature_fn


METADATA_FEATURES["log_points"] = _make_feature(
    lambda s, now: max(float(s.score), 0.0) if s.is_hn else 0.0, norm_val=1000.0
)

METADATA_FEATURES["log_comments"] = _make_feature(
    lambda s, now: max(float(s.comment_count or 0.0), 0.0) if s.is_hn else 0.0,
    norm_val=500.0,
)
METADATA_FEATURES["comment_ratio"] = _make_feature(
    lambda s, now: (
        np.log1p(max(float(s.comment_count or 0.0), 0.0) if s.is_hn else 0.0)
        / (np.log1p(max(float(s.score), 0.0) if s.is_hn else 0.0) + 1.0)
    ),
    log=False,
)
METADATA_FEATURES["title_len"] = _make_feature(
    lambda s, now: float(len(s.title or "")), norm_val=120.0
)
METADATA_FEATURES["text_len"] = _make_feature(
    lambda s, now: float(len(s.text_content or "")), norm_val=5000.0
)
METADATA_FEATURES["is_github"] = _make_feature(
    lambda s, now: 1.0 if s.url and "github.com" in s.url.lower() else 0.0, log=False
)
METADATA_FEATURES["is_pdf"] = _make_feature(
    lambda s, now: 1.0 if s.url and s.url.lower().endswith(".pdf") else 0.0, log=False
)
METADATA_FEATURES["comments_count"] = _make_feature(
    lambda s, now: max(float(s.comment_count or 0.0), 0.0) if s.is_hn else 0.0,
    norm_val=15.0,
)
METADATA_FEATURES["is_hn"] = _make_feature(
    lambda s, now: 1.0 if s.is_hn else 0.0, log=False
)


def _meta_source_trust(stories: list[Story], now: float) -> NDArray[np.float32]:
    trust = getattr(_rank_cache, "domain_trust", {})
    vals: list[float] = []
    for s in stories:
        domain = extract_domain_with_fallback(s.url, is_hn=s.is_hn)
        vals.append(trust.get(domain, 0.5) if domain else 0.5)
    return np.array(vals, dtype=np.float32).reshape(-1, 1)


METADATA_FEATURES["source_trust"] = _meta_source_trust


def _make_telemetry_feature(
    extract_fn: Callable[[Story, dict, dict], float],
    norm_val: float = 1.0,
    log: bool = True,
) -> Callable[[list[Story], float], NDArray[np.float32]]:
    def _feature_fn(stories: list[Story], now: float) -> NDArray[np.float32]:
        from api.telemetry_features import load_telemetry_stats

        story_stats, domain_stats = load_telemetry_stats()

        vals = np.array(
            [extract_fn(s, story_stats, domain_stats) for s in stories],
            dtype=np.float32,
        )
        if log:
            vals = np.log1p(vals)
            if norm_val != 1.0:
                vals /= np.log1p(norm_val)
        elif norm_val != 1.0:
            vals /= norm_val
        return vals.reshape(-1, 1)

    return _feature_fn


METADATA_FEATURES["impression_count"] = _make_telemetry_feature(
    lambda s, ss, ds: float(ss[s.id].impression_count) if s.id in ss else 0.0,
    norm_val=200.0,
)

METADATA_FEATURES["click_count"] = _make_telemetry_feature(
    lambda s, ss, ds: float(ss[s.id].click_count) if s.id in ss else 0.0, norm_val=20.0
)

METADATA_FEATURES["click_ratio"] = _make_telemetry_feature(
    lambda s, ss, ds: float(ss[s.id].click_ratio) if s.id in ss else 0.0, log=False
)

METADATA_FEATURES["days_since_last_impression"] = _make_telemetry_feature(
    lambda s, ss, ds: (
        (30.0 - float(ss[s.id].days_since_last_impression)) / 30.0
        if s.id in ss
        else 0.0
    ),
    log=False,
)


def _domain_ctr_extract(s: Story, ss: dict, ds: dict) -> float:
    from api.telemetry_features import extract_domain_with_fallback

    d = extract_domain_with_fallback(s.url, is_hn=s.is_hn)
    return float(ds[d].domain_ctr) if d and d in ds else 0.0


METADATA_FEATURES["domain_ctr"] = _make_telemetry_feature(
    _domain_ctr_extract, log=False
)


def _domain_imp_extract(s: Story, ss: dict, ds: dict) -> float:
    from api.telemetry_features import extract_domain_with_fallback

    d = extract_domain_with_fallback(s.url, is_hn=s.is_hn)
    return float(ds[d].domain_impression_count) if d and d in ds else 0.0


METADATA_FEATURES["domain_impression_count"] = _make_telemetry_feature(
    _domain_imp_extract, norm_val=500.0
)


def _meta_local_density(stories: list[Story], now: float) -> NDArray[np.float32]:
    """Mean pairwise cosine similarity within the candidate pool.

    High = crowded topic (many similar candidates).
    Low = niche story (sparse region of embedding space).
    Independent of feedback history — works for cold start.
    """
    cache = _rank_cache.local_density
    if cache is not None and len(stories) <= len(cache):
        size = min(len(stories), len(cache))
        return cache[:size].reshape(-1, 1).astype(np.float32)
    return np.zeros((len(stories), 1), dtype=np.float32)


METADATA_FEATURES["local_density"] = _meta_local_density


def _meta_story_age(stories: list[Story], now: float) -> NDArray[np.float32]:
    """log1p(story age in days) at the moment of decision.

    For training samples (story_id in _story_age_at_vote_map): use vote-time age.
    For inference candidates (cache miss): use current age (now - s.time).
    Story.time == 0 returns 0.0 (missing data sentinel).
    Clock skew (age < 0) clamped to 0.
    """
    vals: list[float] = []
    age_map = getattr(_rank_cache, "story_age_at_vote_map", {})
    for s in stories:
        if s.id in age_map:
            age_days = age_map[s.id]
        elif s.time > 0:
            age_days = (now - s.time) / 86400.0
        else:
            vals.append(0.0)
            continue
        if age_days < 0:
            age_days = 0.0
        vals.append(float(np.log1p(age_days)))
    return np.array(vals, dtype=np.float32).reshape(-1, 1)


METADATA_FEATURES["story_age"] = _meta_story_age


def _meta_cluster_size(stories: list[Story], now: float) -> NDArray[np.float32]:
    """log1p(cluster_size) — how many candidates in the same cluster.

    High = topic saturation (many similar stories).
    Low = niche/novel story (its topic appears rarely in the pool).
    Uses _rank_cache.cluster_size populated in rank_stories.
    Cache miss returns 0.0 (log1p(0) ≈ 0).
    """
    cache = _rank_cache.cluster_size
    if cache is not None and len(stories) <= len(cache):
        size = min(len(stories), len(cache))
        vals = np.log1p(cache[:size].astype(np.float64)).astype(np.float32)
        return vals.reshape(-1, 1)
    return np.zeros((len(stories), 1), dtype=np.float32)


METADATA_FEATURES["cluster_size"] = _meta_cluster_size


def _meta_domain_recency(stories: list[Story], now: float) -> NDArray[np.float32]:
    """log1p(days since last user vote on this domain).

    Uses _domain_recency_map populated from FeedbackRecord.
    Stories from domains never voted on get 365-day sentinel.
    HN discussions (no domain) get the same sentinel.
    """
    from api.url_utils import extract_domain

    _SENTINEL_DAYS = 365.0
    vals: list[float] = []
    recency_map = getattr(_rank_cache, "domain_recency_map", {})
    for s in stories:
        domain = extract_domain(s.url)
        if domain and domain in recency_map:
            days = recency_map[domain]
        else:
            days = _SENTINEL_DAYS
        vals.append(float(np.log1p(max(days, 0.0))))
    return np.array(vals, dtype=np.float32).reshape(-1, 1)


METADATA_FEATURES["domain_recency"] = _meta_domain_recency


def _classifier_metadata_features(
    stories: list[Story],
    config: AppConfig,
    now: float,
    expected_len: int,
) -> NDArray[np.float32]:
    enabled = [f for f in config.classifier.features if f in METADATA_FEATURES]
    if not stories or len(stories) != expected_len:
        return np.zeros((expected_len, len(enabled)), dtype=np.float32)
    if not enabled:
        return np.zeros((expected_len, 0), dtype=np.float32)
    return np.hstack([METADATA_FEATURES[f](stories, now) for f in enabled]).astype(
        np.float32
    )


def build_feature_matrix(
    embeddings: NDArray[np.float32],
    derived_rows: NDArray[np.float32],
    metadata_rows: NDArray[np.float32],
    config: AppConfig,
) -> NDArray[np.float32]:
    columns: list[NDArray[np.float32]] = []
    if config.classifier.raw_embedding_features:
        columns.append(embeddings.astype(np.float32))
    if derived_rows.shape[1] > 0:
        columns.append(derived_rows.astype(np.float32))
    if metadata_rows.shape[1] > 0:
        columns.append(metadata_rows.astype(np.float32))
    if not columns:
        return np.zeros((embeddings.shape[0], 0), dtype=np.float32)
    return np.hstack(columns).astype(np.float32)


def _mask_self_similarity(
    sim: NDArray[np.float32],
    n_ref: int,
    offset: int,
) -> None:
    """Set self-similarity entries to -1.0 in a non-square similarity matrix.

    When ``embs`` is ``[pos; neg]`` stacked vertically and ``ref`` is one of
    the two halves, the self-similarity entry for reference item ``j`` lives
    at ``sim[offset + j, j]``, not on the naive diagonal ``sim[j, j]``.

    Args:
        sim: Similarity matrix of shape ``(n_embs, n_ref)``.
        n_ref: Number of reference items (columns).
        offset: Row index in ``embs`` where the reference block starts.
    """
    n_to_mask = min(n_ref, sim.shape[0] - offset)
    if n_to_mask <= 0:
        return
    rows = np.arange(n_to_mask) + offset
    cols = np.arange(n_to_mask)
    sim[rows, cols] = -1.0


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
    """Compute the first-stage derived similarity features used by the classifier.

    When ``exclude_self_pos`` / ``exclude_self_neg`` are set, the caller
    promises that ``embs`` is laid out as ``[pos_ref; neg_ref]`` (vertically
    stacked).  Self-similarity entries are masked at the correct row offsets:
    positive self-entries start at row 0, negative self-entries start at row
    ``len(pos_ref)``.
    """

    sim_c = (
        cosine_similarity(embs, centroid_ref)
        if centroid_ref.shape[0] > 0
        else np.zeros((len(embs), 0), dtype=np.float32)
    )
    f_centroid_max = (
        np.max(sim_c, axis=1)
        if sim_c.shape[1] > 0
        else np.zeros(len(embs), dtype=np.float32)
    )

    sim_p = (
        cosine_similarity(embs, pos_ref)
        if pos_ref.shape[0] > 0
        else np.zeros((len(embs), 0), dtype=np.float32)
    )
    if exclude_self_pos and sim_p.shape[1] > 0:
        _mask_self_similarity(sim_p, pos_ref.shape[0], offset=0)
    f_closest_pos = (
        np.max(sim_p, axis=1)
        if sim_p.shape[1] > 0
        else np.zeros(len(embs), dtype=np.float32)
    )

    sim_n = (
        cosine_similarity(embs, neg_ref)
        if neg_ref.shape[0] > 0
        else np.zeros((len(embs), 0), dtype=np.float32)
    )
    if exclude_self_neg and sim_n.shape[1] > 0:
        _mask_self_similarity(sim_n, neg_ref.shape[0], offset=pos_ref.shape[0])
    f_closest_neg = (
        np.max(sim_n, axis=1)
        if sim_n.shape[1] > 0
        else np.zeros(len(embs), dtype=np.float32)
    )

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

    f_closest_margin = f_closest_pos - f_closest_neg

    # New: ratio of pos_knn to total similarity, in [0, 1]
    f_pos_neg_ratio = f_knn_pos / (f_knn_pos + f_knn_neg + 1e-6)

    # New: embedding L2 norm normalized to mean training norm
    embs_norms = np.linalg.norm(embs, axis=1)
    f_embedding_magnitude = embs_norms / (embs_norms.mean() + 1e-6)

    return {
        "centroid": f_centroid_max,
        "pos_knn": f_knn_pos,
        "neg_knn": f_knn_neg,
        "pos_neg_ratio": f_pos_neg_ratio,
        "closest_pos": f_closest_pos,
        "closest_neg": f_closest_neg,
        "closest_margin": f_closest_margin,
        "embedding_magnitude": f_embedding_magnitude,
    }


type ClusterItem = tuple[StoryDict, float]


# Global singleton for the model
_model: ONNXEmbeddingModel | None = None
_model_init_lock = threading.Lock()

CACHE_DIR: Path = Path(EMBEDDING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
                    max_length=self.spec.max_tokens,
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
                spec.normalize if normalize_embeddings is None else normalize_embeddings
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
    return init_model()


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
    char_lengths: list[int] = []

    for t in texts:
        text = model_spec.prepare_text(t, is_query=is_query) if model_spec else t
        char_lengths.append(len(text))
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
    if char_lengths:
        import statistics

        mean_chars = int(statistics.mean(char_lengths))
        max_chars = max(char_lengths)
        logger.info(
            "Text stats for %d docs: mean=%d chars, max=%d chars, truncated=%d/%d (limit=%d tokens).",
            len(texts),
            mean_chars,
            max_chars,
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
    return get_embeddings(texts, is_query=False, progress_callback=progress_callback)


def cluster_interests_with_labels(
    embeddings: NDArray[np.float32],
    config: ClusteringConfig = ClusteringConfig(),
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Cluster user interest embeddings using KMeans.
    Returns (centroids, labels) where:
      - centroids: shape (n_clusters, embedding_dim)
      - labels: shape (n_samples,) cluster assignment per sample
    """
    from sklearn.cluster import KMeans

    n_samples = len(embeddings)
    if n_samples == 0:
        return embeddings, np.array([], dtype=np.int32)

    if n_samples < config.min_samples_per_cluster * 2:
        labels = np.zeros(n_samples, dtype=np.int32)
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        return centroid.astype(np.float32), labels

    normalized = _normalize_embeddings(embeddings)

    target = round(math.sqrt(n_samples))
    effective_n_clusters = min(
        config.max_clusters,
        max(config.min_clusters, target),
        n_samples,
    )

    labels = (
        KMeans(
            n_clusters=effective_n_clusters,
            n_init=10,
            random_state=0,
        )
        .fit_predict(normalized)
        .astype(np.int32)
    )

    labels = _refine_cluster_assignments(normalized, labels, config.refine_iters)

    max_size = max(
        config.min_samples_per_cluster,
        min(
            config.max_cluster_size,
            int(math.ceil(n_samples * config.max_cluster_fraction)),
        ),
    )
    max_n_clusters = min(
        config.max_clusters, n_samples // max(1, config.min_samples_per_cluster)
    )
    labels = _split_large_clusters(
        embeddings,
        labels,
        min_size=config.min_samples_per_cluster,
        max_size=max_size,
        max_clusters=max_n_clusters,
    )

    centroids = _centroids_from_labels(normalized, labels)
    centroids, labels, _ = split_outlier_clusters(
        embeddings, labels, config.outlier_similarity_threshold
    )
    centroids = _centroids_from_labels(normalized, labels)
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
    model_or_positive_embeddings: OrdinalThresholdModel
    | NDArray[np.float32]
    | None = None,
    positive_embeddings: NDArray[np.float32] | None = None,
    negative_embeddings: NDArray[np.float32] | None = None,
    config: AppConfig = AppConfig(),
    progress_callback: RankProgressCallback | None = None,
    diagnostics: dict[str, object] | None = None,
    positive_stories: list[Story] | None = None,
    negative_stories: list[Story] | None = None,
    cluster_names: dict[int, str] | None = None,
    cluster_keywords: dict[int, str] | None = None,
) -> list[RankResult]:
    """
    Rank candidate stories with the feedback-trained single model.
    """
    _rank_cache.local_density = None
    _rank_cache.cluster_size = None
    now = time.time()
    _populate_rank_cache_metadata(positive_stories, negative_stories, now)

    if not stories:
        return []
    from api.ordinal_model import (
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
            positive_embeddings = cast(
                NDArray[np.float32], model_or_positive_embeddings
            )

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
    )

    # Display-only scores (always populated regardless of scoring path)
    model_scores: NDArray[np.float32] = np.zeros(len(stories), dtype=np.float32)
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

    # Compute local density cache (mean pairwise cosine sim within candidate pool)
    if len(cand_emb) > 1:
        pair_sim = cosine_similarity(cand_emb, cand_emb)
        pair_sim.flat[:: len(pair_sim) + 1] = 0.0  # zero out diagonal (self)
        _rank_cache.local_density = (pair_sim.sum(axis=1) / (len(cand_emb) - 1)).astype(
            np.float32
        )
    elif len(cand_emb) == 1:
        _rank_cache.local_density = np.zeros(1, dtype=np.float32)
    else:
        _rank_cache.local_density = np.zeros(0, dtype=np.float32)

    # Compute cluster size cache (how many candidates in each cluster)
    if len(cand_emb) >= 2 * config.clustering.min_samples_per_cluster:
        _, cand_labels = cluster_interests_with_labels(
            cand_emb, config=config.clustering
        )
        unique_lbl, counts = np.unique(cand_labels, return_counts=True)
        lbl_to_count = dict(zip(unique_lbl, counts))
        _rank_cache.cluster_size = np.array(
            [lbl_to_count[lbl] for lbl in cand_labels], dtype=np.int32
        )
    elif len(cand_emb) == 1:
        _rank_cache.cluster_size = np.array([1], dtype=np.int32)
    else:
        _rank_cache.cluster_size = np.array([], dtype=np.int32)

    cand_derived = compute_classifier_similarity_features(
        cand_emb,
        X_pos,
        X_neg,
        centroids,
        config.classifier,
    )
    cand_metadata = _classifier_metadata_features(
        stories, config, time.time(), len(stories)
    )
    cand_metadata = cand_metadata[: len(cand_emb)]

    def _stack_with_embeddings(
        derived: dict[str, NDArray[np.float32]],
        embeddings: NDArray[np.float32],
        cfg: ClassifierConfig,
    ) -> NDArray[np.float32]:
        cols: list[NDArray[np.float32]] = []
        if cfg.raw_embedding_features:
            cols.append(embeddings)
        dr = stack_similarity_features(derived, cfg)
        if dr.shape[1] > 0:
            cols.append(dr)
        return (
            np.hstack(cols).astype(np.float32)
            if cols
            else np.zeros((len(embeddings), 0), dtype=np.float32)
        )

    cand_columns: list[NDArray[np.float32]] = []
    if config.classifier.raw_embedding_features:
        cand_columns.append(cand_emb)
    derived_rows = stack_similarity_features(cand_derived, config.classifier)
    if derived_rows.shape[1] > 0:
        cand_columns.append(derived_rows)
    if cand_metadata.shape[1] > 0:
        cand_columns.append(cand_metadata)
    cand_features = (
        np.hstack(cand_columns).astype(np.float32)
        if cand_columns
        else np.zeros((len(cand_emb), 0), dtype=np.float32)
    )
    derived_width = int(
        stack_similarity_features(cand_derived, config.classifier).shape[1]
    )
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
            pos_features = _stack_with_embeddings(pos_derived, X_pos, config.classifier)
            neg_features = _stack_with_embeddings(neg_derived, X_neg, config.classifier)
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
            model = train_model_from_matrix(
                train_rows, train_labels, config.single_model
            )
        else:
            model_scores = cluster_max_scores

    if model is None and not np.any(model_scores):
        model_scores = cluster_max_scores

    if model is not None:
        try:
            expected_n = model.at_least_neutral.steps[0][1].n_features_in_
        except AttributeError:
            expected_n = cand_features.shape[1]

        if expected_n != cand_features.shape[1]:
            # Build full single-model features (embeddings + derived + metadata)
            derived_rows = stack_similarity_features(
                cand_derived,
                config.classifier,
            )
            full_features = build_feature_matrix(
                cand_emb,
                derived_rows,
                cand_metadata,
                config,
            )
            model_scores, downvote, neutral, upvote = _predict_ordinal_outputs(
                model, full_features
            )
        else:
            model_scores, downvote, neutral, upvote = _predict_ordinal_outputs(
                model, cand_features
            )
    else:
        downvote = np.zeros(len(stories), dtype=np.float32)
        neutral = np.zeros(len(stories), dtype=np.float32)
        upvote = np.zeros(len(stories), dtype=np.float32)

    report_progress("scoring", 1, 1, "Scored candidates")

    probs = np.stack([downvote, neutral, upvote], axis=1) + 1e-12
    entropy_per_cand = -np.sum(probs * np.log2(probs), axis=1) / np.log2(3)

    # Deterministic tie-breaking: sort by model_score desc, story_id asc
    story_ids = np.array([s.id for s in stories], dtype=np.int64)
    ranked_indices = np.lexsort((story_ids, -model_scores))

    report_progress("finalize", 0, 1, "Finalizing ranked stories")
    results = [
        RankResult(
            index=int(best_idx),
            model_score=float(model_scores[best_idx]),
            best_fav_index=int(best_fav_indices[best_idx]),
            max_sim_score=float(max_sim_scores[best_idx]),
            knn_score=float(raw_knn_scores[best_idx]),
            max_cluster_score=float(cluster_max_scores[best_idx]),
            p_down=float(downvote[best_idx]),
            p_neutral=float(neutral[best_idx]),
            p_up=float(upvote[best_idx]),
            entropy=float(entropy_per_cand[best_idx]),
        )
        for best_idx in ranked_indices
    ]
    report_progress("finalize", 1, 1, "Finalized ranked stories")
    report_progress("complete", 1, 1, "Reranking complete")
    return results
