from __future__ import annotations
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from api.constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MIN_CLIP,
    EMBEDDING_MODEL_VERSION,
    HN_SCORE_POINTS_EXP,
    TEXT_CONTENT_MAX_LENGTH,
    SEMANTIC_MATCH_THRESHOLD,
)

# Global singleton for the model
_model: Optional[ONNXEmbeddingModel] = None

CACHE_DIR: Path = Path(EMBEDDING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ONNXEmbeddingModel:
    def __init__(self, model_dir: str = "onnx_model") -> None:
        self.model_dir: str = model_dir
        if not Path(f"{model_dir}/model.onnx").exists():
            raise FileNotFoundError(
                f"Model not found in {model_dir}. Please run setup_model.py."
            )

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_dir)
        self.session: ort.InferenceSession = ort.InferenceSession(
            f"{model_dir}/model.onnx"
        )
        self.model_id: str = "bge-base-en-v1.5"

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> NDArray[np.float32]:
        all_embeddings: list[NDArray[np.float32]] = []
        total_items: int = len(texts)

        for i in range(0, total_items, batch_size):
            if progress_callback:
                progress_callback(i, total_items)

            batch: list[str] = texts[i : i + batch_size]
            inputs: Any = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            input_names: list[str] = [node.name for node in self.session.get_inputs()]
            ort_inputs: dict[str, Any] = {
                k: v.astype(np.int64) for k, v in inputs.items() if k in input_names
            }

            outputs: Any = self.session.run(None, ort_inputs)
            last_hidden_state: NDArray[np.float32] = cast(
                NDArray[np.float32], outputs[0]
            )

            # Mean Pooling
            attention_mask: NDArray[Any] = inputs["attention_mask"]
            mask_expanded: NDArray[Any] = np.expand_dims(attention_mask, -1).astype(
                float
            )
            sum_embeddings: NDArray[np.float32] = np.sum(
                last_hidden_state * mask_expanded, axis=1
            )
            sum_mask: NDArray[Any] = np.clip(
                mask_expanded.sum(axis=1), a_min=EMBEDDING_MIN_CLIP, a_max=None
            )
            batch_embeddings: NDArray[np.float32] = sum_embeddings / sum_mask

            if normalize_embeddings:
                norm: NDArray[Any] = np.linalg.norm(
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


def get_model() -> ONNXEmbeddingModel:
    global _model
    if _model is None:
        _model = ONNXEmbeddingModel()
    return _model


def get_embeddings(
    texts: list[str],
    is_query: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> NDArray[np.float32]:
    if not texts:
        return np.array([], dtype=np.float32)

    model: ONNXEmbeddingModel = get_model()
    # BGE-style prefix for queries only
    prefix: str = (
        "Represent this sentence for searching relevant passages: " if is_query else ""
    )
    processed_texts: list[str] = [
        f"{prefix}{t[:TEXT_CONTENT_MAX_LENGTH]}" for t in texts
    ]

    vectors: list[Optional[NDArray[np.float32]]] = []
    to_compute_indices: list[int] = []

    expected_dim: int = 768  # e5-base-v2

    for idx, text in enumerate(processed_texts):
        # Include model version in hash to invalidate cache on model change
        h: str = hashlib.sha256(
            f"{EMBEDDING_MODEL_VERSION}:{text}".encode()
        ).hexdigest()
        cache_path: Path = CACHE_DIR / f"{h}.npy"
        if cache_path.exists():
            try:
                vec: NDArray[np.float32] = np.load(cache_path)
                if vec.shape == (expected_dim,):
                    vectors.append(vec)
                else:
                    vectors.append(None)
                    to_compute_indices.append(idx)
            except Exception:
                vectors.append(None)
                to_compute_indices.append(idx)
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
                f"{EMBEDDING_MODEL_VERSION}:{processed_texts[original_idx]}".encode()
            ).hexdigest()
            np.save(CACHE_DIR / f"{h_res}.npy", vec_res)

    if not vectors or all(v is None for v in vectors):
        return np.zeros((0, expected_dim), dtype=np.float32)

    return np.stack([v for v in vectors if v is not None]).astype(np.float32)


def compute_recency_weights(
    timestamps: list[int], decay_rate: Optional[float] = None
) -> NDArray[np.float32]:
    # decay_rate arg is kept for compatibility but we default to the "Long Term" sigmoid
    # if it is explicitly passed as None or a positive value.
    # If 0.0 is passed, we still return uniform weights.

    if decay_rate is not None and decay_rate <= 0:
        return np.ones(len(timestamps), dtype=np.float32)

    now: float = time.time()
    ages_days: NDArray[Any] = (now - np.array(timestamps)) / 86400

    # Sigmoid parameters for "1 day ~= 1 month, 1 year = 0.5"
    k = 0.01
    inflection = 365

    # 1 / (1 + exp(k * (age - inflection)))
    # We clip exponent to avoid overflow, though ages shouldn't be that huge
    exponent = np.clip(k * (ages_days - inflection), -50, 50)
    weights: NDArray[Any] = 1.0 / (1.0 + np.exp(exponent))

    return np.clip(weights, 0.0, 1.0).astype(np.float32)


def rank_stories(
    stories: list[dict[str, Any]],
    positive_embeddings: Optional[NDArray[np.float32]],
    negative_embeddings: Optional[NDArray[np.float32]] = None,
    positive_weights: Optional[NDArray[np.float32]] = None,
    hn_weight: float = 0.05,
    neg_weight: float = 0.6,
    diversity_lambda: float = 0.15,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[tuple[int, float, int, float]]:
    """
    Returns list of (index, hybrid_score, best_fav_index, max_sim_score).
    """
    if not stories:
        return []

    cand_texts: list[str] = [str(s.get("text_content", "")) for s in stories]
    cand_emb: NDArray[np.float32] = get_embeddings(
        cand_texts, progress_callback=progress_callback
    )

    semantic_scores: NDArray[np.float32]
    max_sim_scores: NDArray[np.float32]
    best_fav_indices: NDArray[np.int64]

    if positive_embeddings is None or len(positive_embeddings) == 0:
        # If no positive signals, use HN scores primarily
        semantic_scores = np.zeros(len(stories), dtype=np.float32)
        max_sim_scores = np.zeros(len(stories), dtype=np.float32)
        best_fav_indices = np.full(len(stories), -1, dtype=np.int64)
    else:
        # 1. Semantic Score (Hybrid: MaxSim + Top-K Mean)
        # We combine the single best match (MaxSim) with the average of the top 3 matches (Density).
        # This rewards "Dense Clusters" of interest (consistent upvotes) while not completely
        # killing "One-Off" interests (high specific match).

        sim_pos: NDArray[np.float32] = cosine_similarity(positive_embeddings, cand_emb)
        if positive_weights is not None:
            sim_pos = sim_pos * positive_weights[:, np.newaxis]

        # A. MaxSim (Nearest Neighbor)
        max_sim = np.max(sim_pos, axis=0)
        max_sim_scores = max_sim

        # B. Top-K Mean (Robustness/Density)
        k = 3
        if sim_pos.shape[0] >= k:
            # partition moves the top k elements to the end of the axis
            top_k = np.partition(sim_pos, -k, axis=0)[-k:, :]
            mean_top_k = np.mean(top_k, axis=0)
        else:
            mean_top_k = max_sim

        # Combined Score
        semantic_scores = (max_sim + mean_top_k) / 2.0

        best_fav_indices = np.argmax(sim_pos, axis=0)

        # Apply strict threshold to filter noise
        # If the best match is < THRESHOLD, we treat it as no match (0.0)
        # and set fav_idx to -1 so the UI doesn't show a bogus reason.
        low_sim_mask = semantic_scores < SEMANTIC_MATCH_THRESHOLD
        semantic_scores[low_sim_mask] = 0.0
        best_fav_indices[low_sim_mask] = -1
        # We also zero out the display score if it didn't pass the threshold filter
        # so we don't show "90% match" on something we ranked as 0.0
        max_sim_scores[low_sim_mask] = 0.0

    # 2. Negative Signal (Penalty)
    if negative_embeddings is not None and len(negative_embeddings) > 0:
        sim_neg: NDArray[np.float32] = np.max(
            cosine_similarity(negative_embeddings, cand_emb), axis=0
        )
        semantic_scores -= neg_weight * sim_neg

    # 3. HN Gravity Score (Now just based on points, no recency bias for candidates)
    points: NDArray[Any] = np.array([int(s.get("score", 0)) for s in stories])

    # We removed the hours/time decay denominator.
    hn_scores: NDArray[Any] = np.power(np.maximum(points - 1, 0), HN_SCORE_POINTS_EXP)

    if hn_scores.max() > 0:
        hn_scores /= hn_scores.max()

    # 4. Hybrid Score
    hybrid_scores: NDArray[np.float32] = (
        1 - hn_weight
    ) * semantic_scores + hn_weight * hn_scores

    # 5. Diversity (MMR)
    results: list[tuple[int, float, int, float]] = []
    selected_mask: NDArray[np.bool_] = np.zeros(len(cand_emb), dtype=bool)
    cand_sim: NDArray[np.float32] = cosine_similarity(cand_emb, cand_emb)

    for _ in range(min(len(stories), 100)):  # Rank top 100
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
            (
                best_idx,
                float(hybrid_scores[best_idx]),
                int(best_fav_indices[best_idx]),
                float(max_sim_scores[best_idx]),
            )
        )
        selected_mask[best_idx] = True

    return results
