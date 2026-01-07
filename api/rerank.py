import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

# Suppress transformers backend warning (we use ONNX)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans
from transformers import AutoTokenizer

from api.constants import (
    CLUSTER_DISTANCE_THRESHOLD,
    CLUSTER_MIN_NORM,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MIN_CLIP,
    HN_SCORE_POINTS_EXP,
    HN_SCORE_TIME_EXP,
    HN_SCORE_TIME_OFFSET,
    RECENCY_DECAY_RATE,
    SIMILARITY_MAX,
    SIMILARITY_MIN,
)

# Global singleton
_model: Optional["ONNXEmbeddingModel"] = None

CACHE_DIR = Path(EMBEDDING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ONNXEmbeddingModel:
    tokenizer: Any
    session: ort.InferenceSession
    model_dir: str
    model_id: str

    def __init__(self, model_dir: str = "bge_model") -> None:
        self.model_dir = model_dir
        # Check if model exists, if not, try to setup
        if not Path(f"{model_dir}/model_quantized.onnx").exists():
            print(f"Model not found in {model_dir}. Attempting to run setup...")
            try:
                import subprocess  # noqa: PLC0415
                import sys  # noqa: PLC0415

                # Check if setup_model.py exists in root
                setup_script = Path("setup_model.py")
                if setup_script.exists():
                     subprocess.check_call([sys.executable, str(setup_script)])
                else:
                     print("setup_model.py not found! Run the setup script manually.")
            except Exception as e:
                print(f"Failed to auto-setup model: {e}")
                print("Please run 'uv run setup_model.py' manually.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        # Use the quantized version for speed
        self.session = ort.InferenceSession(f"{model_dir}/model_quantized.onnx")

        # Determine stable model identifier for cache keying
        # Check if model is Nomic or BGE based on tokenizer metadata
        tokenizer_name = self.tokenizer.name_or_path.lower()
        if "nomic" in tokenizer_name or "onnx_model" in model_dir.lower():
            self.model_id = "nomic-embed-text-v1.5"
        elif "bge" in tokenizer_name or "bge_model" in model_dir.lower():
            self.model_id = "bge-small-en-v1.5"
        else:
            # Fallback: use model directory name + vocab size as identifier
            vocab_size = len(self.tokenizer)
            self.model_id = f"{model_dir}-{vocab_size}"

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> NDArray[np.float32]:
        all_embeddings = []

        total_items = len(texts)
        for i in range(0, total_items, batch_size):
            batch = texts[i : i + batch_size]
            # ... (tokenization and inference)
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="np",
            )

            # ONNX Inference
            input_names = [node.name for node in self.session.get_inputs()]
            ort_inputs = {
                k: v.astype(np.int64) for k, v in inputs.items() if k in input_names
            }

            outputs = self.session.run(None, ort_inputs)
            last_hidden_state = outputs[0]

            # Mean Pooling (Attention Mask)
            attention_mask = inputs["attention_mask"]
            mask_expanded = np.expand_dims(attention_mask, -1).astype(float)

            sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
            sum_mask = np.clip(
                mask_expanded.sum(axis=1), a_min=EMBEDDING_MIN_CLIP, a_max=None
            )

            batch_embeddings = sum_embeddings / sum_mask

            if normalize_embeddings:
                norm = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.clip(
                    norm, a_min=EMBEDDING_MIN_CLIP, a_max=None
                )

            all_embeddings.append(batch_embeddings)

            if progress_callback:
                progress_callback(min(i + batch_size, total_items), total_items)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


def init_model(model_name: str = "bge_model") -> ONNXEmbeddingModel:
    global _model  # noqa: PLW0603
    if _model is None:
        _model = ONNXEmbeddingModel(model_name)
    return _model


def get_model() -> ONNXEmbeddingModel:
    if _model is None:
        return init_model()
    return _model


def get_cache_key(text: str, model_name: str) -> Path:
    content = f"{model_name}:{text}"
    hash_digest = hashlib.sha256(content.encode()).hexdigest()
    return CACHE_DIR / f"{hash_digest}.npy"


def get_embeddings(
    texts: list[str],
    is_query: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """
    Generate embeddings for texts with model-appropriate prefixes.

    Uses cached embeddings when available. Cache keys include model_id
    to prevent cross-contamination when switching models.

    Args:
        texts: List of text strings to embed
        is_query: If True, use query prefix (vs. document prefix)
        progress_callback: Callback(current, total) for embedding progress

    Returns:
        Array of embeddings, shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])

    model = get_model()
    model_id = model.model_id

    # Determine prefix based on model type
    if "nomic" in model_id.lower():
        prefix = "search_query: " if is_query else "search_document: "
    elif "bge" in model_id.lower():
        # BGE uses asymmetric prefixes (query-only)
        bge_query_prefix = "Represent this sentence for searching relevant passages: "
        prefix = bge_query_prefix if is_query else ""
    else:
        # Unknown model, no prefix
        prefix = ""

    processed_texts = [f"{prefix}{t}" for t in texts]

    vectors = []
    indices_to_compute = []

    for idx, text in enumerate(processed_texts):
        cache_path = get_cache_key(text, model_id)
        if cache_path.exists():
            try:
                vectors.append(np.load(cache_path))
            except Exception:
                vectors.append(None)
                indices_to_compute.append(idx)
        else:
            vectors.append(None)
            indices_to_compute.append(idx)

    if indices_to_compute:
        texts_to_compute = [processed_texts[i] for i in indices_to_compute]
        computed_vectors = model.encode(
            texts_to_compute,
            normalize_embeddings=True,
            progress_callback=progress_callback,
        )

        for i, original_idx in enumerate(indices_to_compute):
            vec = computed_vectors[i]
            vectors[original_idx] = vec
            cache_path = get_cache_key(processed_texts[original_idx], model_id)
            np.save(cache_path, vec)

    return np.array(vectors)


def cluster_and_reduce_auto(
    embeddings: NDArray[np.float32],
) -> tuple[NDArray[np.float32], list[int], list[int]]:
    """
    Cluster embeddings using Agglomerative Clustering (Auto K).
    """
    min_cluster_size = 2
    if len(embeddings) < min_cluster_size:
        return embeddings, list(range(len(embeddings))), list(range(len(embeddings)))

    # distance_threshold with cosine: items with similarity above threshold can group.
    # For cosine: distance = 1 - similarity, so threshold 0.8 = similarity > 0.2
    # linkage='complete' ensures all pairs in cluster meet threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="complete",
    ).fit(embeddings)

    labels = clustering.labels_
    unique_labels = np.unique(labels)

    centroids = []
    representative_indices = []

    for c_idx in unique_labels:
        member_indices = np.where(labels == c_idx)[0]
        cluster_embeds = embeddings[member_indices]

        centroid = np.mean(cluster_embeds, axis=0)
        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > CLUSTER_MIN_NORM:
            centroid = centroid / norm
        centroids.append(centroid)

        if len(member_indices) == 1:
            representative_indices.append(member_indices[0])
        else:
            # Replaced cosine_similarity with dot product as embeddings are normalized
            sims = (centroid.reshape(1, -1) @ cluster_embeds.T)[0]
            best_local_idx = np.argmax(sims)
            representative_indices.append(member_indices[best_local_idx])

    return np.array(centroids), representative_indices, labels.tolist()


def cluster_and_reduce(
    embeddings: NDArray[np.float32], k: int
) -> tuple[NDArray[np.float32], list[int], list[int]]:
    if len(embeddings) <= k:
        return embeddings, list(range(len(embeddings))), list(range(len(embeddings)))

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    # Normalize centroids for dot product similarity
    # KMeans centroids are means, so they are not normalized.
    centroids_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(centroids_norm, a_min=1e-9, a_max=None)

    # Replaced cosine_similarity with dot product
    # centroids: (K, D), embeddings: (N, D) -> (K, N)
    sim_matrix = centroids @ embeddings.T
    representative_indices = np.argmax(sim_matrix, axis=1).tolist()

    labels = kmeans.labels_
    if labels is None:
        return centroids, representative_indices, []
    return centroids, representative_indices, labels.tolist()


def rank_embeddings_maxsim(
    candidate_embeddings: NDArray[np.float32], fav_embeddings: NDArray[np.float32]
) -> list[tuple[int, float, int]]:
    if len(candidate_embeddings) == 0 or len(fav_embeddings) == 0:
        return []

    # Replaced cosine_similarity with dot product (assumes normalized inputs)
    sim_matrix = fav_embeddings @ candidate_embeddings.T
    sim_matrix = np.clip(sim_matrix, SIMILARITY_MIN, SIMILARITY_MAX)
    max_scores = np.max(sim_matrix, axis=0)
    best_fav_indices = np.argmax(sim_matrix, axis=0)
    results = []
    pairs = zip(max_scores, best_fav_indices, strict=True)
    for idx, (score, fav_idx) in enumerate(pairs):
        results.append((idx, float(score), int(fav_idx)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def rank_mmr(
    cand_embeddings: NDArray[np.float32],
    fav_embeddings: NDArray[np.float32],
    diversity_penalty: float,
) -> list[tuple[int, float, int]]:
    """
    Maximal Marginal Relevance ranking with vectorized redundancy computation.

    Balances relevance to favorites vs. diversity among selected candidates.
    Time complexity: O(n*m + n^2) where n=candidates, m=favorites.
    """
    if len(cand_embeddings) == 0 or len(fav_embeddings) == 0:
        return []

    # Compute relevance scores (similarity to favorites)
    # Replaced cosine_similarity with dot product
    sim_matrix = fav_embeddings @ cand_embeddings.T
    sim_matrix = np.clip(sim_matrix, SIMILARITY_MIN, SIMILARITY_MAX)
    relevance_scores = np.max(sim_matrix, axis=0)
    best_fav_indices = np.argmax(sim_matrix, axis=0)

    # Precompute candidate-to-candidate similarities (once, upfront)
    # Replaced cosine_similarity with dot product
    cand_sim = cand_embeddings @ cand_embeddings.T

    results = []
    selected_mask = np.zeros(len(cand_embeddings), dtype=bool)

    for _ in range(len(cand_embeddings)):
        # Compute MMR scores for all unselected candidates
        mmr_scores = np.full(len(cand_embeddings), -np.inf)

        unselected_indices = np.where(~selected_mask)[0]
        if len(unselected_indices) == 0:
            break

        if np.any(selected_mask):
            # Vectorized redundancy: max similarity to any selected candidate
            redundancy = np.max(cand_sim[unselected_indices][:, selected_mask], axis=1)
        else:
            redundancy = np.zeros(len(unselected_indices))

        # MMR score = relevance - lambda * redundancy
        mmr_scores[unselected_indices] = (
            relevance_scores[unselected_indices] - diversity_penalty * redundancy
        )

        # Select best candidate
        best_cand_idx = np.argmax(mmr_scores)
        if mmr_scores[best_cand_idx] == -np.inf:
            break

        results.append(
            (
                best_cand_idx,
                float(relevance_scores[best_cand_idx]),
                int(best_fav_indices[best_cand_idx]),
            )
        )
        selected_mask[best_cand_idx] = True

    return results


def rank_candidates(
    candidates: list[str],
    favorites: list[str],
    diversity_lambda: float = 0.0,
    cluster_k: int = 0,
) -> list[tuple[int, float, int]]:
    if not candidates or not favorites:
        return []
    fav_embeddings = get_embeddings(favorites, is_query=True)
    mapping_indices = list(range(len(favorites)))
    if cluster_k > 0 and len(favorites) > cluster_k:
        fav_embeddings, mapping_indices, _ = cluster_and_reduce(
            fav_embeddings, cluster_k
        )
    elif cluster_k == -1 and len(favorites) > 1:
        fav_embeddings, mapping_indices, _ = cluster_and_reduce_auto(fav_embeddings)
    cand_embeddings = get_embeddings(candidates, is_query=False)
    if diversity_lambda > 0.0:
        results = rank_mmr(
            cand_embeddings, fav_embeddings, diversity_penalty=diversity_lambda
        )
    else:
        results = rank_embeddings_maxsim(cand_embeddings, fav_embeddings)
    final_results = []
    for cand_idx, score, cluster_idx in results:
        real_fav_idx = mapping_indices[cluster_idx]
        final_results.append((cand_idx, score, real_fav_idx))
    return final_results


def calculate_hn_score(
    points: int, time_ts: int, current_time: float | None = None
) -> float:
    """
    Calculate HN score with gravity decay.
    Score = (P - 1)^POINTS_EXP / (T + TIME_OFFSET)^TIME_EXP
    """
    if current_time is None:
        current_time = time.time()

    if time_ts > current_time:
        hours_age = 0
    else:
        hours_age = (current_time - time_ts) / 3600

    numerator = (points - 1) ** HN_SCORE_POINTS_EXP if points > 1 else 0
    denominator = (hours_age + HN_SCORE_TIME_OFFSET) ** HN_SCORE_TIME_EXP

    return numerator / denominator


def compute_recency_weights(
    story_timestamps: list[int] | list[float],
    decay_rate: float = RECENCY_DECAY_RATE,
    current_time: float | None = None,
) -> NDArray[np.float32]:
    """
    Compute exponential recency weights for stories.

    Applies time-based decay to downweight older stories. More recent stories
    receive higher weights, making them more influential in semantic ranking.

    Formula: weight = exp(-decay_rate * age_in_days)

    Args:
        story_timestamps: Unix timestamps of stories (seconds since epoch)
        decay_rate: Exponential decay rate per day, default from constants
            - 0.01 (default): slow decay, ~90% weight at 10 days
            - 0.05: medium decay, ~60% weight at 10 days
            - 0.10: fast decay, ~37% weight at 10 days
        current_time: Reference time (defaults to now)

    Returns:
        Array of weights in [0, 1], shape (len(story_timestamps),)

    Example:
        >>> timestamps = [time.time(), time.time() - 86400*10]  # now, 10 days ago
        >>> weights = compute_recency_weights(timestamps)
        >>> weights
        array([1.0, 0.905], dtype=float32)  # Recent story gets full weight
    """
    if current_time is None:
        current_time = time.time()

    timestamps = np.array(story_timestamps)
    age_seconds = current_time - timestamps
    age_days = age_seconds / 86400  # Convert to days

    # Exponential decay: exp(-decay_rate * age)
    weights = np.exp(-decay_rate * age_days)

    # Clip to [0, 1] in case of future timestamps
    weights = np.clip(weights, 0.0, 1.0)

    return weights.astype(np.float32)


def rank_stories(  # noqa: PLR0912, PLR0913, PLR0915
    stories: list[dict],
    cand_embeddings: NDArray[np.float32] | None = None,
    positive_embeddings: NDArray[np.float32] | None = None,
    negative_embeddings: NDArray[np.float32] | None = None,
    positive_weights: NDArray[np.float32] | None = None,
    diversity_lambda: float = 0.0,
    hn_weight: float = 0.25,
    neg_weight: float = 0.6,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[tuple[int, float, int]]:
    """
    Rank stories using hybrid semantic + temporal scoring.

    Hybrid Score = (1 - hn_weight) * Semantic + hn_weight * HN_Score

    Where:
    - Semantic = WeightedMaxSim(Positive) - neg_weight * MaxSim(Negative)
    - HN_Score = Gravity-decayed score normalized to [0, 1]
    - Gravity formula: (P - 1)^0.8 / (T + 2)^1.8

    Args:
        stories: List of story dicts with keys: id, title, score, time, text_content
        cand_embeddings: Precomputed candidate embeddings (computed if None)
        positive_embeddings: Embeddings of user's favorited/upvoted stories
        negative_embeddings: Embeddings of user's hidden stories
        positive_weights: Recency weights for positive stories
            - If provided, applies element-wise multiplication to similarity matrix
            - Higher weights = more influence on ranking
            - Typical formula: exp(-decay * age_in_days) for exponential recency
            - Example: [1.0, 0.95, 0.90, ...] for recent to older favorites
        diversity_lambda: MMR diversity penalty in [0, 1], 0 = no diversity
        hn_weight: Weight for HN gravity score in [0, 1], default 0.15
        neg_weight: Penalty multiplier for negative stories, default 0.5
        progress_callback: Callback(current, total) for ranking progress

    Returns:
        List of (story_idx, hybrid_score, best_match_fav_idx) tuples, sorted descending
    """
    if not stories:
        return []

    # 1. Embeddings
    if cand_embeddings is None:
        cand_texts = [s.get("text_content", s.get("title", "")) for s in stories]
        cand_embeddings = get_embeddings(
            cand_texts, is_query=False, progress_callback=progress_callback
        )

    if len(cand_embeddings) == 0:
        return []

    # 2. Semantic Scores
    semantic_scores = np.zeros(len(cand_embeddings))
    best_fav_indices = np.zeros(len(cand_embeddings), dtype=int) - 1

    if positive_embeddings is not None and len(positive_embeddings) > 0:
        # Replaced cosine_similarity with dot product
        sim_pos = np.clip(
            positive_embeddings @ cand_embeddings.T,
            SIMILARITY_MIN,
            SIMILARITY_MAX,
        )

        if positive_weights is not None:
            # Apply recency weights to similarities before taking max
            weighted_sim = sim_pos * positive_weights[:, np.newaxis]
            pos_scores = np.max(weighted_sim, axis=0)
            best_fav_indices = np.argmax(weighted_sim, axis=0)
        else:
            pos_scores = np.max(sim_pos, axis=0)
            best_fav_indices = np.argmax(sim_pos, axis=0)

        semantic_scores += pos_scores

    if negative_embeddings is not None and len(negative_embeddings) > 0:
        # Replaced cosine_similarity with dot product
        sim_neg = negative_embeddings @ cand_embeddings.T
        neg_scores = np.max(sim_neg, axis=0)
        semantic_scores -= neg_weight * neg_scores

    # 3. HN Scores (Decay)
    now = time.time()

    scores = np.array([s.get("score", 0) for s in stories])
    times = np.array([s.get("time", now) for s in stories])

    hours_age = (now - times) / 3600
    hours_age = np.maximum(hours_age, 0)

    # Vectorized calculate_hn_score logic
    # Score = (P - 1)^POINTS_EXP / (T + TIME_OFFSET)^TIME_EXP
    numerator = np.power(np.maximum(scores - 1, 0), HN_SCORE_POINTS_EXP)
    denominator = np.power(hours_age + HN_SCORE_TIME_OFFSET, HN_SCORE_TIME_EXP)

    hn_scores = numerator / denominator

    # Normalize HN scores to 0-1 range
    if hn_scores.max() > 0:
        hn_scores = hn_scores / hn_scores.max()

    # 4. Hybrid Score
    hybrid_scores = (1 - hn_weight) * semantic_scores + (hn_weight * hn_scores)

    # 5. Ranking (MaxSim or MMR)
    results = []
    if diversity_lambda > 0:
        # Use optimized vectorized MMR implementation
        # rank_mmr returns (idx, relevance_score, best_fav_idx)
        # We need to inject our hybrid_scores as the relevance signal
        
        # Compute candidate-to-candidate similarities (once, upfront)
        # Replaced cosine_similarity with dot product
        cand_sim = cand_embeddings @ cand_embeddings.T

        selected_mask = np.zeros(len(cand_embeddings), dtype=bool)

        for _ in range(len(cand_embeddings)):
            mmr_scores = np.full(len(cand_embeddings), -np.inf)
            unselected_indices = np.where(~selected_mask)[0]
            if len(unselected_indices) == 0:
                break

            if np.any(selected_mask):
                redundancy = np.max(cand_sim[unselected_indices][:, selected_mask], axis=1)
            else:
                redundancy = np.zeros(len(unselected_indices))

            # MMR score = hybrid_relevance - lambda * redundancy
            mmr_scores[unselected_indices] = (
                hybrid_scores[unselected_indices] - diversity_lambda * redundancy
            )

            best_cand_idx = np.argmax(mmr_scores)
            if mmr_scores[best_cand_idx] == -np.inf:
                break

            results.append(
                (
                    best_cand_idx,
                    float(hybrid_scores[best_cand_idx]),
                    int(best_fav_indices[best_cand_idx]),
                )
            )
            selected_mask[best_cand_idx] = True
    else:
        # Simple Sort
        temp_list = []
        for i in range(len(stories)):
            temp_list.append(
                {
                    "idx": i,
                    "score": hybrid_scores[i],
                    "time": stories[i].get("time", 0),
                    "id": stories[i].get("id", 0),
                    "fav_idx": best_fav_indices[i],
                }
            )

        temp_list.sort(key=lambda x: (x["score"], x["time"], x["id"]), reverse=True)

        for item in temp_list:
            results.append((item["idx"], float(item["score"]), int(item["fav_idx"])))

    return results
