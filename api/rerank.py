from __future__ import annotations
import hashlib
import json
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
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    MIN_SAMPLES_PER_CLUSTER,
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
        
        providers = ["CPUExecutionProvider"]

        self.session: ort.InferenceSession = ort.InferenceSession(
            f"{model_dir}/model.onnx",
            providers=providers
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


def init_model() -> ONNXEmbeddingModel:
    global _model
    if _model is None:
        _model = ONNXEmbeddingModel()
    return _model


def get_model() -> ONNXEmbeddingModel:
    return init_model()


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


def cluster_interests(
    embeddings: NDArray[np.float32],
    weights: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Cluster user interest embeddings into K centroids.
    Returns centroids array of shape (n_clusters, embedding_dim).
    """
    centroids, _ = cluster_interests_with_labels(embeddings, weights)
    return centroids

def cluster_interests_with_labels(
    embeddings: NDArray[np.float32],
    weights: Optional[NDArray[np.float32]] = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Cluster user interest embeddings using Agglomerative Clustering.
    Uses silhouette score to find optimal cluster count.
    Returns (centroids, labels) where:
      - centroids: shape (n_clusters, embedding_dim)
      - labels: shape (n_samples,) cluster assignment per sample
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

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

    # Normalize embeddings for cosine-like behavior with euclidean distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normalized = embeddings / norms

    # Search for highest k with acceptable silhouette (>= 0.1)
    # This gives more granular clusters while maintaining coherence
    min_k = max(MIN_CLUSTERS, int(np.sqrt(n_samples) * 0.7))
    max_k = min(MAX_CLUSTERS, int(np.sqrt(n_samples) * 2.5), n_samples // MIN_SAMPLES_PER_CLUSTER)

    best_labels: NDArray[np.int32] = np.zeros(n_samples, dtype=np.int32)
    silhouette_threshold = 0.1

    # Search from high to low k, pick first that meets threshold
    for k in range(max_k, min_k - 1, -1):
        agg = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
        )
        labels = agg.fit_predict(normalized)
        score = float(silhouette_score(normalized, labels))
        if score >= silhouette_threshold:
            best_labels = labels.astype(np.int32)
            break
    else:
        # No k met threshold, use max_k anyway
        agg = AgglomerativeClustering(n_clusters=max_k, metric="euclidean", linkage="ward")
        best_labels = agg.fit_predict(normalized).astype(np.int32)

    # Compute centroids from labels (in original embedding space)
    unique_labels = sorted(set(best_labels))
    centroids = []
    for lbl in unique_labels:
        mask = best_labels == lbl
        if weights is not None:
            cluster_weights = weights[mask]
            centroid = np.average(embeddings[mask], axis=0, weights=cluster_weights)
        else:
            centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)

    return np.array(centroids, dtype=np.float32), best_labels

# Global rate limiter state
_last_call_time: float = 0.0
# Conservative rate limit: 1 call per 4 seconds (~15 RPM) to stay safely within free tier limits
_min_interval: float = 4.0

def _rate_limit() -> None:
    """Enforce rate limiting for API calls."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_call_time = time.time()

def generate_single_cluster_name(items: list[tuple[dict[str, Any], float]]) -> str:
    """Generate a name for a single cluster using Gemini API."""
    import os
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Misc"

    # Get top titles by weight
    sorted_items = sorted(items, key=lambda x: -x[1])[:10]
    titles = []
    for story, _ in sorted_items:
        title = str(story.get("title", "")).strip()
        for prefix in ["Show HN:", "Ask HN:", "Tell HN:"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        if title:
            titles.append(title)

    titles_text = "\n".join(f"- {t}" for t in titles)

    if not titles_text:
        return "Misc"

    prompt = f"""
What single topic best describes these titles? Reply with ONLY 1-3 words.

{titles_text}

Topic:"""

    try:
        _rate_limit()
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 20},
        )
        name = response.text
        if not name:
            return "Misc"
        
        name = name.strip().strip('"').strip("'")
        # Truncate if too long
        words = name.split()[:4]
        return " ".join(words) if words else "Misc"
    except Exception:
        return "Misc"

def generate_cluster_names(
    clusters: dict[int, list[tuple[dict[str, Any], float]]],
) -> dict[int, str]:
    """Generate cluster names using Gemini API."""
    if not clusters:
        return {}
    return {cid: generate_single_cluster_name(items) for cid, items in clusters.items()}

TLDR_CACHE_PATH = Path(".cache/tldrs.json")


def _load_tldr_cache() -> dict[str, str]:
    """Load TL;DR cache from disk."""
    if TLDR_CACHE_PATH.exists():
        try:
            return json.loads(TLDR_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}

def _save_tldr_cache(cache: dict[str, str]) -> None:
    """Save TL;DR cache to disk."""
    TLDR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TLDR_CACHE_PATH.write_text(json.dumps(cache))

def generate_story_tldr(story_id: int, title: str, comments: list[str]) -> str:
    """Generate a 1-sentence TL;DR for a story using Gemini API. Cached by story ID."""
    import os
    from google import genai

    if not title:
        return ""

    # Check cache first
    cache = _load_tldr_cache()
    cache_key = str(story_id)
    if cache_key in cache:
        return cache[cache_key]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""

    # Build context from title + top comments
    context = f"Title: {title}"
    if comments:
        context += "\n\nTop comments:\n" + "\n".join(f"- {c[:400]}" for c in comments[:6])

    prompt = f"""
{context}

Provide a concise summary of the story and its discussion. 
Do NOT use introductory phrases like "This story is about", "The discussion reveals", "Here is a summary", or "The article mentions". 
Start directly with the content.

Structure:
- Sentence 1: Core subject or project.
- Sentence 2+: Main technical or philosophical debate and key takeaways from the comments.

IMPORTANT: Put the comments summary (Sentence 2+) on a single new line directly below the first sentence (no empty line)."""

    try:
        _rate_limit()
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 400},
        )
        tldr = response.text
        if not tldr:
            return ""

        tldr = tldr.strip().strip('"').strip("'")
        
        # Aggressive cleaning of conversational filler
        useless_prefixes = [
            "Here is a summary:", "Here is a 2-sentence summary:", "Here is a 3-sentence summary:",
            "This story is about", "The story is about", "This article is about", 
            "The discussion reveals that", "The discussion reveals", 
            "TL;DR:", "TLDR:", "Summary:", "In this story,", "In this article,"
        ]
        
        lower_tldr = tldr.lower()
        for prefix in useless_prefixes:
            if lower_tldr.startswith(prefix.lower()):
                tldr = tldr[len(prefix):].lstrip(":* \n")
                lower_tldr = tldr.lower()

        # Clean up any remaining markdown or list characters
        tldr = tldr.lstrip("*#- ")
        
        # Truncate if too long (extended cutoff)
        if len(tldr) > 800:
            tldr = tldr[:797] + "..."

        # Save to cache
        cache[cache_key] = tldr
        _save_tldr_cache(cache)

        return tldr
    except Exception:
        return ""

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
    diversity_lambda: float = 0.35,  # Literature: 0.3-0.5 for discovery
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[tuple[int, float, int, float]]:
    """
    Returns list of (index, hybrid_score, best_fav_index, max_sim_score).

    Uses multi-interest clustering and MMR diversity reranking.
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
        # 1. Multi-Interest Clustering
        # Cluster positive embeddings into K interest centroids to capture diverse interests
        interest_centroids: NDArray[np.float32] = cluster_interests(
            positive_embeddings, weights=positive_weights
        )

        # 2. Semantic Score using interest centroids
        # For each candidate, find similarity to each interest cluster
        sim_centroids: NDArray[np.float32] = cosine_similarity(
            interest_centroids, cand_emb
        )

        # MaxSim across interest clusters (best matching interest)
        cluster_max_sim = np.max(sim_centroids, axis=0)

        # Mean across all clusters (broad appeal)
        cluster_mean_sim = np.mean(sim_centroids, axis=0)

        # Combined: weight MaxSim higher to preserve niche interests
        semantic_scores = 0.7 * cluster_max_sim + 0.3 * cluster_mean_sim

        # For display score and best_fav_index, use original embeddings (not clusters)
        # This preserves interpretable "match to specific story" display
        sim_pos: NDArray[np.float32] = cosine_similarity(positive_embeddings, cand_emb)
        if positive_weights is not None:
            sim_pos = sim_pos * positive_weights[:, np.newaxis]
        max_sim_scores = np.max(sim_pos, axis=0)
        best_fav_indices = np.argmax(sim_pos, axis=0)

        # Apply threshold based on original MaxSim (more interpretable)
        low_sim_mask = max_sim_scores < SEMANTIC_MATCH_THRESHOLD
        semantic_scores[low_sim_mask] = 0.0
        best_fav_indices[low_sim_mask] = -1
        max_sim_scores[low_sim_mask] = 0.0

    # 3. Negative Signal (Penalty)
    if negative_embeddings is not None and len(negative_embeddings) > 0:
        sim_neg: NDArray[np.float32] = np.max(
            cosine_similarity(negative_embeddings, cand_emb), axis=0
        )
        semantic_scores -= neg_weight * sim_neg

    # 4. HN Gravity Score
    points: NDArray[Any] = np.array([int(s.get("score", 0)) for s in stories])
    hn_scores: NDArray[Any] = np.power(np.maximum(points - 1, 0), HN_SCORE_POINTS_EXP)

    if hn_scores.max() > 0:
        hn_scores /= hn_scores.max()

    # 5. Hybrid Score
    hybrid_scores: NDArray[np.float32] = (
        1 - hn_weight
    ) * semantic_scores + hn_weight * hn_scores

    # 6. Diversity (MMR)
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

REASON_CACHE_PATH = Path(".cache/reasons.json")


def _load_reason_cache() -> dict[str, str]:
    if REASON_CACHE_PATH.exists():
        try:
            return json.loads(REASON_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}

def _save_reason_cache(cache: dict[str, str]) -> None:
    REASON_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REASON_CACHE_PATH.write_text(json.dumps(cache))

def generate_similarity_reason(cand_title: str, cand_comments: list[str], history_title: str) -> str:
    """Generate a short reason why two stories are similar."""
    import os
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""

    # Create a unique key for the pair
    key = hashlib.sha256(f"{cand_title}|{history_title}".encode()).hexdigest()
    
    cache = _load_reason_cache()
    if key in cache:
        return cache[key]

    comments_text = "\\n".join(f"- {c[:200]}" for c in cand_comments[:3])
    
    prompt = f"""
Candidate Story: "{cand_title}"
User's Past Interest: "{history_title}"

Context from Candidate:
{comments_text}

Identify the specific shared technical topic or theme.
Reply with ONE short phrase (max 10 words) starting with a lowercase verb (e.g. "discusses...", "explores...", "relates to...").
"""

    try:
        _rate_limit()
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config={"temperature": 0.1, "max_output_tokens": 30},
        )
        reason = response.text
        if not reason:
            return ""

        reason = reason.strip().strip('"').strip("'")
        
        # Ensure it starts lowercase if it's a verb phrase
        if reason and reason[0].isupper() and " " in reason:
             reason = reason[0].lower() + reason[1:]

        cache[key] = reason
        _save_reason_cache(cache)
        return reason
    except Exception:
        return ""