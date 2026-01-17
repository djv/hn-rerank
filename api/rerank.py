from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, cast

import httpx
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from api.constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MIN_CLIP,
    EMBEDDING_MODEL_VERSION,
    HN_SCORE_NORMALIZATION_CAP,
    KNN_NEIGHBORS,
    LLM_CLUSTER_BATCH_SIZE,
    LLM_CLUSTER_NAME_MAX_WORDS,
    LLM_CLUSTER_NAME_MODEL,
    LLM_HTTP_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_TLDR_BATCH_SIZE,
    LLM_TLDR_MAX_TOKENS,
    LLM_TLDR_MODEL,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    MIN_SAMPLES_PER_CLUSTER,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_DIVERSITY_LAMBDA_CLASSIFIER,
    RANKING_HN_WEIGHT,
    RANKING_MAX_RESULTS,
    RANKING_NEGATIVE_WEIGHT,
    RATE_LIMIT_429_BACKOFF_BASE,
    RATE_LIMIT_ERROR_BACKOFF_BASE,
    RATE_LIMIT_JITTER_MAX,
    RATE_LIMIT_MAX_TOKENS,
    RATE_LIMIT_REFILL_RATE,
    SEMANTIC_SIGMOID_K,
    SEMANTIC_SIGMOID_THRESHOLD,
    TEXT_CONTENT_MAX_LENGTH,
)
from api.models import RankResult, Story

logger = logging.getLogger(__name__)


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
            f"{model_dir}/model.onnx", providers=providers
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

    expected_dim: int = 768  # bge-base-en-v1.5

    for idx, text in enumerate(processed_texts):
        # Include model version in hash to invalidate cache on model change
        h: str = hashlib.sha256(
            f"{EMBEDDING_MODEL_VERSION}:{text}".encode()
        ).hexdigest()
        cache_path_npz: Path = CACHE_DIR / f"{h}.npz"
        cache_path_npy: Path = CACHE_DIR / f"{h}.npy"

        vec: Optional[NDArray[np.float32]] = None
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
                f"{EMBEDDING_MODEL_VERSION}:{processed_texts[original_idx]}".encode()
            ).hexdigest()
            # Use compressed format for cache efficiency
            np.savez_compressed(CACHE_DIR / f"{h_res}.npz", embedding=vec_res)

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

    # Search for optimal k with best silhouette score
    # Higher multipliers = more granular clusters (better topic separation)
    min_k = max(MIN_CLUSTERS, int(np.sqrt(n_samples) * 1.2))
    max_k = min(
        MAX_CLUSTERS,
        int(np.sqrt(n_samples) * 3.5),
        n_samples // MIN_SAMPLES_PER_CLUSTER,
    )

    best_labels: NDArray[np.int32] = np.zeros(n_samples, dtype=np.int32)
    best_score = -1.0
    best_k = min_k

    # Search all k values, pick the one with highest silhouette score
    # Bias toward more clusters: only pick fewer if significantly better (> 0.03)
    for k in range(min_k, max_k + 1):
        agg = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        labels = agg.fit_predict(normalized)
        score = float(silhouette_score(normalized, labels))
        # Prefer more clusters unless fewer is significantly better
        if score > best_score + 0.03 or (score >= best_score - 0.01 and k > best_k):
            best_score = score
            best_k = k
            best_labels = labels.astype(np.int32)

    # Renumber labels to be consecutive (0, 1, 2...)
    # Note: We no longer force-merge tiny clusters - let them exist naturally
    # Merging can leave gaps (e.g., 0, 2, 3 if 1 was merged)
    unique_final = sorted(set(best_labels))
    mapping = {old: new for new, old in enumerate(unique_final)}
    # Use vectorized mapping
    # Create a lookup array for fast mapping
    if unique_final:
        max_label = max(unique_final)
        lookup = np.zeros(max_label + 1, dtype=np.int32)
        for old, new in mapping.items():
            lookup[old] = new
        best_labels = lookup[best_labels]

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
_token_bucket: float = RATE_LIMIT_MAX_TOKENS
_last_refill: float = time.time()


async def _rate_limit() -> None:
    """Enforce rate limiting using a Token Bucket algorithm."""
    global _token_bucket, _last_refill
    import random

    while True:
        now = time.time()
        # Refill tokens based on time passed
        elapsed = now - _last_refill
        _token_bucket = min(
            RATE_LIMIT_MAX_TOKENS, _token_bucket + elapsed * RATE_LIMIT_REFILL_RATE
        )
        _last_refill = now

        if _token_bucket >= 1.0:
            _token_bucket -= 1.0
            return

        # Calculate wait time needed for next token
        wait_time = (1.0 - _token_bucket) / RATE_LIMIT_REFILL_RATE
        # Add a small buffer and jitter to prevent thundering herds
        await asyncio.sleep(wait_time + 0.1 + random.uniform(0, RATE_LIMIT_JITTER_MAX))


CLUSTER_NAME_CACHE_PATH = Path(".cache/cluster_names.json")


def _load_cluster_name_cache() -> dict[str, str]:
    if CLUSTER_NAME_CACHE_PATH.exists():
        try:
            return json.loads(CLUSTER_NAME_CACHE_PATH.read_text())
        except Exception as e:
            logger.warning(f"Failed to load cluster name cache: {e}")
    return {}


def _save_cluster_name_cache(cache: dict[str, str]) -> None:
    CLUSTER_NAME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTER_NAME_CACHE_PATH.write_text(json.dumps(cache))


def _safe_json_loads(text: str) -> dict[Any, Any]:
    """Safely load JSON, handling potential markdown blocks."""
    if not text:
        return {}

    clean_text = text.strip()
    if clean_text.startswith("```"):
        # Extract content between triple backticks
        lines = clean_text.split("\n")
        # Find first line with { or [ after the first ```
        start_idx = 1
        while start_idx < len(lines) and not (
            lines[start_idx].strip().startswith("{}")
            or lines[start_idx].strip().startswith("[")
        ):
            start_idx += 1

        # Find the last line with } or ] before the last ```
        end_idx = len(lines) - 1
        while end_idx > start_idx and not (
            lines[end_idx].strip().endswith("}") or lines[end_idx].strip().endswith("]")
        ):
            end_idx -= 1

        if end_idx >= start_idx:
            clean_text = "\n".join(lines[start_idx : end_idx + 1])

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode failed, trying regex fallback: {e}")
        # Fallback: try to find anything between { and }
        import re

        match = re.search(r"({.*})", clean_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception as e2:
                logger.warning(f"JSON regex fallback also failed: {e2}")
        return {}


async def _generate_with_retry(
    model: str = LLM_TLDR_MODEL,
    contents: Optional[Any] = None,
    config: Optional[dict[str, Any]] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Groq API with exponential backoff retry logic using httpx."""
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, skipping LLM call")
        return None

    # Handle Gemini-style 'contents' to OpenAI-style 'messages'
    messages = []
    if isinstance(contents, str):
        messages = [{"role": "user", "content": contents}]
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict) and "parts" in item:
                text = "".join([p.get("text", "") for p in item["parts"]])
                messages.append({"role": "user", "content": text})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": config.get("temperature", LLM_TEMPERATURE)
        if config
        else LLM_TEMPERATURE,
    }

    if config and config.get("response_mime_type") == "application/json":
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                await _rate_limit()
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=LLM_HTTP_TIMEOUT,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]

                if resp.status_code == 429:
                    logger.warning(
                        f"Groq rate limit hit (429). Attempt {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(RATE_LIMIT_429_BACKOFF_BASE * (2**attempt))
                    continue

                logger.error(f"Groq API error {resp.status_code}: {resp.text}")
                return None

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Groq API call failed after {max_retries} retries: {e}"
                    )
                    return None
                delay = RATE_LIMIT_ERROR_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"Groq API attempt {attempt + 1} failed: {e}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
    return None


async def generate_batch_cluster_names(
    clusters: dict[int, list[tuple[dict[str, Any], float]]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[int, str]:
    """Generate names for multiple clusters in a single API call to save quota."""
    if not clusters:
        return {}

    cache = _load_cluster_name_cache()
    results: dict[int, str] = {}
    to_generate: dict[int, list[tuple[dict[str, Any], float]]] = {}

    for cid, items in clusters.items():
        # Generate cache key based on sorted story IDs
        story_ids = sorted([str(s.get("id", s.get("objectID", ""))) for s, _ in items])
        cache_key = hashlib.sha256(",".join(story_ids).encode()).hexdigest()

        cached_val = cache.get(cache_key)
        if cached_val and cached_val != "Misc" and len(cached_val.strip()) > 0:
            results[cid] = cached_val
        else:
            to_generate[cid] = items

    if not to_generate:
        if progress_callback:
            progress_callback(len(clusters), len(clusters))
        return {cid: results.get(cid, "Misc") for cid in clusters}

    cid_list = list(to_generate.keys())

    for i in range(0, len(cid_list), LLM_CLUSTER_BATCH_SIZE):
        batch_cids = cid_list[i : i + LLM_CLUSTER_BATCH_SIZE]
        batch_prompts = []

        for cid in batch_cids:
            items = to_generate[cid]
            # Use top 5 items for context (titles + comments)
            sorted_items = sorted(items, key=lambda x: -x[1])[:5]

            cluster_data = []
            for s, _ in sorted_items:
                title = str(s.get("title", "")).strip()
                if not title:
                    continue

                # Get top 2 comments, truncate to 150 chars
                comments = s.get("comments", [])
                context = ""
                if comments:
                    snippets = [c[:150].replace("\n", " ") for c in comments[:2]]
                    context = "; ".join(snippets)

                cluster_data.append({"title": title, "context": context})

            # Serialize to JSON for cleaner prompt structure
            batch_prompts.append(f"Cluster {cid}: {json.dumps(cluster_data)}")

        full_prompt = f"""
Name each cluster with a 1-4 word COMMON theme.

Rules:
- Name MUST fit EVERY story in the group, not just the top one
- If stories share a broader category (bikes, transit, maps → "Urban Mobility"), use that
- Only list subtopics if truly unrelated: "Space, Bio & AI"
- Avoid vague terms: "Technology", "News", "Interesting", "Analysis"
- Good: "Distributed Systems", "Urban Transit", "Career Advice"

Return JSON where keys are cluster IDs: {{ "0": "Theme", "1": "Theme" }}

Groups:
{chr(10).join(batch_prompts)}

JSON:"""

        try:
            text = await _generate_with_retry(
                model=LLM_CLUSTER_NAME_MODEL,
                contents=full_prompt,
                config={
                    "temperature": LLM_TEMPERATURE,
                    "response_mime_type": "application/json",
                },
            )

            if text:
                batch_results = _safe_json_loads(text)
                for cid_str, name in batch_results.items():
                    try:
                        cid = int(cid_str)
                        # Truncate to max words (allows multi-topic names)
                        final_name = (
                            str(name).strip().split()[:LLM_CLUSTER_NAME_MAX_WORDS]
                        )
                        final_name = " ".join(final_name)
                        # Strip trailing conjunctions/punctuation left by truncation
                        final_name = final_name.rstrip(" ,&").rstrip()
                        if final_name.endswith(" and") or final_name.endswith(" or"):
                            final_name = final_name.rsplit(" ", 1)[0]

                        results[cid] = final_name
                        # Save to cache
                        items = to_generate[cid]
                        story_ids = sorted(
                            [str(s.get("id", s.get("objectID", ""))) for s, _ in items]
                        )
                        cache_key = hashlib.sha256(
                            ",".join(story_ids).encode()
                        ).hexdigest()
                        cache[cache_key] = final_name
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Failed to parse cluster name for {cid_str}: {e}")
                        continue

            if progress_callback:
                progress_callback(len(results), len(clusters))
        except Exception as e:
            logger.warning(f"Cluster naming batch failed: {e}")

    _save_cluster_name_cache(cache)

    # Final results with better fallback than "Misc"
    final_results = {}
    for cid, items in clusters.items():
        if cid in results:
            final_results[cid] = results[cid]
        else:
            # Fallback to the top story title if LLM naming failed
            sorted_items = sorted(items, key=lambda x: -x[1])
            fallback_name = "Misc"
            if sorted_items:
                top_title = str(sorted_items[0][0].get("title", "")).strip()
                # Clean up HN prefixes
                for prefix in ["Show HN:", "Ask HN:", "Tell HN:"]:
                    if top_title.startswith(prefix):
                        top_title = top_title[len(prefix) :].strip()

                if top_title:
                    fallback_name = (
                        (top_title[:30] + "...") if len(top_title) > 33 else top_title
                    )

            final_results[cid] = fallback_name

    return final_results


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
    TLDR_CACHE_PATH.write_text(json.dumps(cache))


async def generate_batch_tldrs(
    stories: list[dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[int, str]:
    """Generate TL;DRs for multiple stories in batches to save API quota."""
    if not stories:
        return {}

    cache = _load_tldr_cache()
    results: dict[int, str] = {}
    to_generate: list[dict[str, Any]] = []

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
        batch = to_generate[i : i + LLM_TLDR_BATCH_SIZE]

        stories_formatted = []
        for s in batch:
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
                        tldr_clean = tldr.strip().strip('"').strip("'")
                        results[sid] = tldr_clean
                        cache[str(sid)] = tldr_clean
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Failed to parse TLDR for {sid_str}: {e}")
                        continue

            if progress_callback:
                progress_callback(completed_initial + i + len(batch), len(stories))

        except Exception as e:
            logger.warning(f"TLDR batch generation failed: {e}")

    _save_tldr_cache(cache)
    return {
        int(s["id"]): results.get(int(s["id"]), cache.get(str(s["id"]), ""))
        for s in stories
    }


def rank_stories(
    stories: list[Story],
    positive_embeddings: Optional[NDArray[np.float32]],
    negative_embeddings: Optional[NDArray[np.float32]] = None,
    positive_weights: Optional[NDArray[np.float32]] = None,
    hn_weight: float = RANKING_HN_WEIGHT,
    neg_weight: float = RANKING_NEGATIVE_WEIGHT,
    diversity_lambda: float = RANKING_DIVERSITY_LAMBDA,
    use_classifier: bool = False,
    use_contrastive: bool = False,
    knn_k: int = KNN_NEIGHBORS,
    progress_callback: Optional[Callable[[int, int], None]] = None,
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
            _, labels = cluster_interests_with_labels(X_pos, positive_weights)
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

            # Negatives get standard weight 1.0
            neg_sample_weights = np.ones(len(X_neg))

            sample_weights = np.concatenate([pos_sample_weights, neg_sample_weights])

            clf = LogisticRegression(class_weight="balanced", solver="liblinear", C=1.0)
            clf.fit(X_train, y_train, sample_weight=sample_weights)

            # Predict probabilities (class 1 = positive interest)
            probs = clf.predict_proba(cand_emb)[:, 1]
            semantic_scores = probs.astype(np.float32)

            # We still need max_sim_scores for the UI "Similar to..."
            sim_pos_ui = cosine_similarity(positive_embeddings, cand_emb)
            max_sim_scores = np.max(sim_pos_ui, axis=0)
            best_fav_indices = np.argmax(sim_pos_ui, axis=0)

            # Compute k-NN scores for display
            k = min(len(positive_embeddings), knn_k)
            if k > 0:
                top_k_sims = np.partition(sim_pos_ui, -k, axis=0)[-k:, :]
                raw_knn_scores = np.mean(top_k_sims, axis=0).astype(np.float32)
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
            # 1. k-Nearest Neighbors Scoring
            # Instead of centroids, compare candidates to actual history items.
            # This handles irregular interest shapes better than spherical centroids.
            
            # Calculate full similarity matrix: (n_history, n_candidates)
            # NOTE: Don't apply recency weights yet - we want pure semantic match first
            sim_matrix: NDArray[np.float32] = cosine_similarity(
                positive_embeddings, cand_emb
            )

            # Find top K neighbors for each candidate (along axis 0)
            k = min(len(positive_embeddings), knn_k)
            if k > 0:
                # np.partition moves the top K elements to the end
                # We take the last k rows (which are the largest)
                top_k_sims = np.partition(sim_matrix, -k, axis=0)[-k:, :]
                # Compute score as mean of top K matches
                knn_scores = np.mean(top_k_sims, axis=0)
            else:
                knn_scores = np.zeros(len(stories), dtype=np.float32)

            # Store raw k-NN scores for display before sigmoid
            raw_knn_scores = knn_scores.astype(np.float32)
            semantic_scores = knn_scores

            # For display score and best_fav_index, use the single best match
            # This preserves interpretable "match to specific story" display
            max_sim_scores = np.max(sim_matrix, axis=0)
            best_fav_indices = np.argmax(sim_matrix, axis=0)

            # Apply soft sigmoid activation instead of hard threshold
            # This suppresses noise while preserving strong signals
            semantic_scores = 1.0 / (
                1.0
                + np.exp(
                    -SEMANTIC_SIGMOID_K * (semantic_scores - SEMANTIC_SIGMOID_THRESHOLD)
                )
            )

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
                knn_neg: NDArray[np.float32] = np.mean(top_k_neg, axis=0)
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
    points: NDArray[Any] = np.array([float(s.score) for s in stories])
    hn_scores = np.log1p(points) / np.log1p(
        max(points.max(), HN_SCORE_NORMALIZATION_CAP)
    )

    # 5. Hybrid Score
    hybrid_scores: NDArray[np.float32] = (
        1 - hn_weight
    ) * semantic_scores + hn_weight * hn_scores

    # 6. Diversity (MMR)
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
            )
        )
        selected_mask[best_idx] = True

    return results
