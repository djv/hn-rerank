from __future__ import annotations
import asyncio
import hashlib
import json
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
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    MIN_SAMPLES_PER_CLUSTER,
    TEXT_CONTENT_MAX_LENGTH,
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
            except Exception:
                pass
        elif cache_path_npy.exists():
            try:
                vec = np.load(cache_path_npy)
            except Exception:
                pass

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

    # Search for highest k with acceptable silhouette (>= 0.1)
    # This gives more granular clusters while maintaining coherence
    # Tuned to favor slightly fewer, broader clusters
    min_k = max(MIN_CLUSTERS, int(np.sqrt(n_samples) * 0.8))
    max_k = min(MAX_CLUSTERS, int(np.sqrt(n_samples) * 2.5), n_samples // MIN_SAMPLES_PER_CLUSTER)

    best_labels: NDArray[np.int32] = np.zeros(n_samples, dtype=np.int32)
    silhouette_threshold = 0.14

    # Search from high to low k, pick first that meets threshold
    for k in range(max_k, min_k - 1, -1):
        agg = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        labels = agg.fit_predict(normalized)
        score = float(silhouette_score(normalized, labels))
        if score >= silhouette_threshold:
            best_labels = labels.astype(np.int32)
            break
    else:
        # No k met threshold, use max_k anyway
        agg = AgglomerativeClustering(
            n_clusters=max_k, metric="cosine", linkage="average"
        )
        best_labels = agg.fit_predict(normalized).astype(np.int32)

    # Post-process: Merge tiny clusters (< MIN_SAMPLES_PER_CLUSTER)
    # We iteratively merge the smallest cluster into its nearest neighbor
    # until all clusters meet the size requirement or we are down to 1 cluster.
    while True:
        unique, counts = np.unique(best_labels, return_counts=True)
        small_clusters = unique[counts < MIN_SAMPLES_PER_CLUSTER]
        if len(small_clusters) == 0:
            break
        
        # If we only have 1 cluster left, we can't merge further
        if len(unique) <= 1:
            break
            
        # Calculate centroids of current clusters for merging distance
        current_centroids = {}
        for lbl in unique:
            current_centroids[lbl] = np.mean(normalized[best_labels == lbl], axis=0)
            
        # Pick the smallest cluster to merge
        # Sort small clusters by size (ascending), pick first
        # Actually, just picking the first found is fine, but sorting by size ensures we merge 1s before 2s
        target_small = small_clusters[np.argmin(counts[counts < MIN_SAMPLES_PER_CLUSTER])]
        target_centroid = current_centroids[target_small]
        
        # Find nearest other centroid
        best_sim = -2.0  # Cosine sim is [-1, 1]
        merge_partner = -1
        
        for other_lbl in unique:
            if other_lbl == target_small:
                continue
            
            sim = np.dot(target_centroid, current_centroids[other_lbl])
            if sim > best_sim:
                best_sim = sim
                merge_partner = other_lbl
        
        # Merge target_small into merge_partner
        if merge_partner != -1:
            best_labels[best_labels == target_small] = merge_partner
        else:
            # Should not happen if len(unique) > 1
            break

    # Renumber labels to be consecutive (0, 1, 2...)
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
_token_bucket: float = 1.0
_last_refill: float = time.time()
# Conservative rate limit: 1 call per 4 seconds (~15 RPM)
_refill_rate: float = 1.0 / 4.0
_max_tokens: float = 1.0


async def _rate_limit() -> None:
    """Enforce rate limiting using a Token Bucket algorithm."""
    global _token_bucket, _last_refill

    while True:
        now = time.time()
        # Refill tokens based on time passed
        elapsed = now - _last_refill
        _token_bucket = min(_max_tokens, _token_bucket + elapsed * _refill_rate)
        _last_refill = now

        if _token_bucket >= 1.0:
            _token_bucket -= 1.0
            return

        # Calculate wait time needed for next token
        wait_time = (1.0 - _token_bucket) / _refill_rate
        # Add a small buffer and jitter to prevent thundering herds
        import random

        await asyncio.sleep(wait_time + 0.1 + random.uniform(0, 0.5))


CLUSTER_NAME_CACHE_PATH = Path(".cache/cluster_names.json")


def _load_cluster_name_cache() -> dict[str, str]:
    if CLUSTER_NAME_CACHE_PATH.exists():
        try:
            return json.loads(CLUSTER_NAME_CACHE_PATH.read_text())
        except Exception:
            pass
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
    except json.JSONDecodeError:
        # Fallback: try to find anything between { and }
        import re

        match = re.search(r"({.*})", clean_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        return {}


async def _generate_with_retry(
    model: str = "llama-3.1-8b-instant",
    contents: Optional[Any] = None,
    config: Optional[dict[str, Any]] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Groq API with exponential backoff retry logic using httpx."""
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
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
        "temperature": config.get("temperature", 0.2) if config else 0.2,
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
                    timeout=30.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]

                if resp.status_code == 429:
                    print(
                        f"[DEBUG] Groq Rate Limit hit (429). Attempt {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(20.0 * (2**attempt))
                    continue

                print(f"[ERROR] Groq API error {resp.status_code}: {resp.text}")
                return None

            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"[ERROR] Groq API call failed after {max_retries} retries: {e}"
                    )
                    return None
                delay = 10.0 * (2**attempt)
                await asyncio.sleep(delay)
    return None


async def generate_single_cluster_name(
    items: list[tuple[dict[str, Any], float]],
) -> str:
    """Generate a name for a single cluster using Groq API."""

    # Generate cache key based on sorted story IDs
    story_ids = sorted([str(s.get("id", s.get("objectID", ""))) for s, _ in items])
    cache_key = hashlib.sha256(",".join(story_ids).encode()).hexdigest()

    cache = _load_cluster_name_cache()
    if cache_key in cache:
        return cache[cache_key]

    # Get top titles by weight
    sorted_items = sorted(items, key=lambda x: -x[1])[:10]
    titles = []
    for story, _ in sorted_items:
        title = str(story.get("title", "")).strip()
        for prefix in ["Show HN:", "Ask HN:", "Tell HN:"]:
            if title.startswith(prefix):
                title = title[len(prefix) :].strip()
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
        text = await _generate_with_retry(
            model="llama-3.1-8b-instant",
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 20},
        )
        if not text:
            return "Misc"

        name = text.strip().strip('"').strip("'")
        # Truncate if too long (6 words allows multi-topic names like "AI, Space & Biology")
        words = name.split()[:6]
        final_name = " ".join(words) if words else "Misc"
        # Strip trailing conjunctions/punctuation left by truncation
        final_name = final_name.rstrip(" ,&").rstrip()
        if final_name.endswith(" and") or final_name.endswith(" or"):
            final_name = final_name.rsplit(" ", 1)[0]

        cache[cache_key] = final_name
        _save_cluster_name_cache(cache)

        return final_name
    except Exception:
        return "Misc"


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

    # Batch clusters together (max 10 per request)
    BATCH_SIZE = 10
    cid_list = list(to_generate.keys())

    for i in range(0, len(cid_list), BATCH_SIZE):
        batch_cids = cid_list[i : i + BATCH_SIZE]
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
Identify a highly specific 1-3 word shared topic/theme for each of these {len(batch_cids)} story groups.
- The input is a list of stories (title + comment context) for each cluster.
- Use the context to identify the true subject.
- If a group contains distinct/unrelated topics (e.g. Space AND Biology), list them: "Space, Bio & AI".
- Avoid generic terms like "Technology", "News", or "Interesting".
- Be specific (e.g. "Distributed Systems", "Sci-Fi Books", "Career Advice").

Return ONLY a JSON object where keys are the EXACT Cluster IDs provided and values are the topics.

Example: {{ "0": "React Hooks", "1": "Space, Bio & AI" }}

Groups:
{chr(10).join(batch_prompts)}

JSON Output:"""

        try:
            text = await _generate_with_retry(
                model="llama-3.1-8b-instant",
                contents=full_prompt,
                config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            )

            if text:
                batch_results = _safe_json_loads(text)
                for cid_str, name in batch_results.items():
                    try:
                        cid = int(cid_str)
                        # Truncate to 6 words (allows multi-topic names)
                        final_name = str(name).strip().split()[:6]
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
                    except (ValueError, TypeError):
                        continue

            if progress_callback:
                progress_callback(len(results), len(clusters))
        except Exception:
            pass

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


async def generate_cluster_names(
    clusters: dict[int, list[tuple[dict[str, Any], float]]],
) -> dict[int, str]:
    """Generate cluster names using Groq API."""
    return await generate_batch_cluster_names(clusters)


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

    # Chunking: 5 stories per request
    BATCH_SIZE = 5
    total_to_gen = len(to_generate)
    completed_initial = len(stories) - total_to_gen

    for i in range(0, total_to_gen, BATCH_SIZE):
        batch = to_generate[i : i + BATCH_SIZE]

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
For each story below, provide a 2-sentence summary (core subject, and key takeaway).
Return ONLY a JSON object where keys are the story IDs (strings) and values are the summaries.

Example: {{ "123": "Story is about X. Key takeaway is Y." }}

Stories:
{batch_context}

JSON Output:"""

        try:
            text = await _generate_with_retry(
                model="llama-3.1-8b-instant",
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 2000,
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
                    except (ValueError, TypeError):
                        continue

            if progress_callback:
                progress_callback(completed_initial + i + len(batch), len(stories))

        except Exception:
            pass

    _save_tldr_cache(cache)
    return {
        int(s["id"]): results.get(int(s["id"]), cache.get(str(s["id"]), ""))
        for s in stories
    }


async def generate_story_tldr(story_id: int, title: str, comments: list[str]) -> str:
    """Generate a 1-sentence TL;DR for a story using Groq API. Cached by story ID."""
    if not title:
        return ""

    # Check cache first
    cache = _load_tldr_cache()
    cache_key = str(story_id)
    if cache_key in cache:
        return cache[cache_key]

    # Build context from title + top comments
    context = f"Title: {title}"
    if comments:
        context += "\n\nTop comments:\n" + "\n".join(
            f"- {c[:400]}" for c in comments[:6]
        )

    prompt = f"""
{context}

Provide a concise summary of the story and its discussion. 
Do NOT use introductory phrases. 
Start directly with the content.

Structure:
- Sentence 1: Core subject or project.
- Sentence 2+: Main technical or philosophical debate and key takeaways from the comments.

IMPORTANT: Put the comments summary (Sentence 2+) on a single new line directly below the first sentence (no empty line)."""

    try:
        text = await _generate_with_retry(
            model="llama-3.1-8b-instant",
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 400},
        )
        if not text:
            return ""

        tldr = text.strip().strip('"').strip("'")

        # Aggressive cleaning of conversational filler
        useless_prefixes = [
            "Here is a summary:",
            "Here is a 2-sentence summary:",
            "Here is a 3-sentence summary:",
            "This story is about",
            "The story is about",
            "This article is about",
            "The discussion reveals that",
            "The discussion reveals",
            "TL;DR:",
            "TLDR:",
            "Summary:",
            "In this story,",
            "In this article,",
        ]

        lower_tldr = tldr.lower()
        for prefix in useless_prefixes:
            if lower_tldr.startswith(prefix.lower()):
                tldr = tldr[len(prefix) :].lstrip(":* \n")
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
    if decay_rate is not None and decay_rate <= 0:
        return np.ones(len(timestamps), dtype=np.float32)

    now: float = time.time()
    ages_days: NDArray[Any] = (now - np.array(timestamps)) / 86400

    k = 0.01
    inflection = 365

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
    diversity_lambda: float = 0.45,  # Increased for better discovery (was 0.35)
    use_classifier: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[tuple[int, float, int, float]]:
    """
    Returns list of (index, hybrid_score, best_fav_index, max_sim_score).

    Uses multi-interest clustering and MMR diversity reranking.
    Optionally uses a Logistic Regression classifier if sufficient data exists.
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
            weight_map = {lbl: 1.0 / np.log1p(count) for lbl, count in zip(unique_labels, counts)}
            
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
            
            classifier_success = True
            # Classifier probabilities are sharp (often >0.9 for dominant clusters).
            # We increase diversity penalty to ensure we skip to the next cluster.
            diversity_lambda = max(diversity_lambda, 0.6)
        except Exception:
            # Fallback to heuristic on error
            pass

    if not classifier_success:
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

            # Combined: weight MaxSim much higher to preserve niche interests and avoid noise dilution
            # 0.95 MaxSim + 0.05 MeanSim balances specific match with broad appeal
            semantic_scores = 0.95 * cluster_max_sim + 0.05 * cluster_mean_sim

            # For display score and best_fav_index, use original embeddings (not clusters)
            # This preserves interpretable "match to specific story" display
            # NOTE: Don't apply recency weights here - we want pure semantic match for UI
            # (recency is already factored into ranking via cluster centroids)
            sim_pos: NDArray[np.float32] = cosine_similarity(positive_embeddings, cand_emb)
            max_sim_scores = np.max(sim_pos, axis=0)
            best_fav_indices = np.argmax(sim_pos, axis=0)

            # Apply soft sigmoid activation instead of hard threshold
            # This suppresses noise while preserving strong signals
            # k=15 makes the transition reasonably steep around the threshold
            # threshold=0.35 is permissive for users with eclectic tastes
            k = 15.0
            threshold = 0.35
            semantic_scores = 1.0 / (1.0 + np.exp(-k * (semantic_scores - threshold)))

        # 3. Negative Signal (Penalty) - Only applies in heuristic mode
        if negative_embeddings is not None and len(negative_embeddings) > 0:
            sim_neg: NDArray[np.float32] = np.max(
                cosine_similarity(negative_embeddings, cand_emb), axis=0
            )
            semantic_scores -= neg_weight * sim_neg

    # 4. HN Gravity Score (Log-scaled)
    # We use a log scale so that 500+ points punch through without dominating everything
    points: NDArray[Any] = np.array([float(s.get("score", 0)) for s in stories])
    hn_scores = np.log1p(points) / np.log1p(max(points.max(), 500))

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

