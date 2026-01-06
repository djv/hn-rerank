import numpy as np
import hashlib
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering

# Global singleton
_model = None

CACHE_DIR = Path(".cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ONNXEmbeddingModel:
    def __init__(self, model_dir="bge_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        # Use the quantized version for speed
        self.session = ort.InferenceSession(f"{model_dir}/model_quantized.onnx")

    def encode(self, texts, normalize_embeddings=True, batch_size=4):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Tokenize
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
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

            batch_embeddings = sum_embeddings / sum_mask

            if normalize_embeddings:
                norm = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.clip(
                    norm, a_min=1e-9, a_max=None
                )

            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


def init_model(model_name: str = "bge_model"):
    global _model
    if _model is None:
        _model = ONNXEmbeddingModel(model_name)
    return _model


def get_model():
    if _model is None:
        return init_model()
    return _model


def get_cache_key(text: str, model_name: str) -> Path:
    content = f"{model_name}:{text}"
    hash_digest = hashlib.sha256(content.encode()).hexdigest()
    return CACHE_DIR / f"{hash_digest}.npy"


def get_embeddings(texts: list[str], is_query: bool = False) -> np.ndarray:
    if not texts:
        return np.array([])

    model = get_model()
    model_id = "onnx-bge-base-1.5"

    # BGE does not require the "search_document:" prefix, only a query prefix
    if is_query:
        prefix = "Represent this sentence for searching relevant passages: "
    else:
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
        computed_vectors = model.encode(texts_to_compute, normalize_embeddings=True)

        for i, original_idx in enumerate(indices_to_compute):
            vec = computed_vectors[i]
            vectors[original_idx] = vec
            cache_path = get_cache_key(processed_texts[original_idx], model_id)
            np.save(cache_path, vec)

    return np.array(vectors)


def cluster_and_reduce_auto(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Cluster embeddings using Agglomerative Clustering (Auto K).
    """
    if len(embeddings) < 2:
        return embeddings, list(range(len(embeddings))), list(range(len(embeddings)))

    # distance_threshold 0.8 means items with >0.2 similarity can group.
    # linkage='complete' or 'average' works well.
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.8, metric="euclidean", linkage="complete"
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
        if norm > 1e-9:
            centroid = centroid / norm
        centroids.append(centroid)

        if len(member_indices) == 1:
            representative_indices.append(member_indices[0])
        else:
            sims = cosine_similarity(centroid.reshape(1, -1), cluster_embeds)[0]
            best_local_idx = np.argmax(sims)
            representative_indices.append(member_indices[best_local_idx])

    return np.array(centroids), representative_indices, labels.tolist()


def cluster_and_reduce(
    embeddings: np.ndarray, k: int
) -> tuple[np.ndarray, list[int], list[int]]:
    if len(embeddings) <= k:
        return embeddings, list(range(len(embeddings))), list(range(len(embeddings)))

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    sim_matrix = cosine_similarity(centroids, embeddings)
    representative_indices = np.argmax(sim_matrix, axis=1).tolist()

    return centroids, representative_indices, kmeans.labels_.tolist()


def rank_embeddings_maxsim(
    candidate_embeddings: np.ndarray, fav_embeddings: np.ndarray
) -> list[tuple[int, float, int]]:
    if len(candidate_embeddings) == 0 or len(fav_embeddings) == 0:
        return []
    sim_matrix = cosine_similarity(fav_embeddings, candidate_embeddings)
    max_scores = np.max(sim_matrix, axis=0)
    best_fav_indices = np.argmax(sim_matrix, axis=0)
    results = []
    for idx, (score, fav_idx) in enumerate(zip(max_scores, best_fav_indices)):
        results.append((idx, float(score), int(fav_idx)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def rank_mmr(
    cand_embeddings: np.ndarray,
    fav_embeddings: np.ndarray,
    diversity_penalty: float,
    cand_texts: list[str] = None,
    fav_texts: list[str] = None,
) -> list[tuple[int, float, int]]:
    if len(cand_embeddings) == 0 or len(fav_embeddings) == 0:
        return []
    sim_matrix = cosine_similarity(fav_embeddings, cand_embeddings)
    relevance_scores = np.max(sim_matrix, axis=0)
    results = []
    selected_indices = []
    for _ in range(len(cand_embeddings)):
        best_score = -float("inf")
        best_cand_idx = -1
        for cand_idx in range(len(cand_embeddings)):
            if cand_idx in selected_indices:
                continue
            relevance = relevance_scores[cand_idx]
            redundancy = 0.0
            if selected_indices:
                picked_embeds = cand_embeddings[selected_indices]
                cand_embed = cand_embeddings[cand_idx].reshape(1, -1)
                redundancy = np.max(cosine_similarity(cand_embed, picked_embeds)[0])
            score = relevance - (diversity_penalty * redundancy)
            if score > best_score:
                best_score = score
                best_cand_idx = cand_idx
        if best_cand_idx != -1:
            best_fav_idx = np.argmax(sim_matrix[:, best_cand_idx])
            results.append(
                (
                    best_cand_idx,
                    float(relevance_scores[best_cand_idx]),
                    int(best_fav_idx),
                )
            )
            selected_indices.append(best_cand_idx)
        else:
            break
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


def calculate_hn_score(points: int, time_ts: int, current_time: float = None) -> float:
    """
    Calculate HN score with gravity decay.
    Score = (P - 1)^0.8 / (T + 2)^1.8
    """
    if current_time is None:
        import time

        current_time = time.time()

    hours_age = (current_time - time_ts) / 3600
    if hours_age < 0:
        hours_age = 0

    numerator = (points - 1) ** 0.8 if points > 1 else 0
    denominator = (hours_age + 2) ** 1.8

    return numerator / denominator


def rank_stories(
    stories: list[dict],
    cand_embeddings: np.ndarray = None,
    positive_embeddings: np.ndarray = None,
    negative_embeddings: np.ndarray = None,
    positive_weights: np.ndarray = None,
    diversity_lambda: float = 0.0,
    hn_weight: float = 0.15,
    neg_weight: float = 0.5,
) -> list[tuple[int, float, int]]:
    """
    Rank stories using hybrid score: (1-w)*Semantic + w*HN_Score.
    Semantic = WeightedMaxSim(Positive) - neg_weight * MaxSim(Negative).
    """
    if not stories:
        return []

    # 1. Embeddings
    if cand_embeddings is None:
        cand_texts = [s.get("text_content", s.get("title", "")) for s in stories]
        cand_embeddings = get_embeddings(cand_texts, is_query=False)

    if len(cand_embeddings) == 0:
        return []

    # 2. Semantic Scores
    semantic_scores = np.zeros(len(cand_embeddings))
    best_fav_indices = np.zeros(len(cand_embeddings), dtype=int) - 1

    if positive_embeddings is not None and len(positive_embeddings) > 0:
        sim_pos = cosine_similarity(positive_embeddings, cand_embeddings)

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
        sim_neg = cosine_similarity(negative_embeddings, cand_embeddings)
        neg_scores = np.max(sim_neg, axis=0)
        semantic_scores -= neg_weight * neg_scores

    # 3. HN Scores (Decay)
    import time

    now = time.time()
    hn_scores = np.array(
        [
            calculate_hn_score(s.get("score", 0), s.get("time", now), now)
            for s in stories
        ]
    )

    # Normalize HN scores to 0-1 range
    if hn_scores.max() > 0:
        hn_scores = hn_scores / hn_scores.max()

    # 4. Hybrid Score
    hybrid_scores = (1 - hn_weight) * semantic_scores + (hn_weight * hn_scores)

    # 5. Ranking (MaxSim or MMR)
    results = []

    if diversity_lambda > 0:
        # MMR Logic
        selected_indices = []

        for _ in range(len(cand_embeddings)):
            best_mmr_score = -float("inf")
            best_cand_idx = -1

            for cand_idx in range(len(cand_embeddings)):
                if cand_idx in selected_indices:
                    continue

                relevance = hybrid_scores[cand_idx]
                redundancy = 0.0

                if selected_indices:
                    picked_embeds = cand_embeddings[selected_indices]
                    cand_embed = cand_embeddings[cand_idx].reshape(1, -1)
                    redundancy = np.max(cosine_similarity(cand_embed, picked_embeds)[0])

                mmr_score = relevance - (diversity_lambda * redundancy)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_cand_idx = cand_idx

            if best_cand_idx != -1:
                results.append(
                    (
                        best_cand_idx,
                        float(hybrid_scores[best_cand_idx]),
                        int(best_fav_indices[best_cand_idx]),
                    )
                )
                selected_indices.append(best_cand_idx)
            else:
                break
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
