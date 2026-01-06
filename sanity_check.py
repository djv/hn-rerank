#!/usr/bin/env -S uv run
"""Sanity check for HN rerank app."""
import asyncio
import time
import numpy as np
from api.rerank import (
    init_model,
    get_embeddings,
    rank_embeddings_maxsim,
    rank_mmr,
    rank_stories,
    compute_recency_weights,
    cluster_and_reduce_auto,
)
from api.fetching import fetch_article_text


def test_model_loading():
    """Test ONNX model loads successfully."""
    print("1. Testing model loading...")
    model = init_model("onnx_model")  # Use Nomic model
    assert model is not None
    assert hasattr(model, "encode")
    assert hasattr(model, "model_id")
    print(f"   ✓ Model loaded: {model.model_id}")


def test_embeddings():
    """Test embedding generation and caching."""
    print("\n2. Testing embeddings...")

    texts = [
        "Machine learning and artificial intelligence",
        "Deep learning neural networks",
        "Python programming tutorial",
    ]

    # First call (no cache)
    start = time.time()
    embeddings1 = get_embeddings(texts, is_query=False)
    t1 = time.time() - start

    assert embeddings1.shape == (3, 768), f"Expected (3, 768), got {embeddings1.shape}"
    print(f"   ✓ Generated embeddings: {embeddings1.shape} in {t1:.3f}s")

    # Second call (cached)
    start = time.time()
    embeddings2 = get_embeddings(texts, is_query=False)
    t2 = time.time() - start

    assert (embeddings1 == embeddings2).all(), "Cached embeddings mismatch"
    print(f"   ✓ Cache hit: {t2:.3f}s ({t1/t2:.1f}x speedup)")


def test_maxsim_ranking():
    """Test MaxSim ranking algorithm."""
    print("\n3. Testing MaxSim ranking...")

    favorites = [
        "Rust programming language systems performance",
        "Functional programming Haskell types",
    ]
    candidates = [
        "WebAssembly Rust compilation target",
        "JavaScript async await promises",
        "Type theory category theory",
        "Python pandas data analysis",
    ]

    fav_emb = get_embeddings(favorites, is_query=True)
    cand_emb = get_embeddings(candidates, is_query=False)

    results = rank_embeddings_maxsim(cand_emb, fav_emb)

    assert len(results) == 4
    assert results[0][1] > results[-1][1], "Results not sorted"

    # Expect Rust/WASM and Type theory to rank high
    top_idx = results[0][0]
    assert top_idx in [0, 2], f"Expected Rust or Type theory top, got: {candidates[top_idx]}"

    print(f"   ✓ Ranked {len(results)} candidates")
    print(f"     Top: {candidates[results[0][0]]} (score: {results[0][1]:.3f})")


def test_mmr_ranking():
    """Test MMR diversity ranking."""
    print("\n4. Testing MMR ranking...")

    favorites = ["Machine learning artificial intelligence"]
    candidates = [
        "Deep learning neural networks CNN",
        "Deep learning transformers attention",
        "Rust systems programming memory safety",
        "Python programming language tutorial",
    ]

    fav_emb = get_embeddings(favorites, is_query=True)
    cand_emb = get_embeddings(candidates, is_query=False)

    # Test with diversity penalty
    results = rank_mmr(cand_emb, fav_emb, diversity_penalty=0.5)

    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    assert all(0 <= idx < 4 for idx, _, _ in results), "Invalid indices"
    assert all(0.0 <= score <= 1.0 for _, score, _ in results), "Invalid scores"

    # Verify all candidates appear exactly once
    indices = [idx for idx, _, _ in results]
    assert len(set(indices)) == 4, "Duplicate indices in results"

    print(f"   ✓ MMR ranking works ({len(results)} candidates)")
    for i, (idx, score, _) in enumerate(results[:2]):
        print(f"     #{i+1}: {candidates[idx][:40]}... (score: {score:.3f})")


def test_clustering():
    """Test clustering reduces embeddings correctly."""
    print("\n5. Testing clustering...")

    # Create diverse stories that should form distinct clusters
    stories = [
        "Python programming tutorial",
        "Python best practices",
        "Rust memory safety",
        "Rust concurrency",
        "JavaScript async",
        "JavaScript promises",
        "Haskell functional programming",
        "Haskell type system",
        "Machine learning PyTorch",
        "Deep learning TensorFlow",
    ]

    embeddings = get_embeddings(stories, is_query=True)
    centroids, rep_indices, labels = cluster_and_reduce_auto(embeddings)

    n_clusters = len(centroids)
    assert 1 <= n_clusters <= len(stories), f"Invalid cluster count: {n_clusters}"
    assert len(rep_indices) == n_clusters
    assert len(centroids) == n_clusters

    # Verify centroids are normalized
    norms = np.linalg.norm(centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01), "Centroids not normalized"

    print(f"   ✓ Clustered {len(stories)} items → {n_clusters} cluster(s)")
    if n_clusters > 1:
        print(f"     Found {n_clusters} clusters")
        rep_indices = [np.where(labels == i)[0][0] for i in range(n_clusters)]
        print(f"     Representatives: {[stories[i] for i in rep_indices[:3]]}")
    else:
        print("     All items similar (1 cluster)")


def test_recency_weights():
    """Test recency weight computation."""
    print("\n6. Testing recency weights...")

    now = time.time()
    timestamps = [
        int(now),  # Now
        int(now - 86400),  # 1 day ago
        int(now - 86400 * 10),  # 10 days ago
        int(now - 86400 * 100),  # 100 days ago
    ]

    weights = compute_recency_weights(timestamps, decay_rate=0.01)

    assert len(weights) == 4
    assert weights[0] >= weights[1] >= weights[2] >= weights[3], "Weights not decreasing"
    assert 0.99 <= weights[0] <= 1.0, f"Recent weight should be ~1.0, got {weights[0]}"
    assert 0.35 <= weights[3] <= 0.40, f"100-day weight should be ~0.37, got {weights[3]}"

    print(f"   ✓ Recency weights: {weights}")


def test_hybrid_ranking():
    """Test hybrid semantic + HN score ranking."""
    print("\n7. Testing hybrid ranking...")

    now = time.time()
    stories = [
        {
            "id": 1,
            "title": "Python programming tutorial",
            "score": 100,
            "time": int(now - 3600),  # 1 hour old
            "text_content": "Learn Python programming from scratch",
        },
        {
            "id": 2,
            "title": "Rust systems programming",
            "score": 500,
            "time": int(now - 86400 * 7),  # 7 days old, high score
            "text_content": "Rust for systems programming",
        },
        {
            "id": 3,
            "title": "JavaScript tutorial",
            "score": 50,
            "time": int(now - 86400),  # 1 day old, low score
            "text_content": "Learn JavaScript basics",
        },
    ]

    favorites = ["Python programming"]
    fav_emb = get_embeddings(favorites, is_query=True)

    # Test with semantic only (hn_weight=0)
    results_semantic = rank_stories(
        stories, positive_embeddings=fav_emb, hn_weight=0.0
    )
    assert results_semantic[0][0] == 0, "Python should rank first (semantic)"

    # Test with HN weight
    results_hybrid = rank_stories(
        stories, positive_embeddings=fav_emb, hn_weight=0.5
    )
    # With HN weight, fresh high-scoring Rust might compete with Python

    print(f"   ✓ Semantic-only top: story #{stories[results_semantic[0][0]]['id']}")
    print(f"   ✓ Hybrid top: story #{stories[results_hybrid[0][0]]['id']}")


async def test_article_fetching():
    """Test article text extraction."""
    print("\n8. Testing article fetching...")

    # Test with a known stable URL (example.com)
    url = "http://example.com"
    text = await fetch_article_text(url)

    # example.com should return some text
    assert isinstance(text, str)
    print(f"   ✓ Fetched {len(text)} chars from {url}")


def main():
    """Run all sanity checks."""
    print("=" * 60)
    print("HN Rerank Sanity Check")
    print("=" * 60)

    try:
        test_model_loading()
        test_embeddings()
        test_maxsim_ranking()
        test_mmr_ranking()
        test_clustering()
        test_recency_weights()
        test_hybrid_ranking()
        asyncio.run(test_article_fetching())

        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
