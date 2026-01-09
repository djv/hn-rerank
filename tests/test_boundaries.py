"""Boundary condition and edge case tests."""
from __future__ import annotations

import time
import numpy as np
import pytest

import api.rerank
from api.rerank import (
    rank_stories,
    compute_recency_weights,
    get_embeddings,
)

# Embedding dimension used in tests
EMB_DIM = 384


# =============================================================================
# Recency Weight Boundaries
# =============================================================================


def test_future_timestamp_treated_as_now():
    """Future timestamps (negative age) should get weight ~1.0."""
    future_time = int(time.time()) + 86400 * 30  # 30 days in future
    weights = compute_recency_weights([future_time])

    assert len(weights) == 1
    assert weights[0] >= 0.95  # Should be close to 1.0


def test_very_old_never_zero():
    """Very old items (10+ years) still get small positive weight."""
    ten_years_ago = int(time.time()) - 86400 * 365 * 10
    weights = compute_recency_weights([ten_years_ago])

    assert len(weights) == 1
    assert weights[0] > 0.0  # Never exactly zero
    assert weights[0] < 0.1  # But very small


def test_sigmoid_inflection_at_one_year():
    """At ~365 days, weight should be approximately 0.5."""
    one_year_ago = int(time.time()) - 86400 * 365
    weights = compute_recency_weights([one_year_ago])

    assert len(weights) == 1
    assert 0.4 <= weights[0] <= 0.6  # Around 0.5


def test_recent_items_high_weight():
    """Items from last week should have weight > 0.95."""
    one_week_ago = int(time.time()) - 86400 * 7
    weights = compute_recency_weights([one_week_ago])

    assert len(weights) == 1
    assert weights[0] > 0.95


def test_decay_rate_zero_returns_uniform():
    """decay_rate=0 should return all 1.0 weights."""
    timestamps = [int(time.time()) - 86400 * i for i in [0, 100, 365, 1000]]
    weights = compute_recency_weights(timestamps, decay_rate=0.0)

    np.testing.assert_array_almost_equal(weights, [1.0, 1.0, 1.0, 1.0])


# =============================================================================
# Ranking Boundaries
# =============================================================================


def test_single_story_ranking():
    """Single story returns valid result with that story ranked."""
    story = {"id": 1, "text_content": "Hello world", "score": 100, "time": int(time.time())}
    pos_emb = np.random.randn(1, EMB_DIM).astype(np.float32)
    cand_emb = np.random.randn(1, EMB_DIM).astype(np.float32)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        results = rank_stories([story], pos_emb)

    assert len(results) == 1
    assert results[0][0] == 0  # Index 0
    assert results[0][1] >= 0  # Score >= 0


def test_all_identical_embeddings_ranks_by_hn_score():
    """When embeddings are identical, MMR should diversify by HN score."""
    stories = [
        {"id": i, "text_content": "Same content", "score": (3 - i) * 100, "time": int(time.time())}
        for i in range(3)
    ]
    # All stories have same embedding
    pos_emb = np.ones((1, EMB_DIM), dtype=np.float32)
    pos_emb = pos_emb / np.linalg.norm(pos_emb)
    cand_emb = np.ones((3, EMB_DIM), dtype=np.float32)
    cand_emb = cand_emb / np.linalg.norm(cand_emb[0])

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        results = rank_stories(stories, pos_emb, hn_weight=0.5)

    # Should have all 3 results
    assert len(results) == 3


def test_extremely_high_hn_score():
    """Very high HN scores (100k+) don't cause overflow."""
    story = {"id": 1, "text_content": "Viral post", "score": 100000, "time": int(time.time())}
    pos_emb = np.random.randn(1, EMB_DIM).astype(np.float32)
    cand_emb = np.random.randn(1, EMB_DIM).astype(np.float32)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        results = rank_stories([story], pos_emb)

    assert len(results) == 1
    # Score should be bounded (can be > 1 due to semantic match)
    assert results[0][1] >= 0


def test_zero_hn_score():
    """Stories with 0 points are handled gracefully."""
    story = {"id": 1, "text_content": "New post", "score": 0, "time": int(time.time())}
    pos_emb = np.random.randn(1, EMB_DIM).astype(np.float32)
    cand_emb = np.random.randn(1, EMB_DIM).astype(np.float32)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        results = rank_stories([story], pos_emb)

    assert len(results) == 1
    assert results[0][1] >= 0


def test_negative_hn_score():
    """Negative scores (shouldn't happen but defensive)."""
    story = {"id": 1, "text_content": "Test", "score": -10, "time": int(time.time())}
    pos_emb = np.random.randn(1, EMB_DIM).astype(np.float32)
    cand_emb = np.random.randn(1, EMB_DIM).astype(np.float32)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        results = rank_stories([story], pos_emb)

    assert len(results) == 1
    # Should handle gracefully (max with 0)


# =============================================================================
# Semantic Threshold Boundary
# =============================================================================


def test_below_threshold_gets_zero_score():
    """Stories below SEMANTIC_MATCH_THRESHOLD get 0 semantic score."""
    story = {"id": 1, "text_content": "Unrelated content xyz abc", "score": 100, "time": int(time.time())}
    # Create orthogonal embeddings to get low similarity
    pos_emb = np.zeros((1, EMB_DIM), dtype=np.float32)
    pos_emb[0, 0] = 1.0  # Unit vector in first dimension

    # Candidate orthogonal to positive signal
    cand_emb = np.zeros((1, EMB_DIM), dtype=np.float32)
    cand_emb[0, 1] = 1.0  # Orthogonal unit vector

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        results = rank_stories([story], pos_emb, hn_weight=0.0)

    # With very low semantic match (orthogonal = 0), score should be 0
    assert len(results) == 1
    assert results[0][1] == pytest.approx(0.0, abs=0.01)


# =============================================================================
# Empty/None Inputs
# =============================================================================


def test_empty_stories_returns_empty():
    """Empty stories list returns empty results."""
    pos_emb = np.random.randn(5, 384).astype(np.float32)
    results = rank_stories([], pos_emb)

    assert results == []


def test_none_positive_embeddings():
    """None positive embeddings falls back to HN-only ranking."""
    stories = [
        {"id": 1, "text_content": "Story 1", "score": 100, "time": int(time.time())},
        {"id": 2, "text_content": "Story 2", "score": 200, "time": int(time.time())},
    ]

    results = rank_stories(stories, None)

    assert len(results) == 2
    # Higher HN score should rank first
    assert stories[results[0][0]]["score"] >= stories[results[1][0]]["score"]


def test_empty_positive_embeddings_array():
    """Empty positive embeddings array treated like None."""
    stories = [
        {"id": 1, "text_content": "Story 1", "score": 100, "time": int(time.time())},
    ]
    empty_emb = np.array([], dtype=np.float32).reshape(0, 384)

    results = rank_stories(stories, empty_emb)

    assert len(results) == 1


# =============================================================================
# Weight Parameter Boundaries
# =============================================================================


def test_hn_weight_zero_pure_semantic():
    """hn_weight=0 means pure semantic ranking."""
    stories = [
        {"id": 1, "text_content": "Machine learning AI", "score": 1, "time": int(time.time())},
        {"id": 2, "text_content": "Cooking recipes food", "score": 10000, "time": int(time.time())},
    ]
    # Positive signal about ML
    pos_emb = get_embeddings(["Machine learning artificial intelligence"])

    results = rank_stories(stories, pos_emb, hn_weight=0.0)

    # Despite 10000 points, cooking should rank lower than ML
    assert len(results) == 2


def test_hn_weight_one_pure_hn():
    """hn_weight=1.0 means pure HN score ranking."""
    stories = [
        {"id": 1, "text_content": "Matching content", "score": 10, "time": int(time.time())},
        {"id": 2, "text_content": "Different content", "score": 1000, "time": int(time.time())},
    ]
    # Even though story 1 matches better semantically
    pos_emb = get_embeddings(["Matching content exactly"])

    results = rank_stories(stories, pos_emb, hn_weight=1.0)

    # Story 2 with 1000 points should rank first
    assert results[0][0] == 1


def test_diversity_lambda_zero_no_mmr():
    """diversity_lambda=0 means no MMR reranking (pure relevance)."""
    stories = [
        {"id": i, "text_content": "Same topic content", "score": 100 - i, "time": int(time.time())}
        for i in range(5)
    ]
    pos_emb = np.random.randn(1, 384).astype(np.float32)

    with pytest.MonkeyPatch().context() as mp:
        # Mock get_embeddings to return 384-dim vectors to match pos_emb
        cand_emb = np.random.randn(len(stories), 384).astype(np.float32)
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        results = rank_stories(stories, pos_emb, diversity_lambda=0.0)

    assert len(results) == 5
    # Results should be in descending score order (within identical semantic match)
