import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from api.rerank import compute_recency_weights, rank_stories

@given(st.lists(st.integers(min_value=0, max_value=2000000000), min_size=1, max_size=100))
def test_recency_weights_invariants(timestamps):
    """
    Invariants for recency weights:
    1. All weights are in [0, 1].
    2. If T1 > T2 (T1 is newer), weight(T1) >= weight(T2).
    """
    weights = compute_recency_weights(timestamps)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    
    if len(timestamps) > 1:
        # Sort indices by timestamp descending (newest first)
        sorted_indices = np.argsort(timestamps)[::-1]
        sorted_weights = weights[sorted_indices]
        # Weight should be monotonic with respect to time
        # Allowing small epsilon for floating point
        diffs = np.diff(sorted_weights)
        assert np.all(diffs <= 1e-7)

@settings(deadline=None)
@given(
    num_candidates=st.integers(min_value=1, max_value=20),
    num_favorites=st.integers(min_value=1, max_value=10)
)
def test_ranking_invariants(num_candidates, num_favorites):
    """
    Invariants for rank_stories:
    1. Returns a list of (idx, score, fav_idx).
    2. Number of results <= min(num_candidates, 100).
    3. Scores are sorted descending.
    4. fav_idx corresponds to a valid index in positive_embeddings (or -1).
    """
    # Mock embeddings (random unit vectors)
    def random_unit_vectors(n, dim=384):
        vecs = np.random.randn(n, dim).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    pos_emb = random_unit_vectors(num_favorites)
    cand_emb = random_unit_vectors(num_candidates)
    
    stories = [
        {"id": i, "score": 100, "time": 1000, "text_content": f"Story {i}"}
        for i in range(num_candidates)
    ]
    
    # Mock get_embeddings to return our cand_emb
    import api.rerank
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)
        
        results = rank_stories(
            stories, 
            positive_embeddings=pos_emb, 
            hn_weight=0.0, # Pure semantic for this test
            diversity_lambda=0.0
        )
        
        assert len(results) == num_candidates
        
        last_score = float('inf')
        for idx, score, fav_idx in results:
            assert 0 <= idx < num_candidates
            assert score <= last_score + 1e-7
            assert 0 <= fav_idx < num_favorites
            last_score = score

@given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=10))
def test_hn_gravity_normalization(scores):
    """
    Invariants:
    1. HN scores should be normalized between 0 and 1.
    """
    stories = [
        {"id": i, "score": s, "time": 1000, "text_content": ""}
        for i, s in enumerate(scores)
    ]
    
    # We need dummy embeddings
    dummy_pos = np.zeros((1, 384), dtype=np.float32)
    dummy_cand = np.zeros((len(stories), 384), dtype=np.float32)
    
    import api.rerank
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: dummy_cand)
        
        # Pure HN weight
        results = rank_stories(
            stories, 
            positive_embeddings=dummy_pos, 
            hn_weight=1.0, 
            diversity_lambda=0.0
        )
        
        for _, score, _ in results:
            assert 0.0 <= score <= 1.000001
