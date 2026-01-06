
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from api import rerank

@pytest.fixture
def mock_stories():
    return [
        {"id": 1, "title": "S1", "score": 10, "time": 1000},
        {"id": 2, "title": "S2", "score": 20, "time": 2000},
    ]

@pytest.fixture
def mock_embeddings():
    # 2 candidates, 2 dimensions
    return np.array([
        [1.0, 0.0], # S1
        [0.0, 1.0], # S2
    ])

def test_rank_stories_empty():
    assert rerank.rank_stories([]) == []

def test_rank_stories_no_embeddings(mock_stories):
    with patch("api.rerank.get_embeddings", return_value=np.array([])):
        assert rerank.rank_stories(mock_stories) == []

def test_rank_stories_positive_weights(mock_stories, mock_embeddings):
    # Favorites: F1 matches S1.
    fav_embeddings = np.array([[1.0, 0.0]])

    # Weights: weight for F1 is 0.5 (low importance)
    positive_weights = np.array([0.5])

    # rank_stories calculates semantic score = max(sim * weight)
    # S1 sim with F1 = 1.0 * 0.5 = 0.5
    # S2 sim with F1 = 0.0 * 0.5 = 0.0

    # HN Score: S2 (20 pts) > S1 (10 pts) and S2 is newer.
    # But if we force HN weight to 0, it's pure semantic.

    with patch("api.rerank.get_embeddings", return_value=mock_embeddings):
        # We pass pre-calculated cand_embeddings to avoid mock call affecting logic
        results = rerank.rank_stories(
            mock_stories,
            cand_embeddings=mock_embeddings,
            positive_embeddings=fav_embeddings,
            positive_weights=positive_weights,
            hn_weight=0.0
        )

        # S1 should be first (score 0.5 vs 0.0)
        assert results[0][0] == 0 # Index of S1
        assert np.isclose(results[0][1], 0.5)

def test_rank_stories_negative_embeddings(mock_stories, mock_embeddings):
    # F1 matches S1. Neg1 matches S2.
    fav_embeddings = np.array([[1.0, 0.0]])
    neg_embeddings = np.array([[0.0, 1.0]])

    # Pure semantic
    # S1: pos=1.0, neg=0.0 -> score 1.0
    # S2: pos=0.0, neg=1.0 -> score 0.0 - (0.5 * 1.0) = -0.5

    results = rerank.rank_stories(
        mock_stories,
        cand_embeddings=mock_embeddings,
        positive_embeddings=fav_embeddings,
        negative_embeddings=neg_embeddings,
        neg_weight=0.5,
        hn_weight=0.0
    )

    assert results[0][0] == 0 # S1
    assert results[1][0] == 1 # S2
    assert results[1][1] < 0

def test_rank_stories_mmr():
    # S1: Best match.
    # S2: Almost identical to S1.
    # S3: Lower relevance, but unique (orthogonal).
    stories = [
        {"id": 1, "title": "S1", "score": 10, "time": 1000},
        {"id": 2, "title": "S2 (Redundant)", "score": 10, "time": 1000},
        {"id": 3, "title": "S3 (Unique)", "score": 10, "time": 1000},
    ]
    embeddings = np.array([
        [1.0, 0.0],       # S1
        [0.99, 0.01],     # S2 (Redundant with S1)
        [0.0, 1.0],       # S3 (Orthogonal to S1)
    ])
    fav_embeddings = np.array([[1.0, 0.0]]) # Likes S1 direction

    # High diversity penalty
    results = rerank.rank_stories(
        stories,
        cand_embeddings=embeddings,
        positive_embeddings=fav_embeddings,
        diversity_lambda=10.0,
        hn_weight=0.0
    )

    # Expected: S1 (best), then S3 (unique), then S2 (penalized)
    # S1 score: 1.0
    # S2 score: 0.99 - (10 * 0.99) = -8.9
    # S3 score: 0.0 (relevance to fav is 0) - (10 * 0) = 0.0

    # Wait, S3 relevance to [1.0, 0.0] is 0.0.
    # So S3 score is 0.0.
    # S2 score is -8.9.
    # So S3 > S2.

    # Order: S1 (idx 0), S3 (idx 2), S2 (idx 1)

    indices = [r[0] for r in results]
    assert indices == [0, 2, 1]

def test_rank_stories_hn_decay():
    # Only HN score matters
    stories = [
        {"id": 1, "title": "Old High Score", "score": 500, "time": 1000},
        {"id": 2, "title": "New Low Score", "score": 10, "time": 2000000000}, # Far in future/recent
    ]
    # Mock current time to be close to New
    current_time = 2000000000

    # If we use hn_weight=1.0, semantic doesn't matter
    # We need to manually invoke calculate_hn_score logic implicitly via rank_stories

    with patch("time.time", return_value=current_time):
        results = rerank.rank_stories(
            stories,
            cand_embeddings=np.zeros((2, 2)), # Neutral embeddings
            positive_embeddings=None,
            hn_weight=1.0
        )
        # S2 is fresh (0 hours old). S1 is very old.
        # S2 should probably win despite lower score?
        # Let's check the formula: (P-1)^0.8 / (Age+2)^1.8
        # S1: Age huge. Score ~0.
        # S2: Age 0. Score > 0.

        assert results[0][0] == 1 # S2 first
