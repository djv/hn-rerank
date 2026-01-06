import numpy as np
import time
from api import rerank


def test_rank_stories_weighted_logic():
    """
    Verify that rank_stories combines semantic and HN scores correctly.
    """
    now = time.time()

    # 2 Candidates
    # C1: Sem match 0.5, Points 10 (Low HN score)
    # C2: Sem match 0.5, Points 500 (High HN score)
    stories = [
        {"id": 1, "title": "Old/Low", "score": 10, "time": now - 3600 * 24},  # 24h old
        {"id": 2, "title": "New/High", "score": 500, "time": now - 3600},  # 1h old
    ]

    # Embeddings: Both match F1 with 0.5
    # F1: [1, 0]
    # C1: [0.5, 0.866]
    # C2: [0.5, 0.866]
    fav_embeddings = np.array([[1.0, 0.0]])
    cand_embeddings = np.array([[0.5, 0.866], [0.5, 0.866]])

    # Run with 50% weight to make HN score impact obvious
    results = rerank.rank_stories(
        stories,
        cand_embeddings=cand_embeddings,
        positive_embeddings=fav_embeddings,
        negative_embeddings=None,
        hn_weight=0.5,
    )

    # C2 should be ranked first because of higher HN score
    assert results[0][0] == 1  # Index 1 is C2
    assert results[1][0] == 0  # Index 0 is C1

    # Verify score > 0.5 (since HN boost)
    # C2 score = 0.5 * 0.5 + 0.5 * (normalized_hn ~ 1.0) = 0.75
    assert results[0][1] > 0.6


def test_rank_stories_tie_breaking():
    """
    Verify tie breaking by time.
    """
    now = time.time()

    # C1 and C2 identical scores and embeddings
    stories = [
        {"id": 1, "title": "Older", "score": 100, "time": now - 100},
        {"id": 2, "title": "Newer", "score": 100, "time": now - 10},
    ]

    fav_embeddings = np.array([[1.0]])
    cand_embeddings = np.array([[1.0], [1.0]])

    results = rerank.rank_stories(
        stories,
        cand_embeddings=cand_embeddings,
        positive_embeddings=fav_embeddings,
        negative_embeddings=None,
        hn_weight=0.5,
    )

    # Newer (Index 1) should be first
    assert results[0][0] == 1
    assert results[1][0] == 0
