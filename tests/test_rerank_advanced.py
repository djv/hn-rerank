import numpy as np

from api import rerank


def test_calculate_hn_score():
    # Fresh story (0 hours old)
    s1 = rerank.calculate_hn_score(100, 1000, current_time=1000)
    # Older story (10 hours old)
    s2 = rerank.calculate_hn_score(100, 1000 - 36000, current_time=1000)
    assert s1 > s2

    # Low points
    s3 = rerank.calculate_hn_score(1, 1000, current_time=1000)
    assert s3 == 0

def test_rank_mmr_edge_cases():
    # Empty inputs
    assert rerank.rank_mmr(np.array([]), np.array([]), 0.5) == []

    # Single candidate
    cand = np.array([[1.0, 0.0]])
    fav = np.array([[1.0, 0.0]])
    res = rerank.rank_mmr(cand, fav, 0.5)
    assert len(res) == 1
    assert res[0][0] == 0 # index 0
    assert res[0][1] == 1.0 # score 1.0

def test_rank_stories_no_positives():
    stories = [{"id": 1, "score": 10, "time": 1000}]
    # Should work without crashing even if positive_embeddings is None
    res = rerank.rank_stories(stories, cand_embeddings=np.array([[1.0, 0.0]]))
    assert len(res) == 1

def test_rank_stories_diversity():
    stories = [
        {"id": 1, "score": 100, "time": 1000, "text_content": "A"},
        {"id": 2, "score": 100, "time": 1000, "text_content": "B"}
    ]
    cand_emb = np.array([[1.0, 0.0], [0.99, 0.01]]) # Very similar
    pos_emb = np.array([[1.0, 0.0]])

    # High diversity penalty should favor different items (though here they are both similar to pos)
    res = rerank.rank_stories(stories, cand_embeddings=cand_emb, positive_embeddings=pos_emb, diversity_lambda=0.8)
    assert len(res) == 2

def test_cluster_and_reduce_single():
    emb = np.array([[1.0, 0.0]])
    c, r, labels = rerank.cluster_and_reduce_auto(emb)
    assert len(c) == 1
    assert r == [0]
    assert labels == [0]
