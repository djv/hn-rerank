import numpy as np
from unittest.mock import patch, MagicMock
from api.rerank import rank_stories
from api.models import Story

def _make_stories(n: int) -> list[Story]:
    return [
        Story(id=i, title=f"S{i}", url=None, score=100, time=1000, text_content=f"S{i}")
        for i in range(n)
    ]

def test_max_cluster_score_populated_in_classifier_path():
    """Ensure max_cluster_score is non-zero when using the classifier."""
    stories = _make_stories(5)
    
    # 5 positives, 5 negatives to trigger classifier
    pos_emb = np.random.rand(5, 768).astype(np.float32)
    neg_emb = np.random.rand(5, 768).astype(np.float32)
    cand_emb = np.random.rand(5, 768).astype(np.float32)
    
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
    ):
        # Mock clustering to return 2 non-zero centroids
        mock_cluster.return_value = (np.random.rand(2, 768).astype(np.float32), np.array([0,0,0,1,1]))
        
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.zeros((5, 2))
        mock_clf.predict_proba.return_value[:, 1] = 0.5
        
        results = rank_stories(stories, pos_emb, neg_emb, use_classifier=True)
        
        assert len(results) == 5
        for r in results:
            # max_cluster_score should be populated and non-zero
            assert r.max_cluster_score > 0.0, f"Result {r.index} has 0% cluster match"
            # It should also be reasonable (cosine sim usually > 0 for random positive vectors)
            assert 0.0 <= r.max_cluster_score <= 1.0

def test_max_cluster_score_populated_in_heuristic_path():
    """Ensure max_cluster_score is non-zero when using heuristic fallback."""
    stories = _make_stories(5)
    
    # Not enough data for classifier
    pos_emb = np.random.rand(2, 768).astype(np.float32)
    cand_emb = np.random.rand(5, 768).astype(np.float32)
    
    with patch("api.rerank.get_embeddings", return_value=cand_emb):
        results = rank_stories(stories, pos_emb, use_classifier=False)
        
        assert len(results) == 5
        for r in results:
            assert r.max_cluster_score > 0.0, f"Result {r.index} has 0% cluster match"
