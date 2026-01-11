import numpy as np
from unittest.mock import patch, MagicMock
from api.rerank import rank_stories

def test_rank_stories_with_classifier():
    """Test that classifier path is taken when flag is set and enough data exists."""
    stories = [{"id": i, "score": 100, "time": 1000, "text_content": f"Story {i}"} for i in range(10)]
    
    # Create dummy embeddings (5 pos, 5 neg to satisfy threshold)
    pos_emb = np.random.rand(5, 768).astype(np.float32)
    neg_emb = np.random.rand(5, 768).astype(np.float32)
    cand_emb = np.random.rand(10, 768).astype(np.float32)

    with patch("api.rerank.get_embeddings", return_value=cand_emb):
        with patch("api.rerank.cluster_interests_with_labels") as mock_cluster:
            # Mock clustering to return 2 clusters for 5 items
            mock_cluster.return_value = (np.zeros((2, 768)), np.array([0, 0, 0, 1, 1]))
            
            with patch("api.rerank.LogisticRegression") as mock_lr_class:
                mock_clf = MagicMock()
                mock_lr_class.return_value = mock_clf
                # predict_proba returns [prob_0, prob_1]
                mock_clf.predict_proba.return_value = np.zeros((10, 2))
                mock_clf.predict_proba.return_value[:, 1] = np.linspace(0, 1, 10) # fake probs

                results = rank_stories(
                    stories, 
                    pos_emb, 
                    neg_emb, 
                    use_classifier=True
                )

                # Assert classifier was initialized and trained
                mock_lr_class.assert_called_once()
                mock_clf.fit.assert_called_once()
                mock_clf.predict_proba.assert_called_once()

                # Check that sample weights were passed and correct length
                # X_train = 5 pos + 5 neg = 10 samples
                args, kwargs = mock_clf.fit.call_args
                assert "sample_weight" in kwargs
                assert len(kwargs["sample_weight"]) == 10

                # Assert results are returned
                assert len(results) == 10

def test_rank_stories_classifier_fallback():
    """Test that it falls back to heuristic if not enough data."""
    stories = [{"id": 1, "text_content": "A"}]
    # Not enough samples (1 pos, 1 neg)
    pos_emb = np.random.rand(1, 768).astype(np.float32)
    neg_emb = np.random.rand(1, 768).astype(np.float32)
    cand_emb = np.random.rand(1, 768).astype(np.float32)

    with patch("api.rerank.get_embeddings", return_value=cand_emb):
        with patch("api.rerank.LogisticRegression") as mock_lr_class:
            results = rank_stories(
                stories, 
                pos_emb, 
                neg_emb, 
                use_classifier=True
            )
            
            # Should NOT call LogisticRegression because len < 5
            mock_lr_class.assert_not_called()
            assert len(results) == 1
