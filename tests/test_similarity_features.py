import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from api.models import Story
from api.config import AppConfig, ClassifierConfig
from api.rerank import rank_stories

def _make_stories(n: int) -> list[Story]:
    return [
        Story(
            id=i,
            title=f"Story {i}",
            url=None,
            score=100,
            time=1000,
            text_content=f"Story {i}")
        for i in range(n)
    ]

def test_similarity_features_only_feature_mode_shape():
    """
    Test that feature_mode = "similarity_only" works and excludes base embeddings,
    and enables all toggled similarity features correctly.
    """
    stories = _make_stories(10)
    pos_emb = np.random.randn(5, 8).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = np.random.randn(5, 8).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = np.random.randn(10, 8).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    # Config with all new similarity features enabled, and base embeddings disabled
    config = AppConfig(
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="similarity_only",
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=True,
            use_closest_pos_feature=True,
            use_closest_neg_feature=True,
            use_closest_centroid_feature=True,
            use_knn_pos_n1_feature=True,
            use_knn_pos_n3_feature=True,
            use_knn_pos_n5_feature=True,
            use_knn_pos_n10_feature=True,
            use_knn_neg_n1_feature=True,
            use_knn_neg_n3_feature=True,
            use_knn_neg_n5_feature=True,
            use_knn_neg_n10_feature=True,
            use_log_points_feature=False,
            use_log_comments_feature=False,
        )
    )

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class
    ):
        mock_cluster.return_value = (
            np.zeros((2, 8)),
            np.array([0, 0, 0, 1, 1], dtype=np.int32)
        )
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack([
            np.linspace(1, 0, 10),
            np.linspace(0, 1, 10)
        ])

        rank_stories(stories, pos_emb, neg_emb, config=config)

        # X_train should have only derived similarity columns, no 8-dim base embeddings!
        # Base: centroid (1), pos_knn (1), neg_knn (1) = 3 columns
        # Rich: closest_pos (1), closest_neg (1), closest_centroid (1) = 3 columns
        # Knn pos: n1, n3, n5, n10 = 4 columns
        # Knn neg: n1, n3, n5, n10 = 4 columns
        # Total = 3 + 3 + 4 + 4 = 14 columns
        fit_args, _ = mock_clf.fit.call_args
        X_train = fit_args[0]
        assert X_train.shape == (10, 14), f"Expected 14 columns, got {X_train.shape[1]}"

        # Assert all derived feature values are within bounded range [-1.0, 1.0]
        assert np.all(X_train >= -1.0 - 1e-6)
        assert np.all(X_train <= 1.0 + 1e-6)

def test_similarity_features_exact_values():
    """
    Test exact similarity calculations for new feature columns.
    """
    stories = _make_stories(2)
    # 2D handcrafted embeddings
    pos_emb = np.array([
        [1.0, 0.0],  # Pos A
        [0.0, 1.0],  # Pos B
        [0.0, 1.0],  # Pos C
    ], dtype=np.float32)
    
    neg_emb = np.array([
        [-1.0, 0.0], # Neg A
        [0.0, -1.0], # Neg B
    ], dtype=np.float32)
    
    cand_emb = np.array([
        [1.0, 0.0],  # Cand 0: perfect match to Pos A
        [0.6, 0.8],  # Cand 1: mixed match
    ], dtype=np.float32)

    config = AppConfig(
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="similarity_only",
            use_centroid_feature=False,
            use_pos_knn_feature=False,
            use_neg_knn_feature=False,
            use_closest_pos_feature=True,
            use_closest_neg_feature=True,
            use_closest_centroid_feature=True,
            use_knn_pos_n1_feature=True,
            use_knn_pos_n3_feature=True,
            use_knn_neg_n1_feature=True,
            use_knn_neg_n3_feature=True,
            use_log_points_feature=False,
            use_log_comments_feature=False,
            min_positive_examples=1,
            min_negative_examples=1,
        )
    )

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class
    ):
        # Cluster positives: Centroids are [1, 0] and [0, 1]
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mock_cluster.return_value = (
            centroids,
            np.array([0, 1, 1], dtype=np.int32)
        )
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack([
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        rank_stories(stories, pos_emb, neg_emb, config=config)

        # Verify candidate features passed to predict_proba
        predict_args = mock_clf.predict_proba.call_args[0][0]
        # Columns stack order in _combine_classifier_features:
        # closest_pos, closest_neg, closest_centroid, knn_pos_n1, knn_pos_n3, knn_neg_n1, knn_neg_n3
        
        # Candidate 0: [1.0, 0.0]
        # - Cos sims to positives: Pos A -> 1.0, Pos B -> 0.0, Pos C -> 0.0
        #   - closest_pos = 1.0
        #   - knn_pos_n1 = mean(top 1) = 1.0
        #   - knn_pos_n3 = mean(top 3) = mean([1.0, 0.0, 0.0]) = 0.33333
        # - Cos sims to negatives: Neg A -> -1.0, Neg B -> 0.0
        #   - closest_neg = 0.0
        #   - knn_neg_n1 = mean(top 1) = 0.0
        #   - knn_neg_n3 = mean(top 2) = mean([-1.0, 0.0]) = -0.5
        # - Cos sims to centroids: C1 [1,0] -> 1.0, C2 [0,1] -> 0.0
        #   - closest_centroid = 1.0
        
        c0_features = predict_args[0]
        assert c0_features[0] == pytest.approx(1.0, abs=1e-5)   # closest_pos
        assert c0_features[1] == pytest.approx(0.0, abs=1e-5)   # closest_neg
        assert c0_features[2] == pytest.approx(1.0, abs=1e-5)   # closest_centroid
        assert c0_features[3] == pytest.approx(1.0, abs=1e-5)   # knn_pos_n1
        assert c0_features[4] == pytest.approx(0.33333, abs=1e-3) # knn_pos_n3
        assert c0_features[5] == pytest.approx(0.0, abs=1e-5)   # knn_neg_n1
        assert c0_features[6] == pytest.approx(-0.5, abs=1e-5)  # knn_neg_n3
