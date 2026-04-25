import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from api.rerank import rank_stories
from api.models import Story


def _make_stories(n: int) -> list[Story]:
    return [
        Story(
            id=i,
            title=f"Story {i}",
            url=None,
            score=100,
            time=1000,
            text_content=f"Story {i}",
        )
        for i in range(n)
    ]


def test_rank_stories_with_classifier():
    """Test that classifier path is taken when flag is set and enough data exists."""
    stories = _make_stories(10)

    pos_emb = np.random.rand(5, 768).astype(np.float32)
    neg_emb = np.random.rand(5, 768).astype(np.float32)
    cand_emb = np.random.rand(10, 768).astype(np.float32)

    with (
        patch("api.rerank.CLASSIFIER_USE_CENTROID_FEATURE", True),
        patch("api.rerank.CLASSIFIER_USE_POS_KNN_FEATURE", True),
        patch("api.rerank.CLASSIFIER_USE_NEG_KNN_FEATURE", True),
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        mock_cluster.return_value = (np.zeros((2, 768)), np.array([0, 0, 0, 1, 1]))
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.zeros((10, 2))
        mock_clf.predict_proba.return_value[:, 1] = np.linspace(0, 1, 10)

        results = rank_stories(stories, pos_emb, neg_emb, use_classifier=True)

        mock_lr_class.assert_called_once()
        mock_clf.fit.assert_called_once()
        mock_clf.predict_proba.assert_called_once()

        # X_train = 5 pos + 5 neg = 10 samples
        args, kwargs = mock_clf.fit.call_args
        assert "sample_weight" in kwargs
        assert len(kwargs["sample_weight"]) == 10

        assert len(results) == 10


def test_rank_stories_classifier_fallback():
    """Test that it falls back to heuristic if not enough data."""
    stories = [Story(id=1, title="Test", url=None, score=0, time=0, text_content="A")]
    pos_emb = np.random.rand(1, 768).astype(np.float32)
    neg_emb = np.random.rand(1, 768).astype(np.float32)
    cand_emb = np.random.rand(1, 768).astype(np.float32)

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        results = rank_stories(stories, pos_emb, neg_emb, use_classifier=True)
        mock_lr_class.assert_not_called()
        assert len(results) == 1


def test_rank_stories_populates_classifier_diagnostics():
    stories = _make_stories(3)

    pos_emb = np.random.rand(5, 8).astype(np.float32)
    neg_emb = np.random.rand(5, 8).astype(np.float32)
    cand_emb = np.random.rand(3, 8).astype(np.float32)
    diagnostics: dict[str, object] = {}

    with (
        patch("api.rerank.CLASSIFIER_USE_CENTROID_FEATURE", True),
        patch("api.rerank.CLASSIFIER_USE_POS_KNN_FEATURE", True),
        patch("api.rerank.CLASSIFIER_USE_NEG_KNN_FEATURE", False),
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        mock_cluster.return_value = (np.zeros((2, 8)), np.array([0, 0, 0, 1, 1]))
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack([
            np.linspace(1, 0, 3),
            np.linspace(0, 1, 3),
        ])

        rank_stories(
            stories,
            pos_emb,
            neg_emb,
            use_classifier=True,
            diagnostics=diagnostics,
        )

    assert diagnostics["classifier_requested"] is True
    assert diagnostics["classifier_used"] is True
    assert diagnostics["classifier_failure_reason"] is None
    assert diagnostics["positive_count"] == 5
    assert diagnostics["negative_count"] == 5
    assert diagnostics["base_feature_dim"] == 8
    assert diagnostics["derived_feature_dim"] == 2
    assert diagnostics["local_hidden_penalty_applied"] is False


def test_rank_stories_reports_insufficient_examples_diagnostics():
    stories = [Story(id=1, title="Test", url=None, score=0, time=0, text_content="A")]
    pos_emb = np.random.rand(1, 8).astype(np.float32)
    neg_emb = np.random.rand(1, 8).astype(np.float32)
    cand_emb = np.random.rand(1, 8).astype(np.float32)
    diagnostics: dict[str, object] = {}

    with patch("api.rerank.get_embeddings", return_value=cand_emb):
        rank_stories(
            stories,
            pos_emb,
            neg_emb,
            use_classifier=True,
            diagnostics=diagnostics,
        )

    assert diagnostics["classifier_requested"] is True
    assert diagnostics["classifier_used"] is False
    assert diagnostics["classifier_failure_reason"] == "insufficient_examples"


def test_classifier_local_hidden_penalty_reduces_semantic_scores():
    stories = [Story(id=1, title="Test", url=None, score=0, time=0, text_content="A")]
    rng = np.random.default_rng(123)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = neg_emb[:1].copy()
    diagnostics: dict[str, object] = {}

    probs = np.array([0.8], dtype=np.float32)

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
        patch("api.rerank.CLASSIFIER_USE_LOCAL_HIDDEN_PENALTY", True),
        patch("api.rerank.CLASSIFIER_LOCAL_HIDDEN_PENALTY_WEIGHT", 0.5),
        patch("api.rerank.CLASSIFIER_LOCAL_HIDDEN_PENALTY_K", 1),
    ):
        mock_cluster.return_value = (
            np.zeros((2, 8)),
            np.array([0, 0, 0, 1, 1]),
        )
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack([1.0 - probs, probs])

        results = rank_stories(
            stories,
            pos_emb,
            neg_emb,
            use_classifier=True,
            hn_weight=0.0,
            diagnostics=diagnostics,
        )

    assert results[0].semantic_score == pytest.approx(0.3, abs=1e-6)
    assert diagnostics["local_hidden_penalty_applied"] is True
    assert diagnostics["local_hidden_penalty_mean"] == pytest.approx(0.5, abs=1e-6)
    assert diagnostics["local_hidden_penalty_max"] == pytest.approx(0.5, abs=1e-6)


def test_logistic_regression_cv_selects_C():
    """LogisticRegressionCV auto-selects best C from provided grid."""
    from sklearn.linear_model import LogisticRegressionCV

    rng = np.random.default_rng(42)
    # Linearly separable data
    X_pos = rng.normal(loc=2.0, size=(30, 4)).astype(np.float32)
    X_neg = rng.normal(loc=-2.0, size=(30, 4)).astype(np.float32)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(30), np.zeros(30)])

    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        class_weight="balanced",
        solver="liblinear",
        scoring="f1",
        l1_ratios=(0.0,),
        use_legacy_attributes=False,
    )
    clf.fit(X, y)

    # C_ should be auto-selected (not necessarily 1.0)
    assert hasattr(clf, "C_")
    selected_c = float(np.atleast_1d(clf.C_)[0])
    assert selected_c in [0.01, 0.1, 1.0, 10.0]
    # Should fit well on linearly separable data
    assert clf.score(X, y) > 0.95


def test_classifier_feature_augmentation_shape():
    """Classifier training with augmented features produces correct shape (N, dim+3)."""
    stories = _make_stories(10)

    # 5 pos (dim=8), 5 neg
    rng = np.random.default_rng(42)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    with (
        patch("api.rerank.CLASSIFIER_USE_CENTROID_FEATURE", True),
        patch("api.rerank.CLASSIFIER_USE_POS_KNN_FEATURE", True),
        patch("api.rerank.CLASSIFIER_USE_NEG_KNN_FEATURE", True),
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        mock_cluster.return_value = (
            np.zeros((2, 8)),
            np.array([0, 0, 0, 1, 1]),
        )

        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack([
            np.linspace(1, 0, 10),
            np.linspace(0, 1, 10),
        ])

        rank_stories(stories, pos_emb, neg_emb, use_classifier=True)

        # Check X_train shape: 10 samples, dim+3 features
        args, kwargs = mock_clf.fit.call_args
        X_train = args[0]
        assert X_train.shape == (10, 8 + 3), (
            f"Expected (10, 11), got {X_train.shape}"
        )

        # Check candidate features shape
        cand_arg = mock_clf.predict_proba.call_args[0][0]
        assert cand_arg.shape == (10, 8 + 3), (
            f"Expected (10, 11), got {cand_arg.shape}"
        )

        # Derived features should be bounded [-1, 1]
        derived_train = X_train[:, 8:]
        assert np.all(derived_train >= -1.0 - 1e-6)
        assert np.all(derived_train <= 1.0 + 1e-6)


def test_classifier_k_feat_independent():
    """k_feat is decoupled from knn_k — changing CLASSIFIER_K_FEAT affects derived features."""
    stories = _make_stories(10)

    rng = np.random.default_rng(42)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    fit_X_trains = []

    for k_feat_val in [1, 5]:
        with (
            patch("api.rerank.CLASSIFIER_K_FEAT", k_feat_val),
            patch("api.rerank.CLASSIFIER_USE_CENTROID_FEATURE", True),
            patch("api.rerank.CLASSIFIER_USE_POS_KNN_FEATURE", True),
            patch("api.rerank.CLASSIFIER_USE_NEG_KNN_FEATURE", True),
            patch("api.rerank.get_embeddings", return_value=cand_emb),
            patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
            patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
        ):
            mock_cluster.return_value = (
                np.zeros((2, 8)),
                np.array([0, 0, 0, 1, 1]),
            )
            mock_clf = MagicMock()
            mock_lr_class.return_value = mock_clf
            mock_clf.predict_proba.return_value = np.column_stack([
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
            ])

            rank_stories(stories, pos_emb, neg_emb, use_classifier=True)

            args, _ = mock_clf.fit.call_args
            fit_X_trains.append(args[0].copy())

    # Derived features (last 3 cols) should differ when k_feat changes
    derived_1 = fit_X_trains[0][:, 8:]
    derived_5 = fit_X_trains[1][:, 8:]
    assert not np.allclose(derived_1, derived_5), (
        "Derived features should change when CLASSIFIER_K_FEAT changes"
    )


def test_classifier_neg_sample_weight_independent():
    """Negative sample weights in classifier training follow CLASSIFIER_NEG_SAMPLE_WEIGHT."""
    stories = _make_stories(10)

    rng = np.random.default_rng(42)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    neg_weights_seen = []

    for neg_w in [0.5, 1.75]:
        with (
            patch("api.rerank.CLASSIFIER_NEG_SAMPLE_WEIGHT", neg_w),
            patch("api.rerank.CLASSIFIER_USE_CENTROID_FEATURE", True),
            patch("api.rerank.CLASSIFIER_USE_POS_KNN_FEATURE", True),
            patch("api.rerank.CLASSIFIER_USE_NEG_KNN_FEATURE", True),
            patch("api.rerank.get_embeddings", return_value=cand_emb),
            patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
            patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
        ):
            mock_cluster.return_value = (
                np.zeros((2, 8)),
                np.array([0, 0, 0, 1, 1]),
            )
            mock_clf = MagicMock()
            mock_lr_class.return_value = mock_clf
            mock_clf.predict_proba.return_value = np.column_stack([
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
            ])

            rank_stories(stories, pos_emb, neg_emb, use_classifier=True)

            _, kwargs = mock_clf.fit.call_args
            sample_weight = kwargs["sample_weight"]
            neg_weights_seen.append(sample_weight[-len(neg_emb):].copy())

    assert np.allclose(neg_weights_seen[0], 0.5)
    assert np.allclose(neg_weights_seen[1], 1.75)
    assert not np.allclose(neg_weights_seen[0], neg_weights_seen[1])


def test_classifier_feature_ablation_disables_all_derived_columns():
    stories = _make_stories(10)

    rng = np.random.default_rng(7)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    with (
        patch("api.rerank.CLASSIFIER_USE_CENTROID_FEATURE", False),
        patch("api.rerank.CLASSIFIER_USE_POS_KNN_FEATURE", False),
        patch("api.rerank.CLASSIFIER_USE_NEG_KNN_FEATURE", False),
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        mock_cluster.return_value = (
            np.zeros((2, 8)),
            np.array([0, 0, 0, 1, 1]),
        )
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack([
            np.linspace(1, 0, 10),
            np.linspace(0, 1, 10),
        ])

        rank_stories(stories, pos_emb, neg_emb, use_classifier=True)

        fit_args, _ = mock_clf.fit.call_args
        X_train = fit_args[0]
        cand_arg = mock_clf.predict_proba.call_args[0][0]
        assert X_train.shape == (10, 8)
        assert cand_arg.shape == (10, 8)


def test_classifier_scores_are_not_post_penalized_by_hidden_similarity():
    stories = _make_stories(10)

    rng = np.random.default_rng(11)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    probs = np.linspace(0.1, 0.9, 10, dtype=np.float32)

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        mock_cluster.return_value = (
            np.zeros((2, 8)),
            np.array([0, 0, 0, 1, 1]),
        )
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack([1.0 - probs, probs])

        results = rank_stories(
            stories,
            pos_emb,
            neg_emb,
            use_classifier=True,
            hn_weight=0.0,
        )

        actual_scores = np.array([r.semantic_score for r in results])
        np.testing.assert_allclose(actual_scores, probs[::-1], atol=1e-6)


def test_evaluate_cv_parallel_matches_serial():
    """Parallel CV (threads) produces same metrics as serial CV."""
    from evaluate_quality import RankingEvaluator, EvaluationDataset
    from api.models import RankResult

    stories = [
        Story(id=i, title=f"S{i}", url=None, score=100, time=1000, text_content=f"S{i}")
        for i in range(10)
    ]
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(10, 8)).astype(np.float32)
    neg_emb = rng.normal(size=(3, 8)).astype(np.float32)
    pos_weights = np.linspace(0.5, 1.0, 10).astype(np.float32)

    def _run_cv(parallel: bool) -> dict[str, float]:
        evaluator = RankingEvaluator("test_user")
        evaluator.dataset = EvaluationDataset(
            train_stories=stories[:8],
            test_stories=stories[8:],
            neg_stories=[],
            candidates=stories,
            train_embeddings=emb[:8],
            neg_embeddings=neg_emb,
            pos_weights=pos_weights,
            test_ids={8, 9},
        )

        with (
            patch("evaluate_quality.get_embeddings", return_value=emb),
            patch("evaluate_quality.rank_stories") as mock_rank,
        ):
            mock_rank.return_value = [
                RankResult(index=i, hybrid_score=1.0 - i * 0.1,
                           best_fav_index=0, max_sim_score=0.5, knn_score=0.5)
                for i in range(10)
            ]

            # Fix the random seed for deterministic folds
            np.random.seed(123)
            return evaluator.evaluate_cv(
                n_folds=3, k_metrics=[10], report_each=False,
                parallel=parallel,
            )

    serial = _run_cv(parallel=False)
    parallel = _run_cv(parallel=True)

    for key in serial:
        assert serial[key] == pytest.approx(parallel[key], abs=1e-9), (
            f"Mismatch on {key}: serial={serial[key]}, parallel={parallel[key]}"
        )


def test_evaluate_cv_passes_positive_weights():
    """evaluate_cv must pass positive_weights to rank_stories like evaluate does."""
    from evaluate_quality import RankingEvaluator, EvaluationDataset

    # Create minimal dataset
    stories = [
        Story(id=i, title=f"S{i}", url=None, score=100, time=1000, text_content=f"S{i}")
        for i in range(10)
    ]
    emb = np.random.rand(10, 8).astype(np.float32)
    neg_emb = np.random.rand(3, 8).astype(np.float32)
    pos_weights = np.linspace(0.5, 1.0, 10).astype(np.float32)

    evaluator = RankingEvaluator("test_user")
    evaluator.dataset = EvaluationDataset(
        train_stories=stories[:8],
        test_stories=stories[8:],
        neg_stories=[],
        candidates=stories,
        train_embeddings=emb[:8],
        neg_embeddings=neg_emb,
        pos_weights=pos_weights,
        test_ids={8, 9},
    )

    with (
        patch("evaluate_quality.get_embeddings", return_value=emb),
        patch("evaluate_quality.rank_stories") as mock_rank,
    ):
        # Return dummy results
        from api.models import RankResult
        mock_rank.return_value = [
            RankResult(index=i, hybrid_score=1.0 - i * 0.1,
                       best_fav_index=0, max_sim_score=0.5, knn_score=0.5)
            for i in range(10)
        ]

        evaluator.evaluate_cv(
            n_folds=2, k_metrics=[10], report_each=False
        )

        # Every call to rank_stories should have positive_weights
        for c in mock_rank.call_args_list:
            assert "positive_weights" in c.kwargs, (
                "evaluate_cv must pass positive_weights to rank_stories"
            )
            pw = c.kwargs["positive_weights"]
            assert pw is not None
            assert len(pw) > 0


def test_hidden_stories_excluded_from_candidates():
    """Hidden (neg) story IDs must be excluded from the eval candidate pool."""
    from evaluate_quality import RankingEvaluator, EvaluationDataset
    from api.models import RankResult

    # Story 99 is hidden — it should never appear in candidates
    hidden_story = Story(
        id=99, title="Hidden", url=None, score=50, time=1000, text_content="Hidden"
    )
    candidates = _make_stories(5)  # ids 0..4
    # Simulate the bug: hidden story sneaks into candidates
    candidates_with_leak = candidates + [hidden_story]

    neg_stories = [hidden_story]
    train_stories = _make_stories(8)  # ids 0..7
    test_stories = [
        Story(id=50, title="Test", url=None, score=100, time=1000, text_content="Test")
    ]

    rng = np.random.default_rng(42)
    train_emb = rng.normal(size=(8, 8)).astype(np.float32)
    neg_emb = rng.normal(size=(1, 8)).astype(np.float32)

    evaluator = RankingEvaluator("test_user")
    evaluator.dataset = EvaluationDataset(
        train_stories=train_stories,
        test_stories=test_stories,
        neg_stories=neg_stories,
        candidates=candidates_with_leak,
        train_embeddings=train_emb,
        neg_embeddings=neg_emb,
        pos_weights=None,
        test_ids={50},
    )

    with patch("evaluate_quality.get_embeddings") as mock_emb:
        mock_emb.return_value = rng.normal(size=(9, 8)).astype(np.float32)
        with patch("evaluate_quality.rank_stories") as mock_rank:
            # Return results that include the hidden story (index 5 = hidden_story)
            mock_rank.return_value = [
                RankResult(
                    index=i, hybrid_score=1.0 - i * 0.1,
                    best_fav_index=0, max_sim_score=0.5, knn_score=0.5,
                )
                for i in range(len(candidates_with_leak) + 1)  # +1 for injected test
            ]

            # The assertion inside evaluate_cv should catch the leak
            with pytest.raises(AssertionError, match="Hidden stories leaked"):
                evaluator.evaluate_cv(
                    n_folds=2, k_metrics=[10], report_each=False,
                )
