import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from api.rerank import rank_stories
import api.rerank as rerank
from api.models import Story
from api.config import AppConfig, ClassifierConfig, CrossEncoderConfig, RankingConfig


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


def test_rank_stories_progress_completes_after_cross_encoder(monkeypatch):
    stories = _make_stories(2)
    pos_emb = np.ones((1, 768), dtype=np.float32)
    cand_emb = np.vstack(
        [
            np.ones(768, dtype=np.float32),
            np.full(768, 0.5, dtype=np.float32),
        ]
    )
    events = []

    class FakeCrossEncoder:
        def score(self, candidates, queries):
            return np.array([0.5], dtype=np.float32)

    monkeypatch.setattr("api.rerank.get_embeddings", lambda *args, **kwargs: cand_emb)
    monkeypatch.setattr("api.rerank.cluster_interests", lambda *args, **kwargs: pos_emb)
    monkeypatch.setattr("api.rerank.init_ce_model", lambda *args, **kwargs: FakeCrossEncoder())

    rank_stories(
        stories,
        pos_emb,
        config=AppConfig(
            use_classifier=False,
            cross_encoder=CrossEncoderConfig(enabled=True, top_n=2),
        ),
        positive_stories=stories[:1],
        progress_callback=events.append,
    )

    phases = [event["phase"] for event in events]
    cross_encoder_done = next(
        i
        for i, event in enumerate(events)
        if event["phase"] == "cross_encoder" and event["current"] == 2
    )
    assert phases.index("complete") > cross_encoder_done


def test_cross_encoder_score_cache_reuses_prior_score(monkeypatch, tmp_path):
    calls = []

    class FakeCrossEncoder:
        def score(self, candidates, queries):
            calls.append((list(candidates), list(queries)))
            return np.array([0.5, 0.8], dtype=np.float32)

    monkeypatch.setattr(rerank, "CROSS_ENCODER_SCORE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "evict_old_cache_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rerank,
        "_cross_encoder_model_fingerprint",
        lambda model_dir: model_dir,
    )

    kwargs = {
        "ce_model": FakeCrossEncoder(),
        "model_dir": "model-a",
        "candidate_text": "candidate",
        "queries_text": ["query a", "query b"],
    }

    assert rerank._score_cross_encoder_candidate(**kwargs) == pytest.approx(0.8)
    assert rerank._score_cross_encoder_candidate(**kwargs) == pytest.approx(0.8)
    assert len(calls) == 1


def test_cross_encoder_score_cache_misses_when_queries_change(monkeypatch, tmp_path):
    calls = []

    class FakeCrossEncoder:
        def score(self, candidates, queries):
            calls.append(list(queries))
            return np.array([float(len(calls))], dtype=np.float32)

    monkeypatch.setattr(rerank, "CROSS_ENCODER_SCORE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "evict_old_cache_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rerank,
        "_cross_encoder_model_fingerprint",
        lambda model_dir: model_dir,
    )
    ce_model = FakeCrossEncoder()

    first = rerank._score_cross_encoder_candidate(
        ce_model=ce_model,
        model_dir="model-a",
        candidate_text="candidate",
        queries_text=["query a"],
    )
    second = rerank._score_cross_encoder_candidate(
        ce_model=ce_model,
        model_dir="model-a",
        candidate_text="candidate",
        queries_text=["query b"],
    )

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(2.0)
    assert calls == [["query a"], ["query b"]]


def test_cross_encoder_score_cache_ignores_corrupt_file(monkeypatch, tmp_path):
    calls = []

    class FakeCrossEncoder:
        def score(self, candidates, queries):
            calls.append(1)
            return np.array([0.7], dtype=np.float32)

    monkeypatch.setattr(rerank, "CROSS_ENCODER_SCORE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(rerank, "evict_old_cache_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rerank,
        "_cross_encoder_model_fingerprint",
        lambda model_dir: model_dir,
    )
    cache_key = rerank._cross_encoder_cache_key(
        model_dir="model-a",
        candidate_text="candidate",
        queries_text=["query a"],
    )
    (tmp_path / f"{cache_key}.json").write_text("{not json")

    assert rerank._score_cross_encoder_candidate(
        ce_model=FakeCrossEncoder(),
        model_dir="model-a",
        candidate_text="candidate",
        queries_text=["query a"],
    ) == pytest.approx(0.7)
    assert calls == [1]


def test_rank_stories_with_classifier_pairwise():
    """Test that pairwise logistic path is taken when configured."""
    stories = _make_stories(10)

    pos_emb = np.random.rand(5, 768).astype(np.float32)
    neg_emb = np.random.rand(5, 768).astype(np.float32)
    cand_emb = np.random.rand(10, 768).astype(np.float32)

    config = AppConfig(
        use_classifier=True,
        classifier=ClassifierConfig(
            scoring_mode="pairwise_logistic",
            feature_mode="bottleneck",
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=True,
        ),
    )
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegression") as mock_lr_class,
    ):
        mock_cluster.return_value = (np.zeros((2, 768)), np.array([0, 0, 0, 1, 1]))
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.coef_ = np.zeros((1, 3))  # 3 derived features
        
        results = rank_stories(stories, pos_emb, neg_emb, config=config)

        mock_lr_class.assert_called_once()
        mock_clf.fit.assert_called_once()
        
        # In bottleneck mode with 3 derived features, input should be (N, 3)
        args, _ = mock_clf.fit.call_args
        X_pairwise = args[0]
        assert X_pairwise.shape[1] == 3
        
        assert len(results) == 10


def test_rank_stories_with_classifier_cv():
    """Test that LogisticRegressionCV path is taken when configured."""
    stories = _make_stories(10)

    pos_emb = np.random.rand(5, 768).astype(np.float32)
    neg_emb = np.random.rand(5, 768).astype(np.float32)
    cand_emb = np.random.rand(10, 768).astype(np.float32)

    config = AppConfig(
        use_classifier=True,
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="full",
        ),
    )
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        mock_cluster.return_value = (np.zeros((2, 768)), np.array([0, 0, 0, 1, 1]))
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.zeros((10, 2))
        mock_clf.predict_proba.return_value[:, 1] = np.linspace(0, 1, 10)

        results = rank_stories(stories, pos_emb, neg_emb, config=config)

        mock_lr_class.assert_called_once()
        mock_clf.fit.assert_called_once()

        assert len(results) == 10


def test_rank_stories_classifier_fallback():
    """Test that it falls back to heuristic if not enough data."""
    stories = [Story(id=1, title="Test", url=None, score=0, time=0, text_content="A")]
    pos_emb = np.random.rand(1, 768).astype(np.float32)
    neg_emb = np.random.rand(1, 768).astype(np.float32)
    cand_emb = np.random.rand(1, 768).astype(np.float32)

    config = AppConfig(use_classifier=True)
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        results = rank_stories(stories, pos_emb, neg_emb, config=config)
        mock_lr_class.assert_not_called()
        assert len(results) == 1


def test_rank_stories_populates_classifier_diagnostics():
    stories = _make_stories(3)

    pos_emb = np.random.rand(5, 8).astype(np.float32)
    neg_emb = np.random.rand(5, 8).astype(np.float32)
    cand_emb = np.random.rand(3, 8).astype(np.float32)
    diagnostics: dict[str, object] = {}

    config = AppConfig(
        use_classifier=True,
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="full",
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=False,
        ),
    )
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
    ):
        mock_cluster.return_value = (np.zeros((2, 8)), np.array([0, 0, 0, 1, 1]))
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.column_stack(
            [
                np.linspace(1, 0, 3),
                np.linspace(0, 1, 3),
            ]
        )

        rank_stories(
            stories,
            pos_emb,
            neg_emb,
            config=config,
            diagnostics=diagnostics,
        )

    assert diagnostics["classifier_requested"] is True
    assert diagnostics["classifier_used"] is True
    assert diagnostics["classifier_failure_reason"] is None
    assert diagnostics["positive_count"] == 5
    assert diagnostics["negative_count"] == 5
    assert diagnostics["base_feature_dim"] == 8
    assert diagnostics["derived_feature_dim"] == 2
    assert diagnostics["classifier_metadata_features_used"] is False
    assert diagnostics["classifier_metadata_feature_dim"] == 0
    assert diagnostics["local_hidden_penalty_applied"] is False


def test_rank_stories_reports_insufficient_examples_diagnostics():
    stories = [Story(id=1, title="Test", url=None, score=0, time=0, text_content="A")]
    pos_emb = np.random.rand(1, 8).astype(np.float32)
    neg_emb = np.random.rand(1, 8).astype(np.float32)
    cand_emb = np.random.rand(1, 8).astype(np.float32)
    diagnostics: dict[str, object] = {}

    config = AppConfig(use_classifier=True)
    with patch("api.rerank.get_embeddings", return_value=cand_emb):
        rank_stories(
            stories,
            pos_emb,
            neg_emb,
            config=config,
            diagnostics=diagnostics,
        )

    assert diagnostics["classifier_requested"] is True
    assert diagnostics["classifier_used"] is False
    assert diagnostics["classifier_failure_reason"] == "insufficient_examples"


def test_classifier_probability_is_semantic_score_without_post_hidden_penalty():
    stories = [Story(id=1, title="Test", url=None, score=0, time=0, text_content="A")]
    rng = np.random.default_rng(123)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = neg_emb[:1].copy()
    diagnostics: dict[str, object] = {}

    probs = np.array([0.8], dtype=np.float32)

    config = AppConfig(
        use_classifier=True, 
        ranking=RankingConfig(non_semantic_weight=0.0),
        classifier=ClassifierConfig(scoring_mode="logistic_cv", feature_mode="full")
    )
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
            config=config,
            diagnostics=diagnostics,
        )

    assert results[0].semantic_score == pytest.approx(0.8, abs=1e-6)
    assert diagnostics["local_hidden_penalty_applied"] is False
    assert diagnostics["local_hidden_penalty_mean"] == 0.0
    assert diagnostics["local_hidden_penalty_max"] == 0.0


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

    config = AppConfig(
        use_classifier=True,
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="full",
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=True,
        ),
    )
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
        mock_clf.predict_proba.return_value = np.column_stack(
            [
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
            ]
        )

        rank_stories(stories, pos_emb, neg_emb, config=config)

        # Check X_train shape: 10 samples, dim+3 features
        args, kwargs = mock_clf.fit.call_args
        X_train = args[0]
        assert X_train.shape == (10, 8 + 3), f"Expected (10, 11), got {X_train.shape}"

        # Check candidate features shape
        cand_arg = mock_clf.predict_proba.call_args[0][0]
        assert cand_arg.shape == (10, 8 + 3), f"Expected (10, 11), got {cand_arg.shape}"

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
        config = AppConfig(
            use_classifier=True,
            classifier=ClassifierConfig(
                scoring_mode="logistic_cv",
                feature_mode="full",
                k_feat=k_feat_val,
                use_centroid_feature=True,
                use_pos_knn_feature=True,
                use_neg_knn_feature=True,
            ),
        )
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
            mock_clf.predict_proba.return_value = np.column_stack(
                [
                    np.linspace(1, 0, 10),
                    np.linspace(0, 1, 10),
                ]
            )

            rank_stories(stories, pos_emb, neg_emb, config=config)

            args, _ = mock_clf.fit.call_args
            fit_X_trains.append(args[0].copy())

    # Derived features (last 3 cols) should differ when k_feat changes
    derived_1 = fit_X_trains[0][:, 8:]
    derived_5 = fit_X_trains[1][:, 8:]
    assert not np.allclose(derived_1, derived_5), (
        "Derived features should change when CLASSIFIER_K_FEAT changes"
    )


def test_classifier_negative_sample_weights_are_not_tuned():
    """Negative rows use unit sample weights; hidden similarity is a feature, not a knob."""
    stories = _make_stories(10)

    rng = np.random.default_rng(42)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    config = AppConfig(
        use_classifier=True,
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="full",
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=True,
        ),
    )
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
        mock_clf.predict_proba.return_value = np.column_stack(
            [
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
            ]
        )

        rank_stories(stories, pos_emb, neg_emb, config=config)

        _, kwargs = mock_clf.fit.call_args
        sample_weight = kwargs["sample_weight"]
        assert np.allclose(sample_weight[-len(neg_emb) :], 1.0)


def test_classifier_feature_ablation_disables_all_derived_columns():
    stories = _make_stories(10)

    rng = np.random.default_rng(7)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb /= np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-9
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    config = AppConfig(
        use_classifier=True,
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="full",
            use_centroid_feature=False,
            use_pos_knn_feature=False,
            use_neg_knn_feature=False,
        ),
    )
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
        mock_clf.predict_proba.return_value = np.column_stack(
            [
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
            ]
        )

        rank_stories(stories, pos_emb, neg_emb, config=config)

        fit_args, _ = mock_clf.fit.call_args
        X_train = fit_args[0]
        cand_arg = mock_clf.predict_proba.call_args[0][0]
        assert X_train.shape == (10, 8)
        assert cand_arg.shape == (10, 8)


def test_classifier_metadata_features_append_log_points_only():
    stories = [
        Story(
            id=i,
            title=f"Story {i}",
            url=None,
            score=i * 10,
            time=1000,
            text_content=f"Story {i}",
            comment_count=i,
        )
        for i in range(10)
    ]
    pos_stories = [
        Story(
            id=100 + i,
            title=f"Pos {i}",
            url=None,
            score=100 + i,
            time=1000,
            text_content=f"Pos {i}",
            comment_count=10 + i,
        )
        for i in range(5)
    ]
    neg_stories = [
        Story(
            id=200 + i,
            title=f"Neg {i}",
            url=None,
            score=20 + i,
            time=1000,
            text_content=f"Neg {i}",
            comment_count=2 + i,
        )
        for i in range(5)
    ]

    rng = np.random.default_rng(8)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    diagnostics: dict[str, object] = {}

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
        mock_clf.predict_proba.return_value = np.column_stack(
            [
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
            ]
        )

        config = AppConfig(
            use_classifier=True,
            classifier=ClassifierConfig(
                scoring_mode="logistic_cv",
                feature_mode="full",
                use_centroid_feature=False,
                use_pos_knn_feature=False,
                use_neg_knn_feature=False,
                use_log_points_feature=True,
            ),
        )
        rank_stories(
            stories,
            pos_emb,
            neg_emb,
            config=config,
            diagnostics=diagnostics,
            positive_stories=pos_stories,
            negative_stories=neg_stories,
        )

        fit_args, _ = mock_clf.fit.call_args
        X_train = fit_args[0]
        cand_arg = mock_clf.predict_proba.call_args[0][0]
        assert X_train.shape == (10, 9)
        assert cand_arg.shape == (10, 9)
        assert np.all(X_train[:, -1:] >= 0.0)
        assert np.all(X_train[:, -1:] <= 1.0)
        assert diagnostics["classifier_metadata_features_used"] is True
        assert diagnostics["classifier_metadata_feature_dim"] == 1


def test_classifier_metadata_features_skip_when_training_metadata_mismatches():
    stories = _make_stories(10)
    pos_stories = _make_stories(4)  # Mismatch (X_pos is 5)
    neg_stories = _make_stories(4)  # Mismatch (X_neg is 5)

    rng = np.random.default_rng(9)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    cand_emb = rng.normal(size=(10, 8)).astype(np.float32)
    diagnostics: dict[str, object] = {}

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
        mock_clf.predict_proba.return_value = np.column_stack(
            [
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
            ]
        )

        config = AppConfig(
            use_classifier=True,
            classifier=ClassifierConfig(
                scoring_mode="logistic_cv",
                feature_mode="full",
                use_centroid_feature=False,
                use_pos_knn_feature=False,
                use_neg_knn_feature=False,
                use_log_points_feature=True,
            ),
        )
        rank_stories(
            stories,
            pos_emb,
            neg_emb,
            config=config,
            diagnostics=diagnostics,
            positive_stories=pos_stories,
            negative_stories=neg_stories,
        )

        fit_args, _ = mock_clf.fit.call_args
        X_train = fit_args[0]
        cand_arg = mock_clf.predict_proba.call_args[0][0]
        # It has 9 columns because _classifier_metadata_features returned 
        # a zero column due to size mismatch.
        assert X_train.shape == (10, 9)
        assert cand_arg.shape == (10, 9)
        assert np.all(X_train[:, -1] == 0)
        assert diagnostics["classifier_metadata_features_used"] is False
        assert diagnostics["classifier_metadata_feature_dim"] == 1

def test_classifier_metadata_features_still_allow_tuned_hn_blend():
    stories = [
        Story(
            id=1,
            title="Relevant low points",
            url=None,
            score=1,
            time=1000,
            text_content="A",
            comment_count=1,
        ),
        Story(
            id=2,
            title="Less relevant high points",
            url=None,
            score=10000,
            time=1000,
            text_content="B",
            comment_count=1000,
        ),
    ]
    pos_stories = _make_stories(5)
    neg_stories = _make_stories(5)
    for story in pos_stories:
        story.comment_count = 10
    for story in neg_stories:
        story.comment_count = 1

    rng = np.random.default_rng(10)
    pos_emb = rng.normal(size=(5, 8)).astype(np.float32)
    neg_emb = rng.normal(size=(5, 8)).astype(np.float32)
    cand_emb = rng.normal(size=(2, 8)).astype(np.float32)

    from api.config import AdaptiveHNConfig, CrossEncoderConfig, FreshnessConfig
    config = AppConfig(
        use_classifier=True,
        ranking=RankingConfig(non_semantic_weight=1.0),
        adaptive_hn=AdaptiveHNConfig(weight_min=1.0, weight_max=1.0),
        cross_encoder=CrossEncoderConfig(enabled=False),
        freshness=FreshnessConfig(enabled=False),
        classifier=ClassifierConfig(
            scoring_mode="logistic_cv",
            feature_mode="full",
            use_centroid_feature=False,
            use_pos_knn_feature=False,
            use_neg_knn_feature=False,
        ),
    )
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
        probs = np.array([0.6, 0.5], dtype=np.float32)
        mock_clf.predict_proba.return_value = np.column_stack([1.0 - probs, probs])

        results = rank_stories(
            stories,
            pos_emb,
            neg_emb,
            config=config,
            positive_stories=pos_stories,
            negative_stories=neg_stories,
        )

    assert [r.index for r in results] == [1, 0]
    assert results[0].hybrid_score > 0.6


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

    config = AppConfig(
        use_classifier=True, 
        ranking=RankingConfig(non_semantic_weight=0.0),
        classifier=ClassifierConfig(scoring_mode="logistic_cv", feature_mode="full")
    )
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
            config=config,
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

    def _run_cv(parallel: bool) -> dict[str, float]:
        evaluator = RankingEvaluator("test_user")
        evaluator.dataset = EvaluationDataset(
            train_stories=stories[:8],
            test_stories=stories[8:],
            neg_stories=[],
            candidates=stories,
            train_embeddings=emb[:8],
            neg_embeddings=neg_emb,
            test_ids={8, 9},
        )

        with (
            patch("evaluate_quality.get_embeddings", return_value=emb),
            patch("evaluate_quality.rank_stories") as mock_rank,
        ):
            mock_rank.return_value = [
                RankResult(
                    index=i,
                    hybrid_score=1.0 - i * 0.1,
                    best_fav_index=0,
                    max_sim_score=0.5,
                    knn_score=0.5,
                )
                for i in range(10)
            ]

            # Fix the random seed for deterministic folds
            np.random.seed(123)
            return evaluator.evaluate_cv(
                n_folds=3,
                config=AppConfig(),
                k_metrics=[10],
                report_each=False,
                parallel=parallel,
            )

    serial = _run_cv(parallel=False)
    parallel = _run_cv(parallel=True)

    for key in serial:
        assert serial[key] == pytest.approx(parallel[key], abs=1e-9), (
            f"Mismatch on {key}: serial={serial[key]}, parallel={parallel[key]}"
        )


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
        test_ids={50},
    )

    with patch("evaluate_quality.get_embeddings") as mock_emb:
        mock_emb.return_value = rng.normal(size=(9, 8)).astype(np.float32)
        with patch("evaluate_quality.rank_stories") as mock_rank:
            # Return results that include the hidden story (index 5 = hidden_story)
            mock_rank.return_value = [
                RankResult(
                    index=i,
                    hybrid_score=1.0 - i * 0.1,
                    best_fav_index=0,
                    max_sim_score=0.5,
                    knn_score=0.5,
                )
                for i in range(len(candidates_with_leak) + 1)  # +1 for injected test
            ]

            # The assertion inside evaluate_cv should catch the leak
            with pytest.raises(AssertionError, match="Hidden stories leaked"):
                evaluator.evaluate_cv(
                    n_folds=2,
                    config=AppConfig(),
                    k_metrics=[10],
                    report_each=False,
                )
