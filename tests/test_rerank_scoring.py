from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.models import Story
from api.rerank import rank_stories
from api.config import AppConfig, RankingConfig, SemanticConfig
from tests.helpers import unit_rows


def test_semantic_blend_uses_cluster_max_and_knn_components(
    make_stories: Callable[[int], list[Story]],
):
    stories = make_stories(2)
    pos_emb = unit_rows(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    cand_emb = unit_rows(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    centroids = unit_rows([[1.0, 0.0], [0.0, 1.0]])

    config = AppConfig(
        ranking=RankingConfig(non_semantic_weight=0.0),
        semantic=SemanticConfig(maxsim_weight=0.5, meansim_weight=0.5, knn_neighbors=3),
        use_classifier=False,
    )
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests", return_value=centroids),
    ):
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            config=config,
        )

    by_index = {result.index: result for result in results}
    assert by_index[0].max_cluster_score == pytest.approx(1.0)
    assert by_index[0].knn_score == pytest.approx(1.0)
    assert by_index[0].semantic_score == pytest.approx(1.0)

    assert by_index[1].max_cluster_score == pytest.approx(1.0)
    assert by_index[1].knn_score == pytest.approx(0.0)
    # 0.5 input to default sigmoid (k=31.2, t=0.47) is ~0.686
    # k * (x - t) = 31.2 * (0.5 - 0.4749) = 31.2 * 0.0251 = 0.78312
    # sigmoid(0.78312) = 1 / (1 + exp(-0.78312)) = 1 / (1 + 0.4569) = 0.686
    assert by_index[1].semantic_score == pytest.approx(0.686, abs=1e-3)


def test_pure_cluster_max_weight_ignores_diagnostic_knn_score(
    make_stories: Callable[[int], list[Story]],
):
    stories = make_stories(2)
    pos_emb = unit_rows(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    cand_emb = unit_rows([[1.0, 0.0], [0.0, 1.0]])
    centroids = unit_rows([[1.0, 0.0], [0.0, 1.0]])

    config = AppConfig(
        ranking=RankingConfig(non_semantic_weight=0.0),
        semantic=SemanticConfig(maxsim_weight=1.0, meansim_weight=0.0, knn_neighbors=3),
        use_classifier=False,
    )
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests", return_value=centroids),
    ):
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            config=config,
        )

    by_index = {result.index: result for result in results}
    assert by_index[0].semantic_score == pytest.approx(1.0)
    assert by_index[1].semantic_score == pytest.approx(1.0)
    assert by_index[1].knn_score == pytest.approx(0.0)


def test_max_cluster_score_populated_in_classifier_path(
    make_stories: Callable[[int], list[Story]],
):
    stories = make_stories(3)
    pos_emb = unit_rows(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    neg_emb = unit_rows(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, -1.0],
        ]
    )
    cand_emb = unit_rows(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    centroids = unit_rows([[1.0, 0.0], [0.0, 1.0]])

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.LogisticRegressionCV") as mock_lr_class,
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
    ):
        mock_cluster.return_value = (
            centroids,
            np.array([0, 0, 0, 1, 1], dtype=np.int32),
        )
        mock_clf = MagicMock()
        mock_lr_class.return_value = mock_clf
        mock_clf.predict_proba.return_value = np.array(
            [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]], dtype=np.float32
        )

        from api.config import ClassifierConfig
        config = AppConfig(
            use_classifier=True,
            classifier=ClassifierConfig(scoring_mode="logistic_cv", feature_mode="full")
        )
        results = rank_stories(stories, pos_emb, neg_emb, config=config)

    by_index = {result.index: result for result in results}
    assert by_index[0].max_cluster_score == pytest.approx(1.0)
    assert by_index[1].max_cluster_score == pytest.approx(1.0)
    assert by_index[2].max_cluster_score == pytest.approx(0.0)


def test_max_cluster_score_populated_in_heuristic_path(
    make_stories: Callable[[int], list[Story]],
):
    stories = make_stories(2)
    pos_emb = unit_rows([[1.0, 0.0], [0.0, 1.0]])
    cand_emb = unit_rows([[1.0, 0.0], [0.0, 1.0]])
    centroids = unit_rows([[1.0, 0.0], [0.0, 1.0]])

    config = AppConfig(ranking=RankingConfig(non_semantic_weight=0.0), use_classifier=False)
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests", return_value=centroids),
    ):
        results = rank_stories(
            stories,
            pos_emb,
            config=config,
        )

    assert [result.max_cluster_score for result in results] == pytest.approx([1.0, 1.0])


def test_missing_hn_comment_count_is_treated_as_zero():
    stories = [
        Story(
            id=1,
            title="Missing comments",
            url=None,
            score=50,
            time=1000,
            text_content="A",
            comment_count=None,
        ),
        Story(
            id=2,
            title="Has comments",
            url=None,
            score=50,
            time=1000,
            text_content="B",
            comment_count=10,
        ),
    ]
    pos_emb = np.array([[1.0] * 768])
    cand_emb = np.array([[1.0] * 768, [1.0] * 768])

    config = AppConfig(
        ranking=RankingConfig(non_semantic_weight=1.0, comment_ratio=1.0),
        use_classifier=False,
    )
    with patch("api.rerank.get_embeddings", return_value=cand_emb):
        results = rank_stories(stories, pos_emb, config=config)

    assert len(results) == 2
    assert {result.index for result in results} == {0, 1}
