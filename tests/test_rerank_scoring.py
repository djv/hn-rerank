from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.models import Story
from api.rerank import rank_stories
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

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests", return_value=centroids),
        patch("api.rerank.SEMANTIC_MAXSIM_WEIGHT", 0.5),
        patch("api.rerank.SEMANTIC_MEANSIM_WEIGHT", 0.5),
    ):
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            hn_weight=0.0,
            knn_k=3,
            use_classifier=False,
        )

    by_index = {result.index: result for result in results}
    assert by_index[0].max_cluster_score == pytest.approx(1.0)
    assert by_index[0].knn_score == pytest.approx(1.0)
    assert by_index[0].semantic_score == pytest.approx(1.0)

    assert by_index[1].max_cluster_score == pytest.approx(1.0)
    assert by_index[1].knn_score == pytest.approx(0.0)
    assert by_index[1].semantic_score == pytest.approx(0.5)


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

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests", return_value=centroids),
        patch("api.rerank.SEMANTIC_MAXSIM_WEIGHT", 1.0),
        patch("api.rerank.SEMANTIC_MEANSIM_WEIGHT", 0.0),
    ):
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            hn_weight=0.0,
            knn_k=3,
            use_classifier=False,
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

        results = rank_stories(stories, pos_emb, neg_emb, use_classifier=True)

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

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests", return_value=centroids),
    ):
        results = rank_stories(
            stories,
            pos_emb,
            hn_weight=0.0,
            use_classifier=False,
        )

    assert [result.max_cluster_score for result in results] == pytest.approx([1.0, 1.0])
