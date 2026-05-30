from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest

from api.models import Story
from api.rerank import rank_stories
from api.config import AppConfig, SemanticConfig
from helpers import unit_rows


def test_cluster_max_and_knn_score_components(
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
        semantic=SemanticConfig(knn_neighbors=3),
    )
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
    ):
        mock_cluster.return_value = (
            centroids,
            np.array([0, 0, 1, 1], dtype=np.int32),
        )
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            config=config,
        )

    assert [r.model_score for r in results] == sorted(
        [r.model_score for r in results], reverse=True
    )
    by_index = {result.index: result for result in results}
    assert by_index[0].max_cluster_score == pytest.approx(1.0)
    assert by_index[0].knn_score == pytest.approx(1.0)
    assert by_index[0].model_score == pytest.approx(1.0)

    assert by_index[1].model_score == pytest.approx(1.0)
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
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
    ):
        mock_cluster.return_value = (
            centroids,
            np.array([0, 0, 0, 1, 1], dtype=np.int32),
        )

        from api.config import ClassifierConfig

        config = AppConfig(classifier=ClassifierConfig(raw_embedding_features=True))
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

    config = AppConfig()
    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.cluster_interests_with_labels") as mock_cluster,
    ):
        mock_cluster.return_value = (
            centroids,
            np.array([0, 1], dtype=np.int32),
        )
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

    config = AppConfig()
    with patch("api.rerank.get_embeddings", return_value=cand_emb):
        results = rank_stories(stories, pos_emb, config=config)

    assert len(results) == 2
    assert {result.index for result in results} == {0, 1}
    assert all(r.model_score == pytest.approx(1.0) for r in results)
