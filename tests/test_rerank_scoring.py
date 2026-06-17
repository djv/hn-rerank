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


def test_local_density_computes_pairwise_mean() -> None:
    """Local density is mean pairwise cosine similarity within the pool."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_local_density

    # 3 candidates: v0=v1 (identical), v2 orthogonal
    cand_emb = unit_rows([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    stories = [
        Story(id=i, title=f"s{i}", url=None, score=0, time=0, text_content="")
        for i in range(3)
    ]

    # Populate cache (as rank_stories would)
    sim = cand_emb @ cand_emb.T
    np.fill_diagonal(sim, 0.0)
    rerank_mod._rank_cache.local_density = (sim.sum(axis=1) / 2).astype(np.float32)
    try:
        out = _meta_local_density(stories, now=0.0)
        assert out.shape == (3, 1)
        # v0: (1.0 + 0.0) / 2 = 0.5
        # v1: same = 0.5
        # v2: (0.0 + 0.0) / 2 = 0.0
        np.testing.assert_allclose(out.flatten(), [0.5, 0.5, 0.0], atol=1e-6)
    finally:
        rerank_mod._rank_cache.local_density = None


def test_local_density_returns_zeros_for_single_candidate() -> None:
    """Single candidate pool should return density of 0.0."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_local_density
    from api.models import Story

    stories = [Story(id=0, title="s", url=None, score=0, time=0, text_content="")]
    rerank_mod._rank_cache.local_density = np.zeros(1, dtype=np.float32)
    try:
        out = _meta_local_density(stories, now=0.0)
        assert out.shape == (1, 1)
        assert out[0, 0] == 0.0
    finally:
        rerank_mod._rank_cache.local_density = None


def test_story_age_uses_cache_for_training_samples() -> None:
    """Cache hit returns log1p(vote-time age)."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_story_age

    now = 86400.0 * 100  # arbitrary "now"
    rerank_mod._rank_cache.story_age_at_vote_map = {
        1: 2.0
    }  # story was age 2 days when voted

    try:
        stories = [Story(id=1, title="s", url=None, score=0, time=0, text_content="")]
        out = _meta_story_age(stories, now=now)
        np.testing.assert_allclose(out[0, 0], float(np.log1p(2.0)), atol=1e-6)
    finally:
        rerank_mod._rank_cache.story_age_at_vote_map = {}


def test_story_age_falls_back_to_current_age_on_cache_miss() -> None:
    """Cache miss uses log1p((now - story.time) / 86400)."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_story_age

    now = 86400.0 * 100
    story_time = now - 5 * 86400  # posted 5 days ago
    rerank_mod._rank_cache.story_age_at_vote_map = {}

    try:
        stories = [
            Story(
                id=99,
                title="s",
                url=None,
                score=0,
                time=int(story_time),
                text_content="",
            )
        ]
        out = _meta_story_age(stories, now=now)
        np.testing.assert_allclose(out[0, 0], float(np.log1p(5.0)), atol=1e-6)
    finally:
        rerank_mod._rank_cache.story_age_at_vote_map = {}


def test_story_age_returns_zero_when_time_missing() -> None:
    """Story.time == 0 returns 0.0 (missing data sentinel)."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_story_age

    rerank_mod._rank_cache.story_age_at_vote_map = {}
    try:
        stories = [Story(id=1, title="s", url=None, score=0, time=0, text_content="")]
        out = _meta_story_age(stories, now=0.0)
        assert out[0, 0] == 0.0
    finally:
        rerank_mod._rank_cache.story_age_at_vote_map = {}


def test_story_age_clamps_negative_age() -> None:
    """Clock skew (updated_at < time) yields age_days=0, not negative log."""
    from api.rerank import _meta_story_age
    import api.rerank as rerank_mod

    rerank_mod._rank_cache.story_age_at_vote_map = {
        1: -1.0
    }  # clock skew -> negative age
    try:
        stories = [
            Story(id=1, title="s", url=None, score=0, time=1000, text_content="")
        ]
        out = _meta_story_age(stories, now=2000.0)
        assert out[0, 0] == 0.0  # log1p(0) = 0
    finally:
        rerank_mod._rank_cache.story_age_at_vote_map = {}


# cluster_size tests


def test_cluster_size_uses_cache() -> None:
    """_meta_cluster_size returns log1p of cluster size from cache."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_cluster_size

    rerank_mod._rank_cache.cluster_size = np.array([3, 3, 1, 3], dtype=np.int32)
    try:
        stories = [
            Story(id=1, title="a", url=None, score=0, time=0, text_content=""),
            Story(id=2, title="b", url=None, score=0, time=0, text_content=""),
            Story(id=3, title="c", url=None, score=0, time=0, text_content=""),
            Story(id=4, title="d", url=None, score=0, time=0, text_content=""),
        ]
        out = _meta_cluster_size(stories, now=0.0)
        expected = np.array(
            [[np.log1p(3)], [np.log1p(3)], [np.log1p(1)], [np.log1p(3)]]
        )
        np.testing.assert_allclose(out, expected, atol=1e-6)
    finally:
        rerank_mod._rank_cache.cluster_size = None


def test_cluster_size_single_candidate() -> None:
    """Single candidate returns log1p(1)."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_cluster_size

    rerank_mod._rank_cache.cluster_size = np.array([1], dtype=np.int32)
    try:
        stories = [Story(id=1, title="a", url=None, score=0, time=0, text_content="")]
        out = _meta_cluster_size(stories, now=0.0)
        np.testing.assert_allclose(out[0, 0], float(np.log1p(1.0)), atol=1e-6)
    finally:
        rerank_mod._rank_cache.cluster_size = None


def test_cluster_size_cache_miss_returns_zeros() -> None:
    """No cache set returns all zeros."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_cluster_size

    rerank_mod._rank_cache.cluster_size = None
    stories = [Story(id=1, title="a", url=None, score=0, time=0, text_content="")]
    out = _meta_cluster_size(stories, now=0.0)
    assert out.shape == (1, 1)
    assert out[0, 0] == 0.0


# domain_recency tests


def test_domain_recency_uses_cache() -> None:
    """_meta_domain_recency returns log1p(days) for known domain."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_domain_recency

    rerank_mod._rank_cache.domain_recency_map = {"example.com": 3.0}
    try:
        stories = [
            Story(
                id=1,
                title="a",
                url="https://example.com/article",
                score=0,
                time=0,
                text_content="",
            )
        ]
        out = _meta_domain_recency(stories, now=0.0)
        np.testing.assert_allclose(out[0, 0], float(np.log1p(3.0)), atol=1e-6)
    finally:
        rerank_mod._rank_cache.domain_recency_map = {}


def test_domain_recency_unknown_domain_sentinel() -> None:
    """Unknown domain gets the sentinel (365 days)."""
    import api.rerank as rerank_mod
    from api.rerank import _meta_domain_recency

    rerank_mod._rank_cache.domain_recency_map = {}
    stories = [
        Story(
            id=1,
            title="a",
            url="https://unknown.com/art",
            score=0,
            time=0,
            text_content="",
        )
    ]
    out = _meta_domain_recency(stories, now=0.0)
    np.testing.assert_allclose(out[0, 0], float(np.log1p(365.0)), atol=1e-6)


def test_domain_recency_no_url_sentinel() -> None:
    """Story with no URL (Ask HN) gets the sentinel."""
    from api.rerank import _meta_domain_recency
    import api.rerank as rerank_mod

    rerank_mod._rank_cache.domain_recency_map = {}

    stories = [Story(id=1, title="Ask HN", url=None, score=0, time=0, text_content="")]
    out = _meta_domain_recency(stories, now=0.0)
    np.testing.assert_allclose(out[0, 0], float(np.log1p(365.0)), atol=1e-6)
