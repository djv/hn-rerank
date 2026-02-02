"""Property-based tests for clustering functions."""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from api.rerank import (
    cluster_interests_with_labels,
    cluster_interests,
    generate_batch_cluster_names,
    _merge_small_clusters,
    _split_large_clusters,
)
from api.constants import (
    MIN_SAMPLES_PER_CLUSTER,
    MAX_CLUSTER_FRACTION,
    MAX_CLUSTER_SIZE,
    MAX_CLUSTERS,
)


# Strategy for generating valid embeddings (L2-normalized vectors)
def embedding_strategy(n_samples: int, dim: int = 384) -> st.SearchStrategy:
    """Generate random normalized embeddings."""
    return arrays(
        dtype=np.float32,
        shape=(n_samples, dim),
        elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    ).map(lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9))


# =============================================================================
# Clustering Invariants
# =============================================================================


def test_empty_input_returns_empty():
    """Empty embeddings → empty centroids and labels."""
    embeddings = np.array([], dtype=np.float32).reshape(0, 384)
    centroids, labels = cluster_interests_with_labels(embeddings)

    assert centroids.shape[0] == 0 or centroids.shape == (0, 384)
    assert len(labels) == 0


def test_single_sample_returns_single_cluster():
    """Single sample → single cluster containing that sample."""
    embeddings = np.random.randn(1, 384).astype(np.float32)
    centroids, labels = cluster_interests_with_labels(embeddings)

    assert len(centroids) == 1
    assert len(labels) == 1
    assert labels[0] == 0


@given(st.integers(min_value=2, max_value=5))
def test_small_sample_fallback(n_samples: int):
    """n < MIN_SAMPLES*2 returns single cluster."""
    assume(n_samples < MIN_SAMPLES_PER_CLUSTER * 2)
    embeddings = np.random.randn(n_samples, 384).astype(np.float32)
    centroids, labels = cluster_interests_with_labels(embeddings)

    assert len(centroids) == 1
    assert len(labels) == n_samples
    assert all(lbl == 0 for lbl in labels)


@given(st.integers(min_value=10, max_value=25))
@settings(max_examples=10, deadline=10000)
def test_labels_cover_all_samples(n_samples: int):
    """Every sample gets a valid non-negative label."""
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, 384).astype(np.float32)
    _, labels = cluster_interests_with_labels(embeddings)

    assert len(labels) == n_samples
    assert all(lbl >= 0 for lbl in labels)


@given(st.integers(min_value=10, max_value=25))
@settings(max_examples=10, deadline=10000)
def test_centroids_match_label_count(n_samples: int):
    """Number of centroids equals number of unique labels."""
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, 384).astype(np.float32)
    centroids, labels = cluster_interests_with_labels(embeddings)

    n_unique_labels = len(set(labels))
    assert len(centroids) == n_unique_labels


@given(st.integers(min_value=10, max_value=25))
@settings(max_examples=10, deadline=10000)
def test_labels_consecutive_from_zero(n_samples: int):
    """Labels are 0, 1, 2, ... with no gaps."""
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, 384).astype(np.float32)
    _, labels = cluster_interests_with_labels(embeddings)

    unique_labels = sorted(set(labels))
    expected = list(range(len(unique_labels)))
    assert unique_labels == expected


def test_deterministic_with_same_input():
    """Same input → same output (determinism check)."""
    np.random.seed(123)
    embeddings = np.random.randn(30, 384).astype(np.float32)

    centroids1, labels1 = cluster_interests_with_labels(embeddings.copy())
    centroids2, labels2 = cluster_interests_with_labels(embeddings.copy())

    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_almost_equal(centroids1, centroids2)


def test_centroid_is_cluster_mean():
    """Each centroid is the mean of its cluster members (without weights)."""
    np.random.seed(42)
    embeddings = np.random.randn(20, 384).astype(np.float32)
    centroids, labels = cluster_interests_with_labels(embeddings, weights=None)

    for cluster_id in range(len(centroids)):
        mask = labels == cluster_id
        if mask.sum() > 0:
            expected_centroid = embeddings[mask].mean(axis=0)
            np.testing.assert_array_almost_equal(
                centroids[cluster_id], expected_centroid, decimal=5
            )


def test_merge_small_clusters_removes_singletons():
    """Singleton clusters are merged into the nearest larger cluster."""
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, 0.1],
            [0.95, 0.0],
            [-0.95, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)

    merged = _merge_small_clusters(embeddings, labels, min_size=MIN_SAMPLES_PER_CLUSTER)
    counts = np.bincount(merged)

    assert counts.min() >= MIN_SAMPLES_PER_CLUSTER
    assert sorted(set(merged)) == list(range(len(set(merged))))


def test_split_large_clusters_limits_max_size():
    """Large clusters are split to respect max cluster size threshold."""
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2],
            [0.75, 0.25],
            [0.7, 0.3],
            [0.65, 0.35],
            [-1.0, 0.0],
            [-0.9, 0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0] * 8 + [1] * 2, dtype=np.int32)
    desired_clusters = 6

    max_size = max(
        MIN_SAMPLES_PER_CLUSTER,
        min(
            MAX_CLUSTER_SIZE,
            int(np.ceil(len(labels) * MAX_CLUSTER_FRACTION)),
        ),
    )

    split = _split_large_clusters(
        embeddings,
        labels,
        min_size=MIN_SAMPLES_PER_CLUSTER,
        max_size=max_size,
        max_clusters=desired_clusters,
    )
    counts = np.bincount(split)

    allowed_max = max_size + MIN_SAMPLES_PER_CLUSTER - 1
    assert counts.max() <= allowed_max
    assert sorted(set(split)) == list(range(len(set(split))))


def test_split_large_clusters_respects_absolute_cap():
    """Large clusters are split to respect absolute max cluster size."""
    np.random.seed(0)
    embeddings = np.random.randn(100, 8).astype(np.float32)
    labels = np.zeros(100, dtype=np.int32)

    split = _split_large_clusters(
        embeddings,
        labels,
        min_size=MIN_SAMPLES_PER_CLUSTER,
        max_size=MAX_CLUSTER_SIZE,
        max_clusters=MAX_CLUSTERS,
    )
    counts = np.bincount(split)

    allowed_max = MAX_CLUSTER_SIZE + MIN_SAMPLES_PER_CLUSTER - 1
    assert counts.max() <= allowed_max


def test_weighted_centroid():
    """Centroids respect weights when provided."""
    np.random.seed(42)
    embeddings = np.random.randn(20, 384).astype(np.float32)
    weights = np.random.rand(20).astype(np.float32)
    weights = weights / weights.sum()  # Normalize

    centroids, labels = cluster_interests_with_labels(embeddings, weights=weights)

    for cluster_id in range(len(centroids)):
        mask = labels == cluster_id
        if mask.sum() > 0:
            cluster_weights = weights[mask]
            expected_centroid = np.average(
                embeddings[mask], axis=0, weights=cluster_weights
            )
            np.testing.assert_array_almost_equal(
                centroids[cluster_id], expected_centroid, decimal=5
            )


def test_cluster_interests_returns_centroids_only():
    """cluster_interests returns just centroids (wrapper function)."""
    np.random.seed(42)
    embeddings = np.random.randn(20, 384).astype(np.float32)

    centroids = cluster_interests(embeddings)
    centroids_with_labels, _ = cluster_interests_with_labels(embeddings)

    np.testing.assert_array_almost_equal(centroids, centroids_with_labels)


# =============================================================================
# Cluster Naming Tests
# =============================================================================


@pytest.mark.asyncio
async def test_cluster_names_non_empty():
    """Every cluster gets a non-empty name."""
    clusters = {
        0: [
            ({"title": "Introduction to Machine Learning"}, 1.0),
            ({"title": "Deep Learning Fundamentals"}, 0.9),
        ],
        1: [
            ({"title": "Python Programming"}, 0.8),
            ({"title": "JavaScript Tutorial"}, 0.7),
        ],
    }

    # Mock Groq API via httpx
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Technology"}}]
        }
        mock_post.return_value = mock_resp

        names = await generate_batch_cluster_names(clusters)

    assert len(names) == 2
    assert all(isinstance(name, str) for name in names.values())
    assert all(len(name) > 0 for name in names.values())


@pytest.mark.asyncio
async def test_fallback_group_name_on_empty_titles():
    """Empty titles → two-word fallback."""
    clusters = {
        0: [
            ({"title": ""}, 1.0),
            ({"title": ""}, 0.9),
        ],
    }

    names = await generate_batch_cluster_names(clusters)

    assert len(names) == 1
    assert names[0] == "Misc Topic"


@pytest.mark.asyncio
async def test_empty_clusters_returns_empty():
    """Empty clusters dict → empty names dict."""
    names = await generate_batch_cluster_names({})
    assert names == {}


@pytest.mark.asyncio
async def test_names_stripped_of_hn_prefixes():
    """Show HN:, Ask HN:, Tell HN: prefixes are stripped before TF-IDF."""
    clusters = {
        0: [
            ({"title": "Show HN: My Cool Project"}, 1.0),
            ({"title": "Ask HN: Best Practices"}, 0.9),
            ({"title": "Tell HN: Something"}, 0.8),
        ],
    }

    # Mock API via httpx
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Projects"}}]
        }
        mock_post.return_value = mock_resp

        names = await generate_batch_cluster_names(clusters)

    assert "Show" not in names[0] or "Hn" not in names[0]


@pytest.mark.asyncio
async def test_invalid_cluster_name_falls_back():
    clusters = {
        0: [
            ({"title": "Show HN: My Tool"}, 1.0),
        ],
    }

    with patch("api.rerank._load_cluster_name_cache", return_value={}), patch(
        "api.rerank._save_cluster_name_cache", lambda _cache: None
    ), patch(
        "api.rerank._generate_with_retry",
        new=AsyncMock(return_value='{"0": "Not Provided"}'),
    ):
        names = await generate_batch_cluster_names(clusters)

    assert set(names[0].split()) == {"My", "Tool"}


@pytest.mark.asyncio
async def test_llm_cluster_name_requires_title_overlap():
    clusters = {
        0: [
            ({"title": "Tree-sitter vs. Language Servers"}, 1.0),
            ({"title": "Crafting Interpreters"}, 0.9),
        ],
    }

    with patch("api.rerank._load_cluster_name_cache", return_value={}), patch(
        "api.rerank._save_cluster_name_cache", lambda _cache: None
    ), patch(
        "api.rerank._generate_with_retry",
        new=AsyncMock(return_value='{"0": "LLM Coding Agents"}'),
    ), patch(
        "api.rerank._fallback_cluster_name", return_value="Language Systems"
    ):
        names = await generate_batch_cluster_names(clusters)

    assert names[0] == "Language Systems"


@pytest.mark.asyncio
async def test_llm_cluster_name_kept_without_keyword_overlap():
    clusters = {
        0: [
            ({"title": "Transportation systems for urban transit"}, 1.0),
            ({"title": "Systems reliability in autonomous transit"}, 0.9),
        ],
    }

    with patch("api.rerank._load_cluster_name_cache", return_value={}), patch(
        "api.rerank._save_cluster_name_cache", lambda _cache: None
    ), patch(
        "api.rerank._generate_with_retry",
        new=AsyncMock(return_value='{"0": "Transportation Systems"}'),
    ):
        names = await generate_batch_cluster_names(clusters)

    assert names[0] == "Transportation Systems"


@pytest.mark.asyncio
async def test_llm_cluster_name_allows_six_words():
    clusters = {
        0: [
            ({"title": "Large language models for code generation"}, 1.0),
            ({"title": "Neural networks and transformers"}, 0.9),
        ],
    }

    with patch("api.rerank._load_cluster_name_cache", return_value={}), patch(
        "api.rerank._save_cluster_name_cache", lambda _cache: None
    ), patch(
        "api.rerank._generate_with_retry",
        new=AsyncMock(return_value='{"0": "Deep Learning for Large Language Models"}'),
    ):
        names = await generate_batch_cluster_names(clusters)

    assert names[0] == "Deep Learning for Large Language Models"


@pytest.mark.asyncio
async def test_llm_cluster_name_truncates_to_max_words():
    clusters = {
        0: [
            ({"title": "Graph neural networks in drug discovery"}, 1.0),
            ({"title": "Protein structure prediction pipelines"}, 0.9),
        ],
    }

    with patch("api.rerank._load_cluster_name_cache", return_value={}), patch(
        "api.rerank._save_cluster_name_cache", lambda _cache: None
    ), patch(
        "api.rerank._generate_with_retry",
        new=AsyncMock(
            return_value='{"0": "Graph Neural Networks for Drug Discovery Pipelines"}'
        ),
    ):
        names = await generate_batch_cluster_names(clusters)

    assert names[0] == "Graph Neural Networks for Drug Discovery"
