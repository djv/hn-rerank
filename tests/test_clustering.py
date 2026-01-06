import numpy as np
from sklearn.cluster import KMeans


def cluster_favorites(embeddings: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Cluster favorites and return the centroids.
    If n_favorites < n_clusters, return original embeddings.
    """
    if len(embeddings) <= n_clusters:
        return embeddings

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_


def test_clustering_logic():
    """
    Test that clustering correctly identifies centroids of distinct groups.
    """
    # Create 2 distinct groups of vectors
    # Group A: around [1, 0, 0]
    group_a = np.array([[1.0, 0.01, 0.0], [0.99, -0.01, 0.0], [1.01, 0.0, 0.01]])

    # Group B: around [0, 1, 0]
    group_b = np.array([[0.0, 1.0, 0.01], [0.01, 0.99, 0.0], [-0.01, 1.01, 0.0]])

    # Combined
    data = np.vstack([group_a, group_b])  # 6 items

    # Cluster into 2
    centroids = cluster_favorites(data, n_clusters=2)

    assert len(centroids) == 2

    # One centroid should be close to [1, 0, 0]
    # One centroid should be close to [0, 1, 0]

    # We don't know the order, so check if EITHER is close
    has_group_a = False
    has_group_b = False

    for c in centroids:
        if np.allclose(c, [1, 0, 0], atol=0.1):
            has_group_a = True
        if np.allclose(c, [0, 1, 0], atol=0.1):
            has_group_b = True

    assert has_group_a
    assert has_group_b


def test_clustering_small_input():
    """If input size < n_clusters, should return input."""
    data = np.array([[1.0, 0.0], [0.0, 1.0]])
    centroids = cluster_favorites(data, n_clusters=5)
    assert np.array_equal(data, centroids)
