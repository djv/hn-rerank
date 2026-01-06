
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from api import rerank

# Helper strategy to generate (N, D) array with fixed D
def embeddings(n_min, n_max, d):
    return arrays(
        np.float64,
        shape=st.tuples(st.integers(min_value=n_min, max_value=n_max), st.just(d)),
        elements=st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )

@st.composite
def same_dimension_embeddings(draw):
    d = draw(st.integers(min_value=2, max_value=10))  # Shared dimension
    cands = draw(embeddings(1, 20, d))
    faves = draw(embeddings(1, 20, d))
    return cands, faves

@given(data=same_dimension_embeddings())
def test_maxsim_property_correctness(data):
    """
    Property: The score returned by rank_embeddings_maxsim for candidate C
    must match max(dot(C, F) for F in favorites).
    """
    candidates, favorites = data

    # Normalize inputs to simulate unit vectors (like embeddings)
    c_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    c_norm[c_norm == 0] = 1.0
    candidates = candidates / c_norm

    f_norm = np.linalg.norm(favorites, axis=1, keepdims=True)
    f_norm[f_norm == 0] = 1.0
    favorites = favorites / f_norm

    results = rerank.rank_embeddings_maxsim(candidates, favorites)

    # Check each result
    for idx, score, best_fav_idx in results:
        cand_vec = candidates[idx]
        dots = [np.clip(np.dot(cand_vec, f_vec), -1.0, 1.0) for f_vec in favorites]
        expected_max_score = max(dots)
        assert np.isclose(score, expected_max_score, atol=1e-5)
        assert np.isclose(
            np.dot(cand_vec, favorites[best_fav_idx]), expected_max_score, atol=1e-5
        )

@given(data=same_dimension_embeddings())
def test_maxsim_permutation_invariance(data):
    """
    Property: Shuffling candidates should not change their scores.
    """
    candidates, favorites = data

    # Normalize
    c_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    c_norm[c_norm == 0] = 1.0
    candidates = candidates / c_norm

    f_norm = np.linalg.norm(favorites, axis=1, keepdims=True)
    f_norm[f_norm == 0] = 1.0
    favorites = favorites / f_norm

    # Original run
    results_orig = rerank.rank_embeddings_maxsim(candidates, favorites)

    # Store scores mapped by embedding content hash or similar to verify identity
    # But since embeddings are floats, let's just reverse and check strict index mapping

    candidates_rev = candidates[::-1]
    results_rev = rerank.rank_embeddings_maxsim(candidates_rev, favorites)

    # Sort both results by score to compare distributions
    scores_orig = sorted([r[1] for r in results_orig], reverse=True)
    scores_rev = sorted([r[1] for r in results_rev], reverse=True)

    assert np.allclose(scores_orig, scores_rev, atol=1e-5)

@settings(deadline=None)
@given(data=same_dimension_embeddings(), diversity_penalty=st.floats(0.0, 1.0))
def test_mmr_properties(data, diversity_penalty):
    """
    Property: rank_mmr should return all candidates if no cutoff is implicitly applied.
    The first item should be the MaxSim item (if penalty is 0, or if it's the first selection).
    Actually, MMR with penalty > 0 changes the first item only if we seeded it, but here we don't.
    The first item picked by MMR is ALWAYS the one with highest relevance, because redundancy is 0 for the first item.
    """
    candidates, favorites = data

    # Normalize
    c_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    c_norm[c_norm == 0] = 1.0
    candidates = candidates / c_norm

    f_norm = np.linalg.norm(favorites, axis=1, keepdims=True)
    f_norm[f_norm == 0] = 1.0
    favorites = favorites / f_norm

    results_mmr = rerank.rank_mmr(candidates, favorites, diversity_penalty=diversity_penalty)
    results_maxsim = rerank.rank_embeddings_maxsim(candidates, favorites)

    # 1. Output length should be same as candidates
    assert len(results_mmr) == len(candidates)

    # 2. The first item picked by MMR should be the same as the top item in MaxSim
    # Because for the first selection, redundancy is 0.
    if len(results_mmr) > 0:
        first_mmr = results_mmr[0]
        # MaxSim results are sorted by score descending
        first_maxsim = results_maxsim[0]

        assert first_mmr[0] == first_maxsim[0] # Same candidate index
        assert np.isclose(first_mmr[1], first_maxsim[1], atol=1e-5) # Same score

@given(
    embeddings=arrays(
        np.float64,
        shape=st.tuples(st.integers(min_value=2, max_value=20), st.integers(min_value=2, max_value=10)),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
)
def test_cluster_and_reduce_properties(embeddings):
    """
    Property:
    1. Centroids are normalized.
    2. Number of centroids <= input size.
    3. Every point maps to a valid centroid index.
    """
    # Normalize input first
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    embeddings = embeddings / norm

    centroids, rep_indices, labels = rerank.cluster_and_reduce_auto(embeddings)

    # Check 1: Centroids are normalized (unit length), unless they are zero vectors
    c_norms = np.linalg.norm(centroids, axis=1)
    # Check close to 1.0 or 0.0 (if cluster was all zeros)
    is_unit = np.isclose(c_norms, 1.0, atol=1e-5)
    is_zero = np.isclose(c_norms, 0.0, atol=1e-5)
    assert np.all(is_unit | is_zero)

    # Check 2: Number of centroids
    assert len(centroids) <= len(embeddings)

    # Check 3: Labels correspond to valid clusters
    assert len(labels) == len(embeddings)
    unique_labels = set(labels)
    assert len(unique_labels) == len(centroids)
    assert min(labels) >= 0
    assert max(labels) < len(centroids)

