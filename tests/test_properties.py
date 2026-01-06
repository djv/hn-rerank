import numpy as np
from hypothesis import given, strategies as st
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
    d = draw(st.integers(min_value=2, max_value=5))  # Shared dimension
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
    # Avoid div by zero by adding epsilon or simple check
    c_norm[c_norm == 0] = 1.0
    candidates = candidates / c_norm

    f_norm = np.linalg.norm(favorites, axis=1, keepdims=True)
    f_norm[f_norm == 0] = 1.0
    favorites = favorites / f_norm

    results = rerank.rank_embeddings_maxsim(candidates, favorites)

    # Check each result
    for idx, score, best_fav_idx in results:
        cand_vec = candidates[idx]

        # Manually calculate dot products against all favorites and clip to match implementation
        dots = [np.clip(np.dot(cand_vec, f_vec), -1.0, 1.0) for f_vec in favorites]
        expected_max_score = max(dots)

        # Floating point tolerance
        assert np.isclose(score, expected_max_score, atol=1e-5)

        # Check index consistency
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

    # Permute candidates (reverse them)
    candidates_rev = candidates[::-1]
    results_rev = rerank.rank_embeddings_maxsim(candidates_rev, favorites)

    # The TOP score across the entire set should be identical
    # (assuming the set isn't empty, which min_value=1 guarantees)
    # Note: results are sorted by score, so results_orig[0] is the best match in the original set
    # results_rev[0] is the best match in the reversed set. They should be the same.
    assert np.isclose(results_orig[0][1], results_rev[0][1], atol=1e-5)
