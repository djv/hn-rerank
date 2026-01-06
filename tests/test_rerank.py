from unittest.mock import patch

import numpy as np

from api import rerank


def test_rank_embeddings_maxsim_logic():
    """
    Verify MaxSim logic: Takes the max similarity across all favorites.
    """
    # 2 Favorites (Queries)
    # F1: [1, 0]
    # F2: [0, 1]
    fav_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    # 3 Candidates (Documents)
    # C1: [1, 0] (Matches F1 perfectly -> Score should be 1.0)
    # C2: [0.707, 0.707] (Matches both equally -> Cosine Sim is ~0.707)
    # C3: [-1, 0] (Opposite of F1 -> Score should be roughly 0.0 or -1.0 depending on F2 match)
    # C3 vs F1: -1.0
    # C3 vs F2: 0.0
    # Max Score for C3 should be 0.0 (match with F2 is better than F1)

    cand_embeddings = np.array(
        [
            [1.0, 0.0],  # C1
            [0.7071, 0.7071],  # C2
            [-1.0, 0.0],  # C3
        ]
    )

    results = rerank.rank_embeddings_maxsim(cand_embeddings, fav_embeddings)

    # Check C1 (Index 0)
    # Should be rank 1 with score 1.0, matching F1 (Index 0)
    assert results[0][0] == 0
    assert np.isclose(results[0][1], 1.0)
    assert results[0][2] == 0

    # Check C2 (Index 1)
    # Should be rank 2 with score ~0.707, matching F1 or F2 (both equal)
    assert results[1][0] == 1
    assert np.isclose(results[1][1], 0.7071, atol=0.001)

    # Check C3 (Index 2)
    # Should be rank 3 with score 0.0, matching F2 (Index 1) best
    # (C3 dot F1 is -1.0, C3 dot F2 is 0.0. Max is 0.0)
    assert results[2][0] == 2
    assert np.isclose(results[2][1], 0.0, atol=0.001)
    assert results[2][2] == 1


def test_rank_candidates_integration():
    """
    Test the high-level function with mocked embeddings.
    """
    with patch("api.rerank.get_embeddings") as mock_embed:
        # Return dummy embeddings
        mock_embed.side_effect = [
            np.array([[1, 0]]),  # Favorites call
            np.array([[1, 0], [0, 1]]),  # Candidates call
        ]

        results = rerank.rank_candidates(["c1", "c2"], ["f1"])

        # C1 matches F1 (1.0)
        # C2 orthogonal to F1 (0.0)
        assert results[0][0] == 0
        assert results[0][1] > 0.9
        assert results[1][0] == 1
        assert results[1][1] < 0.1
