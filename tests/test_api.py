import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from api.rerank import compute_recency_weights, rank_stories
from api.fetching import fetch_story


def test_recency_weights():
    now = 100000000
    # 0 days, 10 days, 400 days
    timestamps = [now, now - 86400 * 10, now - 86400 * 400]
    with patch("time.time", return_value=now):
        weights = compute_recency_weights(timestamps)
        # 0 days -> ~0.97
        assert weights[0] > 0.9
        # 10 days -> ~0.97 (plateau)
        assert weights[1] > 0.9
        assert weights[1] <= weights[0]  # Monotonic
        # 400 days -> < 0.5 (decay)
        assert weights[2] < 0.5


@pytest.mark.asyncio
async def test_fetch_story_cached():
    mock_client = AsyncMock()
    sid = 123
    story = {"id": sid, "title": "Test"}

    with patch("api.fetching.CACHE_PATH") as mock_path:
        mock_file = mock_path.__truediv__.return_value
        mock_file.exists.return_value = True
        import json
        import time

        mock_file.read_text.return_value = json.dumps(
            {"ts": time.time(), "story": story}
        )

        res = await fetch_story(mock_client, sid)
        assert res is not None
        assert res["id"] == sid
        assert mock_client.get.call_count == 0


@pytest.mark.asyncio
async def test_fetch_story_network_error():
    """Ensure fetch_story returns None on network failure."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Connection refused")

    with patch("api.fetching.CACHE_PATH") as mock_path:
        mock_path.__truediv__.return_value.exists.return_value = False

        res = await fetch_story(mock_client, 123)
        assert res is None


def test_rank_stories_basic():
    stories = [{"id": 1, "score": 100, "time": 1000, "text_content": "A"}]
    pos_emb = np.array([[1.0] * 768])
    # rank_stories expects get_embeddings to work
    with patch("api.rerank.get_embeddings", return_value=np.array([[1.0] * 768])):
        results = rank_stories(stories, pos_emb)
        assert len(results) == 1
        assert results[0][0] == 0


def test_rank_stories_no_positive_signals():
    """
    Regression test: When there are no positive signals (empty upvotes),
    ranking should still work but return 0% match scores and fav_idx=-1.
    """
    stories = [
        {"id": 1, "score": 100, "time": 1000, "text_content": "Story about AI"},
        {"id": 2, "score": 200, "time": 2000, "text_content": "Story about Python"},
    ]

    # No positive signals - pass None for positive_embeddings
    with patch(
        "api.rerank.get_embeddings", return_value=np.array([[0.5] * 768, [0.6] * 768])
    ):
        results = rank_stories(stories, positive_embeddings=None)

        assert len(results) == 2
        # All should have 0 max_sim and -1 fav_idx when no positive signals
        for idx, score, fav_idx, max_sim in results:
            assert max_sim == 0.0, "max_sim should be 0 when no positive signals"
            assert fav_idx == -1, "fav_idx should be -1 when no positive signals"


def test_rank_stories_empty_positive_embeddings():
    """
    Regression test: Empty positive embeddings array should behave like None.
    """
    stories = [{"id": 1, "score": 100, "time": 1000, "text_content": "Test story"}]

    with patch("api.rerank.get_embeddings", return_value=np.array([[0.5] * 768])):
        # Empty array
        results = rank_stories(stories, positive_embeddings=np.array([]))

        assert len(results) == 1
        idx, score, fav_idx, max_sim = results[0]
        assert max_sim == 0.0
        assert fav_idx == -1
