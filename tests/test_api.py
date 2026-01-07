import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from api.rerank import compute_recency_weights, rank_stories
from api.fetching import fetch_story


def test_recency_weights():
    now = 1000000
    timestamps = [now, now - 86400 * 10]
    with patch("time.time", return_value=now):
        weights = compute_recency_weights(timestamps)
        assert weights[0] == 1.0
        assert weights[1] < 1.0


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
