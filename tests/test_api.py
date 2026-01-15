import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from api.rerank import rank_stories
from api.fetching import fetch_story
from api.models import Story


@pytest.mark.asyncio
async def test_fetch_story_cached():
    mock_client = AsyncMock()
    sid = 123
    story = {
        "id": sid,
        "title": "Test",
        "url": None,
        "score": 0,
        "time": 0,
        "comments": [],
        "text_content": "",
    }

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
        assert res.id == sid
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
    stories = [
        Story(id=1, title="Test", url=None, score=100, time=1000, text_content="A")
    ]
    pos_emb = np.array([[1.0] * 768])
    # rank_stories expects get_embeddings to work
    with patch("api.rerank.get_embeddings", return_value=np.array([[1.0] * 768])):
        results = rank_stories(stories, pos_emb)
        assert len(results) == 1
        assert results[0].index == 0


def test_rank_stories_no_positive_signals():
    """
    Regression test: When there are no positive signals (empty upvotes),
    ranking should still work but return 0% match scores and fav_idx=-1.
    """
    stories = [
        Story(
            id=1,
            title="AI Story",
            url=None,
            score=100,
            time=1000,
            text_content="Story about AI",
        ),
        Story(
            id=2,
            title="Python Story",
            url=None,
            score=200,
            time=2000,
            text_content="Story about Python",
        ),
    ]

    # No positive signals - pass None for positive_embeddings
    with patch(
        "api.rerank.get_embeddings", return_value=np.array([[0.5] * 768, [0.6] * 768])
    ):
        results = rank_stories(stories, positive_embeddings=None)

        assert len(results) == 2
        # All should have 0 max_sim and -1 fav_idx when no positive signals
        for result in results:
            assert result.max_sim_score == 0.0, (
                "max_sim should be 0 when no positive signals"
            )
            assert result.best_fav_index == -1, (
                "fav_idx should be -1 when no positive signals"
            )


def test_rank_stories_empty_positive_embeddings():
    """
    Regression test: Empty positive embeddings array should behave like None.
    """
    stories = [
        Story(
            id=1,
            title="Test",
            url=None,
            score=100,
            time=1000,
            text_content="Test story",
        )
    ]

    with patch("api.rerank.get_embeddings", return_value=np.array([[0.5] * 768])):
        # Empty array
        results = rank_stories(stories, positive_embeddings=np.array([]))

        assert len(results) == 1
        result = results[0]
        assert result.max_sim_score == 0.0
        assert result.best_fav_index == -1
