import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock, patch
from api.constants import STORY_CACHE_VERSION
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
            {"ts": time.time(), "version": STORY_CACHE_VERSION, "story": story}
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


@pytest.mark.asyncio
async def test_fetch_story_keeps_title_only_story():
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "type": "story",
        "title": "Title Only",
        "url": "",
        "points": 42,
        "created_at_i": 1700000000,
        "story_text": "",
        "children": [],
    }
    mock_client.get.return_value = mock_response

    with (
        patch("api.fetching.CACHE_PATH") as mock_path,
        patch("api.fetching.atomic_write_json"),
        patch("api.fetching.evict_old_cache_files"),
    ):
        mock_path.__truediv__.return_value.exists.return_value = False

        res = await fetch_story(mock_client, 123)

    assert res is not None
    assert res.title == "Title Only"
    assert res.text_content == "Title Only."


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


def test_rank_stories_disables_freshness_boost_when_configured():
    stories = [
        Story(id=1, title="New", url=None, score=100, time=2000, text_content="New"),
        Story(id=2, title="Old", url=None, score=100, time=1000, text_content="Old"),
    ]
    pos_emb = np.array([[1.0] * 768])
    cand_emb = np.array([[1.0] * 768, [1.0] * 768])

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.FRESHNESS_ENABLED", False),
        patch("api.rerank.FRESHNESS_MAX_BOOST", 0.5),
    ):
        results = rank_stories(stories, pos_emb)

    assert len(results) == 2
    assert all(result.freshness_boost == 0.0 for result in results)


def test_rank_stories_applies_freshness_boost_to_external_stories():
    import time

    now = int(time.time())
    stories = [
        Story(
            id=1,
            title="New external",
            url="https://example.com/new",
            score=0,
            time=now - 3600,
            text_content="Same",
            source="lobsters",
        ),
        Story(
            id=2,
            title="Old external",
            url="https://example.com/old",
            score=0,
            time=now - 48 * 3600,
            text_content="Same",
            source="rss",
        ),
    ]
    pos_emb = np.array([[1.0] * 768])
    cand_emb = np.array([[1.0] * 768, [1.0] * 768])

    with (
        patch("api.rerank.get_embeddings", return_value=cand_emb),
        patch("api.rerank.FRESHNESS_ENABLED", True),
        patch("api.rerank.FRESHNESS_MAX_BOOST", 0.5),
    ):
        results = rank_stories(stories, pos_emb)

    assert len(results) == 2
    assert results[0].index == 0
    assert results[0].freshness_boost > results[1].freshness_boost
    assert results[0].hybrid_score > results[1].hybrid_score
