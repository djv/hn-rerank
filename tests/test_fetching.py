from api.main import fetch_story_with_comments, get_top_stories, fetch_article_text, get_user_data
import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_get_user_data():
    username = "testuser"
    mock_ids = {101, 102}
    mock_story = {"id": 101, "title": "Fav"}
    
    with patch("api.main.HNClient") as MockClass:
        client = MockClass.return_value
        client.fetch_favorites = AsyncMock(return_value=mock_ids)
        client.check_session = AsyncMock(return_value=False)
        client.close = AsyncMock()
        
        with patch("api.main.fetch_story_with_comments", return_value=mock_story):
            pos, neg, exclude = await get_user_data(username)
            assert len(pos) > 0
            assert 101 in exclude

@pytest.mark.asyncio
async def test_fetch_article_text():
    # Now it uses trafilatura
    res = await fetch_article_text("http://example.com")
    assert "domain" in res.lower()

@pytest.mark.asyncio
async def test_get_top_stories():
    mock_ids = [1, 2, 3]
    mock_story = {"id": 1, "title": "Test"}
    
    with patch("httpx.AsyncClient.get") as m_get:
        m_get.return_value = AsyncMock(status_code=200, json=lambda: mock_ids)
        
        with patch("api.main.fetch_story_with_comments", return_value=mock_story):
            res = await get_top_stories(limit=2)
            assert len(res) == 2
            assert res[0] == mock_story



@pytest.mark.asyncio
async def test_fetch_story_aggregates_comments():
    """
    Test that fetch_story_with_comments correctly fetches the story
    and combines title and comments.
    """
    mock_client = AsyncMock()

    # Mock Algolia Item response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": 100,
        "title": "Main Story",
        "url": "http://example.com",
        "points": 100,
        "children": [
            {"text": "Comment 1", "points": 10},
            {"text": "Comment 2", "points": 5},
        ],
    }
    mock_client.get.return_value = mock_response

    # Run
    result = await fetch_story_with_comments(mock_client, 100)

    assert result is not None
    assert result["title"] == "Main Story"
    assert "Comment 1" in result["text_content"]


@pytest.mark.asyncio
async def test_get_top_stories_resilience():
    """
    Test that get_top_stories continues even if individual story fetches fail.
    """
    from api.main import get_top_stories

    mock_client = AsyncMock()
    mock_response_ids = MagicMock()
    mock_response_ids.json.return_value = [1, 2, 3]

    with patch("httpx.AsyncClient", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response_ids

        with patch("api.main.fetch_story_with_comments") as mock_fetch:
            mock_fetch.side_effect = [
                {"id": 1, "title": "Good Story", "text_content": "Good Story"},
                None,
                {"id": 3, "title": "Another Story", "text_content": "Another Story"},
            ]

            stories = await get_top_stories(limit=3)
            assert len(stories) == 2
