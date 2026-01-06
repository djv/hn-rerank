import pytest
from unittest.mock import AsyncMock, patch
from tui_app import HNRerankTUI, StoryItem
import numpy as np
from textual.widgets import ListView

# Mock Data
MOCK_STORY = {
    "id": 123,
    "title": "Test Story",
    "score": 100,
    "url": "http://example.com/test",
    "text_content": "Test Story Content",
    "comments": ["Comment 1"],
}


@pytest.fixture
def mock_rerank():
    with (
        patch("api.rerank.init_model"),
        patch("api.rerank.get_embeddings", return_value=np.zeros((1, 768))),
        patch("api.rerank.rank_stories", return_value=[(0, 0.9, 0)]),
    ):
        yield


@pytest.fixture
def mock_api():
    with (
        patch("tui_app.get_user_data", new_callable=AsyncMock) as m_user,
        patch("tui_app.get_best_stories", new_callable=AsyncMock) as m_cand,
    ):
        m_user.return_value = ([MOCK_STORY], [], {123})
        m_cand.return_value = [MOCK_STORY]
        yield m_user, m_cand


@pytest.fixture
def mock_hn_client():
    with patch("tui_app.HNClient") as MockClass:
        client = MockClass.return_value
        client.vote = AsyncMock(return_value=(True, "Voted"))
        client.hide = AsyncMock(return_value=(True, "Hidden"))
        client.close = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock()
        yield client


@pytest.mark.asyncio
async def test_tui_loading_and_expand(mock_rerank, mock_api, mock_hn_client):
    """Test that TUI loads stories and expands on Enter."""
    app = HNRerankTUI("testuser")
    async with app.run_test() as pilot:
        await pilot.pause(1.0)

        list_view = app.query_one("#story-list", ListView)
        assert len(list_view.children) == 1

        item = list_view.children[0]
        assert isinstance(item, StoryItem)
        # Should be auto-expanded now
        assert item.expanded

        # Press Enter to toggle (collapse)
        await pilot.press("enter")
        assert not item.expanded
        
@pytest.mark.asyncio
async def test_tui_shortcuts(mock_rerank, mock_api, mock_hn_client):
    """Test shortcuts like upvote and hide."""
    app = HNRerankTUI("testuser")
    async with app.run_test() as pilot:
        await pilot.pause(1.0)

        # u -> Upvote
        await pilot.press("u")
        await pilot.pause(0.1)
        mock_hn_client.vote.assert_called_with(123, "up")

        # d -> Hide
        await pilot.press("d")
        await pilot.pause(0.1)
        mock_hn_client.hide.assert_called_with(123)


@pytest.mark.asyncio
async def test_tui_navigation(mock_rerank, mock_api, mock_hn_client):
    """Test j/k navigation."""
    mock_user, mock_cand = mock_api
    mock_cand.return_value = [MOCK_STORY, {**MOCK_STORY, "id": 456, "title": "Second"}]

    # Need to mock rank_stories to return two items
    with patch("api.rerank.rank_stories", return_value=[(0, 0.9, 0), (1, 0.8, 0)]):
        app = HNRerankTUI("testuser")
        async with app.run_test() as pilot:
            await pilot.pause(1.0)

            list_view = app.query_one("#story-list", ListView)
            assert list_view.index == 0

            await pilot.press("down")
            assert list_view.index == 1

            await pilot.press("up")
            assert list_view.index == 0
