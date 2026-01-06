
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from tui_app import HNRerankTUI
# Removed DataTable import as it is not used

@pytest.fixture
def mock_app_deps():
    with patch("tui_app.get_user_data", new_callable=AsyncMock) as m_user, \
         patch("tui_app.get_best_stories", new_callable=AsyncMock) as m_cand, \
         patch("tui_app.HNClient") as MockClient, \
         patch("api.rerank.init_model"), \
         patch("api.rerank.get_embeddings"), \
         patch("api.rerank.rank_stories") as m_rank:

        m_user.return_value = ([], [], set())
        # Added text_content
        m_cand.return_value = [{"id": 1, "title": "Test", "url": "http://test.com", "text_content": "Test content"}]
        m_rank.return_value = [(0, 1.0, 0)]

        client = MockClient.return_value
        client.close = AsyncMock()
        client.login = AsyncMock(return_value=(True, "OK"))

        yield

@pytest.mark.asyncio
async def test_action_open_link(mock_app_deps):
    app = HNRerankTUI("user")
    with patch("webbrowser.open_new_tab") as mock_open:
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            # Focus is on list, press 'v' to open article (binding is v, not o)
            # wait, BINDINGS say: Binding("v", "open_article", "View")
            # But test used "o".

            # Ensure an item is selected/highlighted.
            list_view = app.query_one("#story-list")
            if list_view.children:
                list_view.index = 0

            await pilot.press("v")
            # It might take a moment or need async wait?
            # _open_url calls notify then open_new_tab.
            mock_open.assert_called_with("http://test.com")

@pytest.mark.asyncio
async def test_action_open_hn_link(mock_app_deps):
    app = HNRerankTUI("user")
    with patch("webbrowser.open_new_tab") as mock_open:
        async with app.run_test() as pilot:
            await pilot.pause(0.5)

            list_view = app.query_one("#story-list")
            if list_view.children:
                list_view.index = 0

            # Binding for Comments is 'c'
            await pilot.press("c")
            mock_open.assert_called_with("https://news.ycombinator.com/item?id=1")

@pytest.mark.asyncio
async def test_action_quit(mock_app_deps):
    app = HNRerankTUI("user")
    async with app.run_test() as pilot:
        await pilot.press("q")
        assert not app.is_running

@pytest.mark.asyncio
async def test_load_stories_error_handling(mock_app_deps):
    # Mock rank_stories to fail
    with patch("api.rerank.rank_stories", side_effect=Exception("Rerank failed")):
        app = HNRerankTUI("user")
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            # Check if notification or log happened.
            # Since TUI might catch it and notify, we check notifications
            # Or at least it doesn't crash
            assert app.is_running
            # Verify status bar might say Error or similar if implemented
            # app.query_one("Footer") ...
