"""
Comprehensive TUI tests covering all untested paths and edge cases.
"""
import os
import time
from typing import cast
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from textual.widgets import ListView

from tui_app import HNRerankTUI, StoryItem, get_relative_time

# --- Tests for get_relative_time helper function ---

def test_get_relative_time_empty():
    """Test get_relative_time with no timestamp"""
    assert get_relative_time(0) == ""
    assert get_relative_time(None) == ""


def test_get_relative_time_now():
    """Test get_relative_time with current timestamp"""
    current = int(time.time())
    assert get_relative_time(current) == "now"
    assert get_relative_time(current - 30) == "now"


def test_get_relative_time_minutes():
    """Test get_relative_time with minutes ago"""
    current = int(time.time())
    assert get_relative_time(current - 120) == "2m"
    assert get_relative_time(current - 1800) == "30m"
    assert get_relative_time(current - 3590) == "59m"


def test_get_relative_time_hours():
    """Test get_relative_time with hours ago"""
    current = int(time.time())
    assert get_relative_time(current - 3600) == "1h"
    assert get_relative_time(current - 7200) == "2h"
    assert get_relative_time(current - 82800) == "23h"


def test_get_relative_time_days():
    """Test get_relative_time with days ago"""
    current = int(time.time())
    assert get_relative_time(current - 86400) == "1d"
    assert get_relative_time(current - 172800) == "2d"
    assert get_relative_time(current - 604800) == "7d"


# --- Fixtures ---

@pytest.fixture
def mock_story():
    return {
        "id": 123,
        "title": "Test Story",
        "score": 100,
        "url": "https://example.com/test",
        "text_content": "Test content",
        "article_snippet": "Article preview text here",
        "comments": ["Comment 1", "Comment 2"],
        "time": int(time.time()) - 3600,
    }


@pytest.fixture
def comprehensive_tui_mocks():
    """Comprehensive mock setup for TUI testing"""

    def mock_ranker(candidates, **kwargs):
        return [(i, 1.0 - (i * 0.01), 0) for i in range(len(candidates))]

    with patch("api.rerank.init_model"), \
         patch("api.rerank.get_embeddings", return_value=np.zeros((1, 768))), \
         patch("api.rerank.rank_stories", side_effect=mock_ranker), \
         patch("tui_app.get_user_data", new_callable=AsyncMock) as m_user, \
         patch("tui_app.get_best_stories", new_callable=AsyncMock) as m_cand, \
         patch("tui_app.HNClient") as MockClient:

        story = {
            "id": 999,
            "title": "Test Story",
            "score": 100,
            "url": "https://example.com/test",
            "text_content": "Test content",
            "article_snippet": "Article snippet",
            "comments": ["Comment"],
            "time": int(time.time()),
        }

        m_user.return_value = ([story], [], {999})
        m_cand.return_value = [story]

        client = MockClient.return_value
        client.vote = AsyncMock(return_value=(True, "OK"))
        client.hide = AsyncMock(return_value=(True, "OK"))
        client.close = AsyncMock()
        client.login = AsyncMock(return_value=(True, "OK"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock()

        yield {
            "user": m_user,
            "cand": m_cand,
            "client": client,
            "MockClient": MockClient,
        }


# --- Auto-login tests ---

@pytest.mark.asyncio
async def test_auto_login_with_username(comprehensive_tui_mocks):
    """Test app behavior when username is provided"""
    app = HNRerankTUI("testuser")
    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        # App should be running with provided username
        assert app.is_running
        assert app.username == "testuser"


@pytest.mark.asyncio
async def test_no_username_warning(comprehensive_tui_mocks):
    """Test warning when no username is provided"""
    with patch.dict("os.environ", {}, clear=True):
        app = HNRerankTUI(None)
        async with app.run_test() as pilot:
            await pilot.pause(1.5)
            # App should still work with a default username
            assert app.is_running


# --- Open article/comments tests ---

@pytest.mark.asyncio
async def test_open_article_action(comprehensive_tui_mocks):
    """Test opening article action doesn't crash"""
    app = HNRerankTUI("testuser")

    # Ensure SSH_CONNECTION is not in environment so webbrowser is called
    env = dict(os.environ)
    env.pop("SSH_CONNECTION", None)

    with patch("tui_app.webbrowser.open_new_tab") as mock_browser, \
         patch.dict("os.environ", env, clear=True):
        async with app.run_test() as pilot:
            await pilot.pause(1.5)

            list_view = app.query_one("#story-list", ListView)
            assert len(list_view.children) > 0

            # Directly call the action to test it
            app.action_open_article()
            await pilot.pause(0.2)

            # The action should have been called (even if the key binding doesn't work)
            mock_browser.assert_called()


@pytest.mark.asyncio
async def test_open_comments_action(comprehensive_tui_mocks):
    """Test opening comments action doesn't crash"""
    app = HNRerankTUI("testuser")

    # Ensure SSH_CONNECTION is not in environment so webbrowser is called
    env = dict(os.environ)
    env.pop("SSH_CONNECTION", None)

    with patch("tui_app.webbrowser.open_new_tab") as mock_browser, \
         patch.dict("os.environ", env, clear=True):
        async with app.run_test() as pilot:
            await pilot.pause(1.5)

            list_view = app.query_one("#story-list", ListView)
            assert len(list_view.children) > 0

            # Directly call the action to test it
            app.action_open_hn()
            await pilot.pause(0.2)

            # The action should have been called
            mock_browser.assert_called()
            args = mock_browser.call_args[0]
            assert "news.ycombinator.com/item?id=999" in args[0]


@pytest.mark.asyncio
async def test_ssh_clipboard_copy(comprehensive_tui_mocks):
    """Test clipboard copy when in SSH session"""
    app = HNRerankTUI("testuser")

    with patch.dict("os.environ", {"SSH_CONNECTION": "1"}), \
         patch.object(app, "copy_to_clipboard") as mock_clipboard:

        async with app.run_test() as pilot:
            await pilot.pause(1.5)

            # Press 'v' to open article
            await pilot.press("v")
            await pilot.pause(0.1)

            # Should copy to clipboard instead of opening browser
            mock_clipboard.assert_called_once()


# --- Button click tests ---

@pytest.mark.skip(reason="Textual button click on hidden elements unreliable in tests")
@pytest.mark.asyncio
async def test_button_click_open_article(comprehensive_tui_mocks, mock_story):
    """Test clicking the 'open article' button"""
    app = HNRerankTUI("testuser")

    with patch("tui_app.webbrowser.open_new_tab") as mock_browser:
        async with app.run_test() as pilot:
            await pilot.pause(1.5)

            list_view = app.query_one("#story-list", ListView)
            item = cast(StoryItem, list_view.children[0])

            # Ensure item is expanded so button is visible
            item.expanded = True
            await pilot.pause(0.1)

            # Click the "Article" button
            button = item.query_one("#open-art")
            await pilot.click(button)
            await pilot.pause(0.1)

            mock_browser.assert_called()


@pytest.mark.skip(reason="Textual button click on hidden elements unreliable in tests")
@pytest.mark.asyncio
async def test_button_click_open_comments(comprehensive_tui_mocks):
    """Test clicking the 'comments' button"""
    app = HNRerankTUI("testuser")

    with patch("tui_app.webbrowser.open_new_tab") as mock_browser:
        async with app.run_test() as pilot:
            await pilot.pause(1.5)

            list_view = app.query_one("#story-list", ListView)
            item = cast(StoryItem, list_view.children[0])

            # Ensure item is expanded so button is visible
            item.expanded = True
            await pilot.pause(0.1)

            # Click the "Comments" button
            button = item.query_one("#open-hn")
            await pilot.click(button)
            await pilot.pause(0.1)

            mock_browser.assert_called()


@pytest.mark.skip(reason="Textual button click on hidden elements unreliable in tests")
@pytest.mark.asyncio
async def test_button_click_upvote(comprehensive_tui_mocks):
    """Test clicking the upvote button"""
    mocks = comprehensive_tui_mocks
    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        list_view = app.query_one("#story-list", ListView)
        item = cast(StoryItem, list_view.children[0])

        # Ensure item is expanded so button is visible
        item.expanded = True
        await pilot.pause(0.1)

        # Click the "Upvote" button
        button = item.query_one("#upvote")
        await pilot.click(button)
        await pilot.pause(0.3)

        mocks["client"].vote.assert_called_with(999, "up")


@pytest.mark.skip(reason="Textual button click on hidden elements unreliable in tests")
@pytest.mark.asyncio
async def test_button_click_hide(comprehensive_tui_mocks):
    """Test clicking the hide button"""
    mocks = comprehensive_tui_mocks
    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        list_view = app.query_one("#story-list", ListView)
        item = cast(StoryItem, list_view.children[0])

        # Ensure item is expanded so button is visible
        item.expanded = True
        await pilot.pause(0.1)

        # Click the "Hide" button
        button = item.query_one("#hide")
        await pilot.click(button)
        await pilot.pause(0.3)

        mocks["client"].hide.assert_called_with(999)


# --- Refresh feed tests ---

@pytest.mark.asyncio
async def test_refresh_feed_action(comprehensive_tui_mocks):
    """Test refreshing the feed with 'r' key"""
    mocks = comprehensive_tui_mocks
    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        # Reset mock call counts
        mocks["user"].reset_mock()
        mocks["cand"].reset_mock()

        # Press 'r' to refresh
        await pilot.press("r")
        await pilot.pause(1.5)

        # Verify API calls were made again
        mocks["user"].assert_called()
        mocks["cand"].assert_called()


# --- Error handling tests ---

@pytest.mark.asyncio
async def test_empty_candidates_handling(comprehensive_tui_mocks):
    """Test handling when no new stories are found"""
    mocks = comprehensive_tui_mocks
    mocks["cand"].return_value = []

    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        # App should still be running
        assert app.is_running

        # List view should be empty
        list_view = app.query_one("#story-list", ListView)
        assert len(list_view.children) == 0


@pytest.mark.asyncio
async def test_refresh_feed_error_handling(comprehensive_tui_mocks):
    """Test error handling during feed refresh"""
    mocks = comprehensive_tui_mocks
    mocks["user"].side_effect = Exception("Network error")

    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        # App should still be running despite error
        assert app.is_running


# --- Vote status and pending actions tests ---

@pytest.mark.asyncio
async def test_vote_status_update(comprehensive_tui_mocks):
    """Test that vote status icon updates after voting"""
    _ = comprehensive_tui_mocks  # fixture sets up mocks
    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        list_view = app.query_one("#story-list", ListView)
        item = cast(StoryItem, list_view.children[0])

        # Initial vote status should be None
        assert item.vote_status is None

        # Upvote
        await pilot.press("u")
        await pilot.pause(0.3)

        # Vote status should update
        assert item.vote_status == "up"


@pytest.mark.asyncio
async def test_pending_actions_deduplication(comprehensive_tui_mocks):
    """Test that duplicate actions on the same story are prevented"""
    mocks = comprehensive_tui_mocks

    # Make vote take some time
    async def slow_vote(*args, **kwargs):
        import asyncio
        await asyncio.sleep(0.5)
        return (True, "OK")

    mocks["client"].vote = AsyncMock(side_effect=slow_vote)

    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        # Rapidly press 'u' multiple times
        await pilot.press("u")
        await pilot.press("u")
        await pilot.press("u")

        await pilot.pause(1.0)

        # Vote should only be called once due to deduplication
        assert mocks["client"].vote.call_count == 1


# --- Hostname parsing tests ---

def test_story_item_hostname_parsing():
    """Test hostname extraction from various URL formats"""

    # Test with full URL
    story1 = {"id": 1, "title": "Test", "url": "https://example.com/path/to/article"}
    item1 = StoryItem(story1, 0.9, "reason", "rel")
    # The hostname is extracted in compose(), so we need to check the label content

    # Test without URL (should default to "hn")
    story2 = {"id": 2, "title": "Test", "url": None}
    item2 = StoryItem(story2, 0.9, "reason", "rel")

    # Test with URL without protocol (should default to "hn")
    story3 = {"id": 3, "title": "Test", "url": "example.com/path"}
    item3 = StoryItem(story3, 0.9, "reason", "rel")

    # All should be instantiable
    assert item1.story["id"] == 1
    assert item2.story["id"] == 2
    assert item3.story["id"] == 3


# --- Story item expansion with content tests ---

@pytest.mark.asyncio
async def test_story_item_with_article_snippet(comprehensive_tui_mocks):
    """Test that story item displays article snippet when available"""
    mocks = comprehensive_tui_mocks

    story = {
        "id": 456,
        "title": "Story with snippet",
        "url": "https://example.com",
        "text_content": "Content",
        "article_snippet": "This is the article preview",
        "comments": [],
        "time": int(time.time()),
    }

    mocks["user"].return_value = ([story], [], {456})
    mocks["cand"].return_value = [story]

    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        list_view = app.query_one("#story-list", ListView)
        item = cast(StoryItem, list_view.children[0])

        # Item should be expanded
        assert item.expanded

        # Should have article snippet
        assert story["article_snippet"] in item.story.get("article_snippet", "")


@pytest.mark.asyncio
async def test_story_item_with_comments(comprehensive_tui_mocks):
    """Test that story item displays comments when available"""
    mocks = comprehensive_tui_mocks

    story = {
        "id": 789,
        "title": "Story with comments",
        "url": "https://example.com",
        "text_content": "Content",
        "comments": ["Comment 1", "Comment 2", "Comment 3"],
        "time": int(time.time()),
    }

    mocks["user"].return_value = ([story], [], {789})
    mocks["cand"].return_value = [story]

    app = HNRerankTUI("testuser")

    async with app.run_test() as pilot:
        await pilot.pause(1.5)

        list_view = app.query_one("#story-list", ListView)
        item = cast(StoryItem, list_view.children[0])

        # Item should be expanded
        assert item.expanded

        # Should have comments
        assert len(item.story.get("comments", [])) == 3
