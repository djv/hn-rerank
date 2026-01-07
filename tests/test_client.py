import pytest
from unittest.mock import patch, MagicMock
from api.client import HNClient


@pytest.mark.asyncio
async def test_fetch_user_data_mocked():
    """Test fetch_user_data logic by mocking the low-level scraping."""
    username = "testuser"
    # Mocking _scrape_ids to return dummy sets
    with patch("api.client.HNClient._scrape_ids") as mock_scrape:
        mock_scrape.side_effect = [
            {1, 2},  # favorites
            {3, 4},  # upvoted
            {5, 6},  # hidden
        ]

        async with HNClient() as client:
            with patch("pathlib.Path.exists", return_value=False):
                data = await client.fetch_user_data(username)
                assert "pos" in data
                assert "upvoted" in data
                assert "hidden" in data
                assert data["pos"] == {1, 2}
                assert data["upvoted"] == {3, 4}
                assert data["hidden"] == {5, 6}

                # Verify scrape calls
                mock_scrape.assert_any_call(f"/favorites?id={username}")
                mock_scrape.assert_any_call(f"/upvoted?id={username}")
                mock_scrape.assert_any_call(f"/hidden?id={username}")


@pytest.mark.asyncio
async def test_scrape_ids_logic():
    """Test the low-level _scrape_ids logic with mock HTML."""
    mock_html = """
    <html>
        <tr class="athing" id="123"></tr>
        <tr class="athing" id="456"></tr>
        <a class="morelink" href="?p=2">More</a>
    </html>
    """
    mock_html_page2 = """
    <html>
        <tr class="athing" id="789"></tr>
    </html>
    """

    async with HNClient() as client:
        with patch.object(client.client, "get") as mock_get:
            # Page 1 then Page 2
            mock_get.side_effect = [
                MagicMock(status_code=200, text=mock_html),
                MagicMock(status_code=200, text=mock_html_page2),
            ]

            ids = await client._scrape_ids("/test")
            assert ids == {123, 456, 789}
            assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_scrape_ids_empty():
    """Ensure _scrape_ids handles empty pages gracefully."""
    async with HNClient() as client:
        with patch.object(client.client, "get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, text="<html>No items here</html>"
            )
            ids = await client._scrape_ids("/empty")
            assert ids == set()
