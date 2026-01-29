import pytest
from unittest.mock import patch, MagicMock
from api.client import HNClient


@pytest.mark.asyncio
async def test_fetch_user_data_mocked():
    """Test fetch_user_data logic by mocking the low-level scraping."""
    username = "testuser"

    async with HNClient() as client:
        # Mock HTTP response with logout link and username
        mock_resp = MagicMock(
            text=f'<html><a id="me">{username}</a> | <a>logout</a></html>'
        )

        with patch.object(client.client, "get", return_value=mock_resp):
            with patch.object(client, "_scrape_items") as mock_items:
                with patch.object(client, "_scrape_ids") as mock_ids:
                    # _scrape_items: hidden (always fresh), then favorites (cache miss)
                    mock_items.side_effect = [
                        ({5, 6}, {"http://hidden.com"}),  # hidden
                        ({1, 2}, {"http://fav.com"}),  # favorites
                    ]
                    # _scrape_ids: upvoted
                    mock_ids.return_value = {3, 4}

                    with patch("pathlib.Path.exists", return_value=False):
                        data = await client.fetch_user_data(username)

                        assert "pos" in data
                        assert "upvoted" in data
                        assert "hidden" in data
                        # pos = (favorites | upvoted) - hidden = ({1,2} | {3,4}) - {5,6}
                        assert data["pos"] == {1, 2, 3, 4}
                        assert data["upvoted"] == {3, 4}
                        assert data["hidden"] == {5, 6}

                        # Verify scrape calls
                        mock_items.assert_any_call(f"/hidden?id={username}", max_pages=20)
                        mock_items.assert_any_call(f"/favorites?id={username}", max_pages=15)
                        mock_ids.assert_called_with(f"/upvoted?id={username}", max_pages=15)


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
