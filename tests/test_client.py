import pytest
import respx
from httpx import Response
from unittest.mock import patch
from api.client import HNClient

@pytest.mark.asyncio
@respx.mock
async def test_fetch_user_data_mocked():
    """Test fetch_user_data logic by mocking the low-level scraping."""
    username = "testuser"
    base_url = "https://news.ycombinator.com"

    # Mock login check
    respx.get(f"{base_url}/").mock(return_value=Response(200, text="logout"))

    # Mock _scrape_ids calls
    # The _scrape_ids method appends &p=1, &p=2 etc. to the URL if ? is in path, or ?p=1 if not.

    # Favorites: /favorites?id={username} -> /favorites?id={username}&p=1
    respx.get(f"{base_url}/favorites", params={"id": username, "p": "1"}).mock(
        return_value=Response(200, text='<html><tr class="athing" id="1"></tr><tr class="athing" id="2"></tr></html>')
    )
    # Upvoted
    respx.get(f"{base_url}/upvoted", params={"id": username, "p": "1"}).mock(
        return_value=Response(200, text='<html><tr class="athing" id="3"></tr><tr class="athing" id="4"></tr></html>')
    )
    # Hidden
    respx.get(f"{base_url}/hidden", params={"id": username, "p": "1"}).mock(
        return_value=Response(200, text='<html><tr class="athing" id="5"></tr><tr class="athing" id="6"></tr></html>')
    )

    async with HNClient() as client:
        # Force ignore cache
        with patch("pathlib.Path.exists", return_value=False):
             # We also need to mock writing cache to avoid side effects or errors
             with patch("pathlib.Path.write_text"):
                data = await client.fetch_user_data(username)

    assert "pos" in data
    assert "upvoted" in data
    assert "hidden" in data
    assert data["pos"] == {1, 2}
    assert data["upvoted"] == {3, 4}
    assert data["hidden"] == {5, 6}


@pytest.mark.asyncio
@respx.mock
async def test_scrape_ids_logic():
    """Test the low-level _scrape_ids logic with mock HTML."""
    base_url = "https://news.ycombinator.com"
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

    # Page 1: /test?p=1 (if /test has no query params)
    respx.get(f"{base_url}/test", params={"p": "1"}).mock(return_value=Response(200, text=mock_html))
    # Page 2: /test?p=2
    respx.get(f"{base_url}/test", params={"p": "2"}).mock(return_value=Response(200, text=mock_html_page2))

    async with HNClient() as client:
        ids = await client._scrape_ids("/test")
        assert ids == {123, 456, 789}


@pytest.mark.asyncio
@respx.mock
async def test_scrape_ids_empty():
    """Ensure _scrape_ids handles empty pages gracefully."""
    base_url = "https://news.ycombinator.com"
    # /empty?p=1
    respx.get(f"{base_url}/empty", params={"p": "1"}).mock(
        return_value=Response(200, text="<html>No items here</html>")
    )

    async with HNClient() as client:
        ids = await client._scrape_ids("/empty")
        assert ids == set()
