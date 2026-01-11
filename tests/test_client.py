import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
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
            # Mock login check to return logged in
            with patch.object(client.client, "get") as mock_get:
                mock_get.return_value = MagicMock(text="logout")
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

@pytest.fixture
def mock_httpx_client():
    with patch("api.client.httpx.AsyncClient") as mock:
        yield mock

@pytest.fixture
def hn_client(mock_httpx_client):
    return HNClient()

@pytest.mark.asyncio
async def test_load_cookies(tmp_path):
    cookies = {"session": "test_cookie"}
    cookie_file = tmp_path / "cookies.json"
    cookie_file.write_text(json.dumps(cookies))

    with patch("api.client.COOKIES_FILE", cookie_file):
        # We need to mock httpx.AsyncClient so it returns a MagicMock
        # that has a .cookies attribute
        with patch("api.client.httpx.AsyncClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.cookies = MagicMock()

            client = HNClient()
            # Verify cookies are loaded into the client
            client.client.cookies.update.assert_called_with(cookies)

@pytest.mark.asyncio
async def test_login_success(hn_client):
    # Mock GET /login response
    mock_get_resp = MagicMock()
    mock_get_resp.text = '<html><input name="fnid" value="test_fnid"></html>'
    hn_client.client.get = AsyncMock(return_value=mock_get_resp)

    # Mock POST /login response
    mock_post_resp = MagicMock()
    mock_post_resp.text = "logout" # Indicates successful login
    hn_client.client.post = AsyncMock(return_value=mock_post_resp)

    # Mock cookies
    hn_client.client.cookies = {"user": "testuser"}

    with patch("api.client.COOKIES_FILE") as mock_cookie_file:
        mock_cookie_file.parent.mkdir.return_value = None

        success, msg = await hn_client.login("user", "pass")

        assert success is True
        assert msg == "Success"
        mock_cookie_file.write_text.assert_called()

@pytest.mark.asyncio
async def test_login_failure(hn_client):
    # Mock GET /login
    mock_get_resp = MagicMock()
    mock_get_resp.text = '<html><input name="fnid" value="test_fnid"></html>'
    hn_client.client.get = AsyncMock(return_value=mock_get_resp)

    # Mock POST /login failure
    mock_post_resp = MagicMock()
    mock_post_resp.text = "Bad login"
    hn_client.client.post = AsyncMock(return_value=mock_post_resp)

    success, msg = await hn_client.login("user", "wrongpass")

    assert success is False
    assert "Bad login" in msg

@pytest.mark.asyncio
async def test_fetch_user_data_cached(hn_client, tmp_path):
    user = "cached_user"
    cache_dir = tmp_path
    cache_file = cache_dir / f"{user}.json"

    import time
    data = {
        "ts": time.time(),
        "ids": {
            "pos": [1, 2],
            "upvoted": [3],
            "hidden": []
        }
    }
    cache_file.write_text(json.dumps(data))

    with patch("api.client.USER_CACHE_DIR_PATH", cache_dir):
        # We need to ensure client.get is not called, so we can mock it to raise error
        hn_client.client.get = AsyncMock(side_effect=Exception("Should not be called"))

        result = await hn_client.fetch_user_data(user)
        assert result == {
            "pos": {1, 2},
            "upvoted": {3},
            "hidden": set()
        }

@pytest.mark.asyncio
async def test_fetch_user_data_live(hn_client, tmp_path):
    user = "live_user"
    cache_dir = tmp_path

    # Mock responses
    # 1. Login check
    mock_login_check = MagicMock()
    mock_login_check.text = "logout" # Logged in

    # 2. Favorites
    mock_fav = MagicMock()
    mock_fav.text = '<tr class="athing" id="101"></tr>'

    # 3. Upvoted
    mock_up = MagicMock()
    mock_up.text = '<tr class="athing" id="102"></tr>'

    # 4. Hidden
    mock_hidden = MagicMock()
    mock_hidden.text = '<tr class="athing" id="103"></tr>'

    hn_client.client.get = AsyncMock(side_effect=[
        mock_login_check,
        mock_fav,
        mock_up,
        mock_hidden
    ])

    with patch("api.client.USER_CACHE_DIR_PATH", cache_dir):
        result = await hn_client.fetch_user_data(user)

        assert result["pos"] == {101}
        assert result["upvoted"] == {102}
        assert result["hidden"] == {103}

        # Verify cache was written
        cache_file = cache_dir / f"{user}.json"
        assert cache_file.exists()

@pytest.mark.asyncio
async def test_scrape_ids_pagination(hn_client):
    # Mock pagination
    # Page 1: has morelink
    page1 = MagicMock()
    page1.text = '<tr class="athing" id="1"></tr><a class="morelink" href="next">More</a>'

    # Page 2: no morelink
    page2 = MagicMock()
    page2.text = '<tr class="athing" id="2"></tr>'

    hn_client.client.get = AsyncMock(side_effect=[page1, page2])

    ids = await hn_client._scrape_ids("/test", max_pages=2)
    assert ids == {1, 2}
    assert hn_client.client.get.call_count == 2
