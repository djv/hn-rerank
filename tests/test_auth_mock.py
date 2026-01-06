import pytest
from unittest.mock import MagicMock, patch
from api.client import HNClient


@pytest.mark.asyncio
async def test_login_flow():
    with (
        patch("httpx.AsyncClient.get") as mock_get,
        patch("httpx.AsyncClient.post") as mock_post,
    ):
        # 1. Login Page Mock (GET)
        mock_resp_login_page = MagicMock()
        mock_resp_login_page.status_code = 200
        mock_resp_login_page.text = '<html><input name="fnid" value="12345"></html>'
        mock_get.return_value = mock_resp_login_page

        # 2. Login Post Mock (POST)
        mock_resp_post = MagicMock()
        mock_resp_post.status_code = 200
        mock_resp_post.text = '<html><a href="logout">logout</a></html>'
        mock_post.return_value = mock_resp_post

        client = HNClient()
        success, msg = await client.login("user", "pass")

        assert success is True
        assert client.username == "user"
        mock_post.assert_called_with(
            "/login",
            data={"acct": "user", "pw": "pass", "fnid": "12345", "goto": "news"},
        )


@pytest.mark.asyncio
async def test_vote_flow():
    with patch("httpx.AsyncClient.get") as mock_get:
        client = HNClient()

        # 1. Item Page Mock (GET)
        # Contains vote link
        mock_resp_item = MagicMock()
        mock_resp_item.status_code = 200
        mock_resp_item.text = (
            '<html><a id="up_100" href="vote?id=100&how=up&auth=secret">vote</a></html>'
        )

        # 2. Vote Request Mock (GET)
        mock_resp_vote = MagicMock()
        mock_resp_vote.status_code = 200

        # Side effect: first call returns item, second call returns vote result
        mock_get.side_effect = [mock_resp_item, mock_resp_vote]

        success, msg = await client.vote(100, "up")

        assert success is True
        assert msg == "Voted successfully"

        # Verify calls
        args_list = mock_get.call_args_list
        assert args_list[0][0][0] == "/item?id=100"
        assert args_list[1][0][0] == "/vote?id=100&how=up&auth=secret"
