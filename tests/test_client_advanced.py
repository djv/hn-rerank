import pytest
import respx
from httpx import Response

from api.client import HNClient


@pytest.mark.asyncio
@respx.mock
async def test_client_login_success():
    client = HNClient()

    # Mock GET /login to get fnid
    respx.get("https://news.ycombinator.com/login").mock(return_value=Response(200, text='<input name="fnid" value="test_fnid">'))

    # Mock POST /login
    respx.post("https://news.ycombinator.com/login").mock(return_value=Response(200, text='logout'))

    success, msg = await client.login("user", "pass")
    assert success is True
    assert msg == "Logged in"
    assert client.username == "user"

@pytest.mark.asyncio
@respx.mock
async def test_client_login_failure():
    client = HNClient()
    respx.get("https://news.ycombinator.com/login").mock(return_value=Response(200, text=''))
    respx.post("https://news.ycombinator.com/login").mock(return_value=Response(200, text='login again'))

    success, msg = await client.login("user", "wrong")
    assert success is False
    assert "failed" in msg

@pytest.mark.asyncio
@respx.mock
async def test_client_vote_success():
    client = HNClient()
    item_id = 123

    # Mock item page
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text=f'<a id="up_{item_id}" href="vote?id={item_id}&auth=token"></a> logout')
    )
    # Mock vote action
    respx.get(f"https://news.ycombinator.com/vote?id={item_id}&auth=token").mock(return_value=Response(200))

    success, msg = await client.vote(item_id, "up")
    assert success is True
    assert "successfully" in msg

@pytest.mark.asyncio
@respx.mock
async def test_client_hide_success():
    client = HNClient()
    item_id = 456

    # Mock item page
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text=f'<a href="hide?id={item_id}&auth=token">hide</a> logout')
    )
    # Mock hide action
    respx.get(f"https://news.ycombinator.com/hide?id={item_id}&auth=token").mock(return_value=Response(200))

    success, msg = await client.hide(item_id)
    assert success is True
    assert "successfully" in msg

@pytest.mark.asyncio
@respx.mock
async def test_check_session():
    client = HNClient()
    respx.get("https://news.ycombinator.com/").mock(return_value=Response(200, text='logout'))
    assert await client.check_session() is True

    respx.get("https://news.ycombinator.com/").mock(return_value=Response(200, text='login'))
    assert await client.check_session() is False

@pytest.mark.asyncio
@respx.mock
async def test_client_scrape_list():
    client = HNClient()
    # Mock p1
    respx.get("https://news.ycombinator.com/favorites?id=pg&p=1").mock(
        return_value=Response(200, text='<tr class="athing" id="1"></tr><a class="morelink" href="favorites?id=pg&p=2">More</a>')
    )
    # Mock p2 (no morelink)
    respx.get("https://news.ycombinator.com/favorites?id=pg&p=2").mock(
        return_value=Response(200, text='<tr class="athing" id="2"></tr>')
    )

    ids = await client.fetch_favorites("pg")
    assert ids == {1, 2}
