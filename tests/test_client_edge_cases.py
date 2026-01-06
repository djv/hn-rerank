
import pytest
import respx
import json
from httpx import Response
from api.client import HNClient, COOKIES_FILE
from unittest.mock import patch, mock_open

@pytest.mark.asyncio
@respx.mock
async def test_load_cookies_exception():
    # Mock open to raise exception
    with patch("builtins.open", side_effect=Exception("Read error")):
        client = HNClient()
        # Should handle gracefully
        assert len(client.client.cookies) == 0

@pytest.mark.asyncio
@respx.mock
async def test_login_page_load_failure():
    client = HNClient()
    respx.get("https://news.ycombinator.com/login").mock(return_value=Response(500))

    success, msg = await client.login("user", "pass")
    assert success is False
    assert "Failed to load" in msg

@pytest.mark.asyncio
@respx.mock
async def test_scrape_list_failure():
    client = HNClient()
    respx.get("https://news.ycombinator.com/favorites?id=user&p=1").mock(return_value=Response(500))

    ids = await client.fetch_favorites("user")
    assert ids == set()

@pytest.mark.asyncio
@respx.mock
async def test_scrape_list_empty():
    client = HNClient()
    respx.get("https://news.ycombinator.com/favorites?id=user&p=1").mock(
        return_value=Response(200, text='<html><body>No rows here</body></html>')
    )

    ids = await client.fetch_favorites("user")
    assert ids == set()

@pytest.mark.asyncio
@respx.mock
async def test_wrappers():
    client = HNClient()
    # fetch_upvoted
    respx.get("https://news.ycombinator.com/upvoted?id=user&p=1").mock(
        return_value=Response(200, text='<tr class="athing" id="101"></tr>')
    )
    assert await client.fetch_upvoted("user") == {101}

    # fetch_hidden
    respx.get("https://news.ycombinator.com/hidden?id=user&p=1").mock(
        return_value=Response(200, text='<tr class="athing" id="102"></tr>')
    )
    assert await client.fetch_hidden("user") == {102}

    # fetch_submitted
    respx.get("https://news.ycombinator.com/submitted?id=user&p=1").mock(
        return_value=Response(200, text='<tr class="athing" id="103"></tr>')
    )
    assert await client.fetch_submitted("user") == {103}

@pytest.mark.asyncio
@respx.mock
async def test_vote_edge_cases():
    client = HNClient()
    item_id = 999

    # 1. Failed to load item page
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(return_value=Response(500))
    success, msg = await client.vote(item_id)
    assert not success
    assert "Failed to load" in msg

    # 2. Already voted (un_ID link present)
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text=f'<a id="un_{item_id}">unvote</a>')
    )
    success, msg = await client.vote(item_id)
    assert not success
    assert "Already voted" in msg

    # 3. Not logged in
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text='<a href="login">login</a>')
    )
    success, msg = await client.vote(item_id)
    assert not success
    assert "Not logged in" in msg

    # 4. Link not found (generic)
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text='<div>Nothing</div>')
    )
    success, msg = await client.vote(item_id)
    assert not success
    assert "Vote link 'up' not found" in msg

    # 5. Vote action failed
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text=f'<a id="up_{item_id}" href="vote?id={item_id}&auth=token"></a>')
    )
    respx.get(f"https://news.ycombinator.com/vote?id={item_id}&auth=token").mock(return_value=Response(500))
    success, msg = await client.vote(item_id)
    assert not success
    assert "Vote failed" in msg

@pytest.mark.asyncio
@respx.mock
async def test_hide_edge_cases():
    client = HNClient()
    item_id = 888

    # 1. Failed to load
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(return_value=Response(500))
    success, msg = await client.hide(item_id)
    assert not success
    assert "Failed to load" in msg

    # 2. Already hidden (unhide link)
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text=f'<a href="unhide?id={item_id}&auth=tok">unhide</a>')
    )
    success, msg = await client.hide(item_id)
    assert not success
    assert "Already hidden" in msg

    # 3. Not logged in
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text='<a href="login">login</a>')
    )
    success, msg = await client.hide(item_id)
    assert not success
    assert "Not logged in" in msg

    # 4. Hide action failed
    respx.get(f"https://news.ycombinator.com/item?id={item_id}").mock(
        return_value=Response(200, text=f'<a href="hide?id={item_id}&auth=token">hide</a>')
    )
    respx.get(f"https://news.ycombinator.com/hide?id={item_id}&auth=token").mock(return_value=Response(500))
    success, msg = await client.hide(item_id)
    assert not success
    assert "Hide failed" in msg
