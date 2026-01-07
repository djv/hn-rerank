import pytest
from unittest.mock import patch, MagicMock
from api.client import HNClient

@pytest.mark.asyncio
async def test_fetch_user_data_signals():
    """Test that fetch_user_data correctly scrapes favorites, upvotes, and hidden stories."""
    username = "testuser"
    
    with patch("api.client.HNClient._scrape_ids") as mock_scrape, \
         patch("httpx.AsyncClient.get") as mock_get:
        
        # Mock responses for different signal paths
        # side_effect order: 
        # 1. session check (mock_get)
        # 2. favorites (mock_scrape)
        # 3. upvotes (mock_scrape)
        # 4. hidden (mock_scrape)
        
        mock_scrape.side_effect = [{1}, {2}, {3}]
        
        # Mock session check (implies logged in)
        mock_session_resp = MagicMock()
        mock_session_resp.text = "logout"
        mock_session_resp.status_code = 200
        mock_get.return_value = mock_session_resp
        
        async with HNClient() as client:
            with patch("pathlib.Path.exists", return_value=False):
                data = await client.fetch_user_data(username)
                
                # pos should combine favorites and upvotes
                assert data["pos"] == {1, 2}
                # neg should contain hidden
                assert data["neg"] == {3}
                
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
                MagicMock(status_code=200, text=mock_html_page2)
            ]
            
            ids = await client._scrape_ids("/test")
            assert ids == {123, 456, 789}
            assert mock_get.call_count == 2
