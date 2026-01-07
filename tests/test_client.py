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
