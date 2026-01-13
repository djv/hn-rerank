import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, UTC
from api import fetching
from api.constants import CANDIDATE_CACHE_TTL_ARCHIVE, CANDIDATE_CACHE_TTL_SHORT

@pytest.mark.asyncio
async def test_candidate_cache_ttl_logic():
    """
    Test that windows older than 30 days use the ARCHIVE TTL.
    """
    # Fix "now" to a specific time
    fixed_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    
    # We want to inspect calls to get_cached_candidates
    # It takes (key, ttl)
    
    with patch("api.fetching.datetime") as mock_dt, \
         patch("api.fetching.get_cached_candidates") as mock_get_cache, \
         patch("api.fetching.save_cached_candidates"), \
         patch("api.fetching.httpx.AsyncClient") as mock_client_cls, \
         patch("api.fetching.fetch_story"):
        
        mock_dt.now.return_value = fixed_now
        # We need mock_dt to also delegate other methods if they are used, 
        # but fetching.py imports datetime class. 
        # Actually fetching.py does `from datetime import UTC, datetime, timedelta`.
        # So we patch `api.fetching.datetime`.
        
        mock_get_cache.return_value = None # Always miss, so we can see what TTL was requested
        
        # Mock client response to avoid actual HTTP
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        
        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"{}" # make boolean check passed
            resp.json.return_value = {"hits": []}
            return resp
            
        mock_client.get = mock_get
        mock_client_cls.return_value = mock_client
        
        # We ask for 40 days of stories
        await fetching.get_best_stories(limit=10, days=40)
        
        # Now analyze the calls to get_cached_candidates(key, ttl)
        # We expect multiple calls for different windows.
        
        # 1. Live window (recent) -> Short TTL
        # 2. Recent archive windows (< 30 days) -> 7 days TTL (default logic in current code)
        # 3. Old archive windows (> 30 days) -> Should be ARCHIVE TTL (what we want to implement)
        
        # Let's see what we get currently.
        # Current logic: ttl = CANDIDATE_CACHE_TTL_SHORT if is_live else (7 * 86400)
        
        # Calculate expected timestamps
        # Anchor is last Monday. 
        # fixed_now is Wed Jan 01 2025. Monday was Dec 30 2024.
        # So Anchor is Dec 30 2024.
        
        # Window 1: Live (Dec 30 - Jan 01)
        # Window 2: Archive (Dec 23 - Dec 30)
        # ...
        # Window N: Old (e.g. Nov 01 - Nov 08)
        
        calls = mock_get_cache.call_args_list
        assert len(calls) > 0
        
        found_archive_ttl = False
        found_short_ttl = False
        found_weekly_ttl = False
        
        for call in calls:
            args, _ = call
            ttl = args[1]
            
            if ttl == CANDIDATE_CACHE_TTL_SHORT:
                found_short_ttl = True
            elif ttl == 7 * 86400:
                found_weekly_ttl = True
            elif ttl == CANDIDATE_CACHE_TTL_ARCHIVE:
                found_archive_ttl = True
                
        # With current code, we expect Short and Weekly, but NOT Archive
        print(f"Found TTLs: Short={found_short_ttl}, Weekly={found_weekly_ttl}, Archive={found_archive_ttl}")
        
        # Assertion for the NEW behavior (which should fail now)
        assert found_archive_ttl, "Should use ARCHIVE TTL for old windows"

if __name__ == "__main__":
    # Manually run if needed
    pass
