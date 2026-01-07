import pytest
from unittest.mock import AsyncMock, patch
from api.rerank import compute_recency_weights
from api.fetching import fetch_story

def test_recency_weights():
    now = 1000000
    timestamps = [now, now - 86400 * 10]
    with patch("time.time", return_value=now):
        weights = compute_recency_weights(timestamps)
        assert weights[0] == 1.0
        assert weights[1] < 1.0

@pytest.mark.asyncio
async def test_fetch_story_cached():
    mock_client = AsyncMock()
    sid = 123
    story = {"id": sid, "title": "Test"}
    
    with patch("api.fetching.CACHE_PATH") as mock_path:
        mock_file = mock_path.__truediv__.return_value
        mock_file.exists.return_value = True
        import json
        import time
        mock_file.read_text.return_value = json.dumps({"ts": time.time(), "story": story})
        
        res = await fetch_story(mock_client, sid)
        assert res["id"] == sid
        assert mock_client.get.call_count == 0
