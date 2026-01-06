import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from api.main import get_best_stories
from datetime import datetime, timezone, timedelta


@pytest.mark.asyncio
async def test_get_best_stories_logic():
    """
    Test that get_best_stories calls Algolia with correct params
    and parses the result.
    """
    mock_client = AsyncMock()

    # Mock Algolia Response
    mock_algolia_resp = MagicMock()
    mock_algolia_resp.status_code = 200
    mock_algolia_resp.json.return_value = {
        "hits": [{"objectID": "1001"}, {"objectID": "1002"}]
    }

    # Mock Story Details (Official API fallback)
    # The function calls fetch_story_with_comments for each hit
    mock_story_resp = {"id": 1001, "title": "Test Story"}

    with patch("httpx.AsyncClient", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_algolia_resp

        with patch("api.main.fetch_story_with_comments", return_value=mock_story_resp):
            results = await get_best_stories(limit=50)

            # Check results
            assert len(results) == 2  # 2 hits mocked

            # Check Algolia Call
            call_args = mock_client.get.call_args
            url = call_args[0][0]
            params = call_args[1]["params"]

            assert "hn.algolia.com" in url
            assert params["hitsPerPage"] == 50
            assert "created_at_i>" in params["numericFilters"]

            # Verify timestamp is roughly 30 days ago
            # Format: "created_at_i>TIMESTAMP,points>20"
            filter_str = params["numericFilters"]
            timestamp_part = filter_str.split(",")[0]  # "created_at_i>TIMESTAMP"
            sent_ts = int(timestamp_part.split(">")[1])

            expected_ts = int(
                (datetime.now(timezone.utc) - timedelta(days=30)).timestamp()
            )

            # Allow 5s delta for test execution time
            assert abs(sent_ts - expected_ts) < 5
