import pytest
import json
import time
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from api import main

TEST_STORY_CACHE = Path(".cache_test/stories")

@pytest.fixture
def temp_story_cache():
    # Setup
    old_cache = main.STORY_CACHE_DIR
    main.STORY_CACHE_DIR = TEST_STORY_CACHE
    TEST_STORY_CACHE.mkdir(parents=True, exist_ok=True)
    yield
    # Teardown
    if TEST_STORY_CACHE.exists():
        shutil.rmtree(TEST_STORY_CACHE)
    main.STORY_CACHE_DIR = old_cache

@pytest.mark.asyncio
async def test_fetch_story_cache_migration(temp_story_cache):
    """Test that cache is refreshed if article_snippet is missing for external URLs."""
    story_id = 999
    # 1. Create a "legacy" cache file (missing article_snippet)
    legacy_data = {
        "retrieved_at": time.time(),
        "data": {
            "id": story_id,
            "title": "Legacy Story",
            "url": "http://external.com/article",
            "score": 100,
            "time": int(time.time()),
            "comments": []
        }
    }
    cache_path = TEST_STORY_CACHE / f"{story_id}.json"
    cache_path.write_text(json.dumps(legacy_data))
    
    mock_client = AsyncMock()
    # Mock Algolia response for the re-fetch
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": story_id,
        "title": "Migrated Story",
        "url": "http://external.com/article",
        "children": []
    }
    
    with patch("api.main.fetch_article_text", return_value="New Snippet") as m_fetch_text:
        mock_client.get = AsyncMock(return_value=mock_resp)
        
        # This should trigger a re-fetch because it's an external URL and article_snippet is missing
        story = await main.fetch_story_with_comments(mock_client, story_id)
        
        assert story is not None
        assert story["title"] == "Migrated Story"
        assert story["article_snippet"] == "New Snippet"
        assert m_fetch_text.call_count == 1

@pytest.mark.asyncio
async def test_fetch_story_cache_hit_valid(temp_story_cache):
    """Test that valid cache (with snippet) is returned without re-fetching."""
    story_id = 888
    valid_data = {
        "retrieved_at": time.time(),
        "data": {
            "id": story_id,
            "title": "Valid Story",
            "url": "http://external.com/article",
            "article_snippet": "Already here",
            "score": 100,
            "time": int(time.time()),
            "comments": []
        }
    }
    cache_path = TEST_STORY_CACHE / f"{story_id}.json"
    cache_path.write_text(json.dumps(valid_data))
    
    mock_client = AsyncMock()
    with patch("api.main.fetch_article_text") as m_fetch_text:
        story = await main.fetch_story_with_comments(mock_client, story_id)
        assert story is not None
        assert story["title"] == "Valid Story"
        assert m_fetch_text.call_count == 0