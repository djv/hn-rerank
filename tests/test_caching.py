
import json
import hashlib
from unittest.mock import patch, MagicMock
import pytest
from api import rerank

@pytest.fixture
def temp_cache_file(tmp_path):
    cache_file = tmp_path / "cluster_names.json"
    with patch("api.rerank.CLUSTER_NAME_CACHE_PATH", cache_file):
        yield cache_file

@pytest.mark.asyncio
async def test_cluster_name_cache_hit(temp_cache_file):
    """Test that cached name is returned if present."""
    items = [({"id": 123, "title": "Story 1"}, 1.0)]
    
    # Calculate expected hash
    story_ids = sorted([str(s.get("id")) for s, _ in items])
    cache_key = hashlib.sha256(",".join(story_ids).encode()).hexdigest()
    
    # Seed cache
    cache_content = {cache_key: "Cached Cluster Name"}
    temp_cache_file.write_text(json.dumps(cache_content))
    
    # Call function
    name = await rerank.generate_single_cluster_name(items)
    
    assert name == "Cached Cluster Name"

@pytest.mark.asyncio
async def test_cluster_name_cache_miss_and_save(temp_cache_file):
    """Test that name is generated and saved on cache miss."""
    items = [({"id": 456, "title": "Story 2"}, 1.0)]
    
    # Mock API via httpx
    with patch.dict("os.environ", {"GROQ_API_KEY": "fake_key"}), \
         patch("httpx.AsyncClient.post") as mock_post:
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "New Cluster Name"}}]}
        mock_post.return_value = mock_resp
        
        # Call function
        name = await rerank.generate_single_cluster_name(items)
        
        assert name == "New Cluster Name"
        
        # Verify cache was updated
        cache_content = json.loads(temp_cache_file.read_text())
        
        story_ids = sorted([str(s.get("id")) for s, _ in items])
        cache_key = hashlib.sha256(",".join(story_ids).encode()).hexdigest()
        
        assert cache_key in cache_content
        assert cache_content[cache_key] == "New Cluster Name"

@pytest.mark.asyncio
async def test_cluster_name_cache_miss_no_api_key(temp_cache_file):
    """Test that 'Misc' is returned and NOT cached if API key is missing."""
    items = [({"id": 789, "title": "Story 3"}, 1.0)]
    
    with patch.dict("os.environ", {}, clear=True):
        name = await rerank.generate_single_cluster_name(items)
        
        assert name == "Misc"
        
        # Verify cache was NOT created/updated
        if temp_cache_file.exists():
            # If file exists (maybe from other tests if not isolated? tmp_path should be isolated),
            # check key is not there.
            # But here `temp_cache_file` is fresh per test if fixture works right.
            # Actually, `temp_cache_file` doesn't exist initially unless written.
            # Wait, `_load_cluster_name_cache` doesn't create file, `_save` does.
            # If we don't save, file shouldn't be created if it didn't exist.
            assert not temp_cache_file.exists() or temp_cache_file.read_text() == "{}"

