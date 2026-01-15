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
    clusters = {0: items}

    # Calculate expected hash
    story_ids = sorted([str(s.get("id")) for s, _ in items])
    cache_key = hashlib.sha256(",".join(story_ids).encode()).hexdigest()

    # Seed cache
    cache_content = {cache_key: "Cached Cluster Name"}
    temp_cache_file.write_text(json.dumps(cache_content))

    # Call function
    names = await rerank.generate_batch_cluster_names(clusters)

    assert names[0] == "Cached Cluster Name"


@pytest.mark.asyncio
async def test_cluster_name_cache_miss_and_save(temp_cache_file):
    """Test that name is generated and saved on cache miss."""
    items = [({"id": 456, "title": "Story 2"}, 1.0)]
    clusters = {0: items}

    # Mock API via httpx
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "fake_key"}),
        patch("httpx.AsyncClient.post") as mock_post,
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"0": "New Cluster Name"}'}}]
        }
        mock_post.return_value = mock_resp

        # Call function
        names = await rerank.generate_batch_cluster_names(clusters)

        assert names[0] == "New Cluster Name"

        # Verify cache was updated
        cache_content = json.loads(temp_cache_file.read_text())

        story_ids = sorted([str(s.get("id")) for s, _ in items])
        cache_key = hashlib.sha256(",".join(story_ids).encode()).hexdigest()

        assert cache_key in cache_content
        assert cache_content[cache_key] == "New Cluster Name"


@pytest.mark.asyncio
async def test_cluster_name_fallback_no_api_key(temp_cache_file):
    """Test that fallback name is returned if API key is missing."""
    items = [({"id": 789, "title": "Story 3"}, 1.0)]
    clusters = {0: items}

    with patch.dict("os.environ", {}, clear=True):
        names = await rerank.generate_batch_cluster_names(clusters)

        # Should get fallback (truncated title or "Misc")
        assert names[0] in ["Story 3", "Misc"]
