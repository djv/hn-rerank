import json
from unittest.mock import AsyncMock, patch
import pytest
from api import rerank
from api.models import StoryDict


def make_story(story_id: int, title: str) -> StoryDict:
    return {
        "id": story_id,
        "title": title,
        "url": None,
        "score": 0,
        "time": 0,
        "comments": [],
        "text_content": title,
    }


@pytest.fixture
def temp_cache_file(tmp_path):
    cache_file = tmp_path / "cluster_names.json"
    with patch("api.rerank.CLUSTER_NAME_CACHE_PATH", cache_file):
        yield cache_file


@pytest.mark.asyncio
async def test_cluster_name_cache_hit(temp_cache_file):
    """Test that cached name is returned if present."""
    items: list[tuple[StoryDict, float]] = [(make_story(123, "Story 1"), 1.0)]
    clusters: dict[int, list[tuple[StoryDict, float]]] = {0: items}

    # Calculate expected hash
    story_ids = sorted([str(s.get("id")) for s, _ in items])
    cache_key = rerank._cluster_name_cache_key(
        story_ids, rerank.LLM_CLUSTER_NAME_MODEL_PRIMARY
    )

    # Seed cache
    cache_content = {cache_key: "Cached Cluster Name"}
    temp_cache_file.write_text(json.dumps(cache_content))

    # Call function
    names = await rerank.generate_batch_cluster_names(clusters)

    assert names[0] == "Cached Cluster Name"


@pytest.mark.asyncio
async def test_cluster_name_cache_miss_and_save(temp_cache_file):
    """Test that name is generated and saved on cache miss."""
    items: list[tuple[StoryDict, float]] = [
        (make_story(456, "New Cluster Name Spotlight"), 1.0),
        (make_story(789, "Cluster Name Deep Dive"), 0.8),
    ]
    clusters: dict[int, list[tuple[StoryDict, float]]] = {0: items}

    # Mock API via internal helper to avoid real HTTP
    mock_gen = AsyncMock(return_value="New Cluster Name")
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "fake_key"}),
        patch("api.rerank._generate_with_retry", new=mock_gen),
    ):
        # Call function
        names = await rerank.generate_batch_cluster_names(clusters)

        assert names[0] == "New Cluster Name"
        assert mock_gen.await_count >= 1

        # Verify cache was updated
        cache_content = json.loads(temp_cache_file.read_text())

        story_ids = sorted([str(s.get("id")) for s, _ in items])
        cache_key = rerank._cluster_name_cache_key(
            story_ids, rerank.LLM_CLUSTER_NAME_MODEL_PRIMARY
        )

        assert cache_key in cache_content
        assert cache_content[cache_key] == "New Cluster Name"


@pytest.mark.asyncio
async def test_cluster_name_fallback_no_api_key(temp_cache_file):
    """Test that missing API key raises an error when naming is required."""
    items: list[tuple[StoryDict, float]] = [(make_story(789, "Story 3"), 1.0)]
    clusters: dict[int, list[tuple[StoryDict, float]]] = {0: items}

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError):
            await rerank.generate_batch_cluster_names(clusters)
