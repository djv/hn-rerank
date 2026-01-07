import pytest
import respx
from httpx import Response
from unittest.mock import MagicMock
from api.fetching import get_best_stories, ALGOLIA_BASE


@pytest.mark.asyncio
@respx.mock
async def test_get_best_stories_filtering():
    """
    Test that get_best_stories correctly excludes IDs and respects limits.
    """
    # Mock Algolia Search Response
    search_url = f"{ALGOLIA_BASE}/search"
    respx.get(search_url).mock(
        return_value=Response(
            200,
            json={
                "hits": [
                    {"objectID": "1"},
                    {"objectID": "2"},
                    {"objectID": "3"},
                    {"objectID": "4"},
                ]
            },
        )
    )

    # Mock individual item fetches
    for sid in ["1", "2", "3", "4"]:
        respx.get(f"{ALGOLIA_BASE}/items/{sid}").mock(
            return_value=Response(
                200,
                json={
                    "id": int(sid),
                    "title": f"Story {sid}",
                    "points": 100,
                    "created_at_i": 1600000000,
                    "children": [],
                },
            )
        )

    # Exclude ID 2, limit to 2 stories
    exclude = {2}
    limit = 2

    # We must patch the cache to avoid side effects
    with pytest.MonkeyPatch().context() as mp:
        mock_cache = MagicMock()
        mp.setattr("api.fetching.CACHE_PATH", mock_cache)
        mock_cache.__truediv__.return_value.exists.return_value = False

        stories = await get_best_stories(limit=limit, exclude_ids=exclude)

        assert len(stories) == 2
        ids = [s["id"] for s in stories]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids
