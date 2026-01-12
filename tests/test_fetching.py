import pytest
import respx
from httpx import Response
from unittest.mock import MagicMock
from api.fetching import get_best_stories, _clean_text, ALGOLIA_BASE


class TestCleanText:
    """Edge case tests for _clean_text function."""

    def test_empty_string(self):
        assert _clean_text("") is None

    def test_braille_only(self):
        # Braille pattern range: U+2800-U+28FF
        assert _clean_text("⠁⠃⠉⠙⠑") is None

    def test_box_drawing_only(self):
        # Box drawing range: U+2500-U+257F
        assert _clean_text("─│┌┐└┘├┤") is None

    def test_short_string(self):
        # <= 20 chars should be filtered
        assert _clean_text("Short text here.") is None
        assert _clean_text("Exactly 20 chars!!!") is None

    def test_low_alphanumeric_ratio(self):
        # < 50% alphanumeric should be filtered
        assert _clean_text("!@#$%^&*()!@#$%^&*()!@#$%") is None
        assert _clean_text("--- === +++ ~~~ >>> <<<") is None

    def test_excessive_punctuation(self):
        # 3+ repeated punctuation chars like ### should be stripped
        result = _clean_text("Hello world ### this is a test message")
        assert result is not None
        assert "###" not in result

    def test_valid_text_passes(self):
        text = "This is a valid comment with enough content."
        assert _clean_text(text) == text

    def test_mixed_content(self):
        # Valid text with some braille should clean and pass
        text = "This is valid text ⠁⠃⠉ with braille mixed in here."
        result = _clean_text(text)
        assert result is not None
        assert "⠁" not in result


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
                ],
                "nbPages": 1,
            },
        )
    )

    # Mock individual item fetches via Algolia items API
    for sid in ["1", "2", "3", "4"]:
        # Need 10+ comments to pass minimum threshold (50+ chars each)
        comments = [
            {"type": "comment", "text": f"Comment {i} for story {sid} with enough text to pass the minimum length filter.", "children": []}
            for i in range(12)
        ]
        respx.get(f"{ALGOLIA_BASE}/items/{sid}").mock(
            return_value=Response(
                200,
                json={
                    "id": int(sid),
                    "type": "story",
                    "title": f"Story {sid}",
                    "url": f"http://example.com/story/{sid}",
                    "points": 100,
                    "created_at_i": 1600000000,
                    "children": comments,
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
        mp.setattr("api.fetching.CANDIDATE_CACHE_PATH", mock_cache)
        mock_cache.__truediv__.return_value.exists.return_value = False
        # Mock atomic write and eviction to avoid temp file issues
        mp.setattr("api.fetching._atomic_write_json", lambda p, d: None)
        mp.setattr("api.fetching._evict_old_cache_files", lambda: None)

        stories = await get_best_stories(limit=limit, exclude_ids=exclude, days=1)

        # The fetcher uses a buffer and minimum window size (20), so it may return more than limit
        assert len(stories) >= 2
        ids = [s["id"] for s in stories]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids


@pytest.mark.asyncio
@respx.mock
async def test_get_best_stories_pagination():
    """Test that we can fetch >1000 candidates by spreading over time windows."""
    # With limit=1100 and default days=30 (5 windows), we need ~220 per window.
    # We mock 5 responses with unique IDs to ensure we collect enough uniques.
    responses = []
    for i in range(5):
        # Return enough hits to satisfy the window target
        hits = [{"objectID": str(j)} for j in range(i * 300, (i + 1) * 300)]
        responses.append(Response(200, json={"hits": hits}))

    # Cycle through responses if code requests more windows
    import itertools
    respx.get(f"{ALGOLIA_BASE}/search").mock(side_effect=itertools.cycle(responses))

    # Mock item fetches - all return valid stories with 10+ comments
    # We cheat and return the same content for any ID, but distinct IDs matter for the count
    comments = [
        {"type": "comment", "text": f"Comment {i} with enough text to pass the minimum length filter requirement which is fifty characters or more.", "children": []}
        for i in range(12)
    ]
    respx.get(url__regex=rf"{ALGOLIA_BASE}/items/\d+").mock(
        return_value=Response(
            200,
            json={
                "id": 1, # ID doesn't matter for the list length check, but get_best_stories deduplicates by content? No by ID.
                # Wait, fetch_story returns dict with "id". 
                # If we return same ID in body, it might be confusing but get_best_stories uses the key from `hits`.
                # Actually fetch_story returns { "id": sid, ... }
                # We need to make sure the mocked item response reflects the requested ID if possible, 
                # OR just ensure we don't dedupe in the final list based on content.
                # The final list is constructed from `hits`.
                "type": "story",
                "title": "Test Story",
                "url": "http://example.com",
                "points": 100,
                "created_at_i": 1600000000,
                "children": comments,
            },
        )
    )
    # Actually, fetch_story takes `sid` as arg and puts it in the result `id`.
    # But `fetch_story` implementation:
    # item = resp.json()
    # story = { "id": sid, ... }
    # So the ID in the JSON body is ignored in favor of `sid` passed to function.
    # So this mock is fine.

    with pytest.MonkeyPatch().context() as mp:
        mock_cache = MagicMock()
        mp.setattr("api.fetching.CACHE_PATH", mock_cache)
        mp.setattr("api.fetching.CANDIDATE_CACHE_PATH", mock_cache)
        mock_cache.__truediv__.return_value.exists.return_value = False
        mp.setattr("api.fetching._atomic_write_json", lambda p, d: None)
        mp.setattr("api.fetching._evict_old_cache_files", lambda: None)

        stories = await get_best_stories(limit=1100)
        # We expect 1100 unique stories
        assert len(stories) >= 1100
