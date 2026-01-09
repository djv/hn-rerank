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
            {
                "type": "comment",
                "text": f"Comment {i} for story {sid} with enough text to pass the minimum length filter.",
                "children": [],
            }
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
        mock_cache.__truediv__.return_value.exists.return_value = False
        # Mock atomic write and eviction to avoid temp file issues
        mp.setattr("api.fetching._atomic_write_json", lambda p, d: None)
        mp.setattr("api.fetching._evict_old_cache_files", lambda: None)

        stories = await get_best_stories(limit=limit, exclude_ids=exclude)

        assert len(stories) == 2
        ids = [s["id"] for s in stories]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids


@pytest.mark.asyncio
@respx.mock
async def test_get_best_stories_pagination():
    """Test that pagination works for >1000 candidates."""
    # Mock page 0
    respx.get(f"{ALGOLIA_BASE}/search").mock(
        side_effect=[
            Response(
                200,
                json={
                    "hits": [{"objectID": str(i)} for i in range(1000)],
                    "nbPages": 2,
                },
            ),
            Response(
                200,
                json={
                    "hits": [{"objectID": str(i)} for i in range(1000, 1200)],
                    "nbPages": 2,
                },
            ),
        ]
    )

    # Mock item fetches - all return valid stories with 10+ comments
    comments = [
        {
            "type": "comment",
            "text": f"Comment {i} with enough text to pass the minimum length filter requirement.",
            "children": [],
        }
        for i in range(12)
    ]
    respx.get(url__regex=rf"{ALGOLIA_BASE}/items/\d+").mock(
        return_value=Response(
            200,
            json={
                "id": 1,
                "type": "story",
                "title": "Test Story",
                "url": "http://example.com",
                "points": 100,
                "created_at_i": 1600000000,
                "children": comments,
            },
        )
    )

    with pytest.MonkeyPatch().context() as mp:
        mock_cache = MagicMock()
        mp.setattr("api.fetching.CACHE_PATH", mock_cache)
        mock_cache.__truediv__.return_value.exists.return_value = False
        mp.setattr("api.fetching._atomic_write_json", lambda p, d: None)
        mp.setattr("api.fetching._evict_old_cache_files", lambda: None)

        stories = await get_best_stories(limit=1100)
        # Should have paginated to get >1000
        assert len(stories) == 1100
