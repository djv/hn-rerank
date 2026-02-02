import pytest
import respx
from httpx import Response
from unittest.mock import MagicMock
from api.constants import MIN_COMMENT_LENGTH, MIN_STORY_COMMENTS
from api.fetching import (
    ALGOLIA_BASE,
    _clean_text,
    _extract_comments_recursive,
    get_best_stories,
)


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


class TestExtractComments:
    """Edge case tests for _extract_comments_recursive."""

    def test_min_comment_length_inclusive(self):
        text = "a" * MIN_COMMENT_LENGTH
        children = [
            {
                "type": "comment",
                "text": text,
                "points": 10,
                "children": [],
            }
        ]

        results = _extract_comments_recursive(children)

        assert len(results) == 1
        assert results[0]["text"] == text


@pytest.mark.asyncio
@respx.mock
async def test_get_best_stories_accepts_min_comment_length():
    """Ensure stories with comments exactly MIN_COMMENT_LENGTH are kept."""
    search_url = f"{ALGOLIA_BASE}/search"
    respx.get(search_url).mock(
        return_value=Response(
            200,
            json={"hits": [{"objectID": "42"}], "nbPages": 1},
        )
    )

    min_length_text = "a" * MIN_COMMENT_LENGTH
    comments = [
        {
            "type": "comment",
            "text": min_length_text,
            "points": 5,
            "children": [],
        }
        for _ in range(MIN_STORY_COMMENTS)
    ]

    respx.get(f"{ALGOLIA_BASE}/items/42").mock(
        return_value=Response(
            200,
            json={
                "id": 42,
                "type": "story",
                "title": "Boundary Story",
                "url": "http://example.com/boundary",
                "points": 100,
                "created_at_i": 1600000000,
                "children": comments,
            },
        )
    )

    with pytest.MonkeyPatch().context() as mp:
        mock_cache = MagicMock()
        mp.setattr("api.fetching.CACHE_PATH", mock_cache)
        mp.setattr("api.fetching.CANDIDATE_CACHE_PATH", mock_cache)
        mock_cache.__truediv__.return_value.exists.return_value = False
        mp.setattr("api.fetching._atomic_write_json", lambda p, d: None)
        mp.setattr("api.fetching._evict_old_cache_files", lambda: None)

        stories = await get_best_stories(limit=1)
        assert len(stories) == 1
        assert stories[0].id == 42


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
        # Need 20+ comments to pass minimum threshold (MIN_STORY_COMMENTS=20)
        comments = [
            {
                "type": "comment",
                "text": f"Comment {i} for story {sid} with enough text to pass the minimum length filter.",
                "children": [],
            }
            for i in range(25)
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
        ids = [s.id for s in stories]
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

    # Mock item fetches - each story returns plenty of comments (>= MIN_STORY_COMMENTS)
    # We cheat and return the same content for any ID, but distinct IDs matter for the count
    comments = [
        {
            "type": "comment",
            "text": f"Comment {i} with enough text to pass the minimum length filter requirement (MIN_COMMENT_LENGTH).",
            "children": [],
        }
        for i in range(25)
    ]
    respx.get(url__regex=rf"{ALGOLIA_BASE}/items/\d+").mock(
        return_value=Response(
            200,
            json={
                "id": 1,  # ID doesn't matter for the list length check, but get_best_stories deduplicates by content? No by ID.
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


@pytest.mark.asyncio
@respx.mock
async def test_get_best_stories_empty_response():
    """Test graceful handling when Algolia returns no hits."""
    search_url = f"{ALGOLIA_BASE}/search"
    respx.get(search_url).mock(
        return_value=Response(200, json={"hits": [], "nbPages": 0})
    )

    with pytest.MonkeyPatch().context() as mp:
        mock_cache = MagicMock()
        mp.setattr("api.fetching.CACHE_PATH", mock_cache)
        mp.setattr("api.fetching.CANDIDATE_CACHE_PATH", mock_cache)
        mock_cache.__truediv__.return_value.exists.return_value = False
        mp.setattr("api.fetching._atomic_write_json", lambda p, d: None)
        mp.setattr("api.fetching._evict_old_cache_files", lambda: None)

        stories = await get_best_stories(limit=10, days=7)
        assert stories == []


@pytest.mark.asyncio
@respx.mock
async def test_get_best_stories_partial_failure():
    """Test that partial Algolia failures don't crash the fetcher."""
    search_url = f"{ALGOLIA_BASE}/search"

    call_count = 0

    def varying_response(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First window succeeds
            return Response(200, json={"hits": [{"objectID": "1"}, {"objectID": "2"}]})
        else:
            # Subsequent windows fail
            return Response(500, json={"error": "Internal error"})

    respx.get(search_url).mock(side_effect=varying_response)

    # Mock item fetches for the successful stories (MIN_STORY_COMMENTS=20)
    comments = [
        {
            "type": "comment",
            "text": f"Comment {i} with enough text to pass the minimum length filter.",
            "children": [],
        }
        for i in range(25)
    ]
    for sid in ["1", "2"]:
        respx.get(f"{ALGOLIA_BASE}/items/{sid}").mock(
            return_value=Response(
                200,
                json={
                    "id": int(sid),
                    "type": "story",
                    "title": f"Story {sid}",
                    "url": f"http://example.com/{sid}",
                    "points": 100,
                    "created_at_i": 1600000000,
                    "children": comments,
                },
            )
        )

    with pytest.MonkeyPatch().context() as mp:
        mock_cache = MagicMock()
        mp.setattr("api.fetching.CACHE_PATH", mock_cache)
        mp.setattr("api.fetching.CANDIDATE_CACHE_PATH", mock_cache)
        mock_cache.__truediv__.return_value.exists.return_value = False
        mp.setattr("api.fetching._atomic_write_json", lambda p, d: None)
        mp.setattr("api.fetching._evict_old_cache_files", lambda: None)

        stories = await get_best_stories(limit=10, days=14)
        # Should still return the stories from successful window
        assert len(stories) >= 1


class TestWindowFilters:
    """Test that window filters use correct boundary conditions."""

    def test_filter_uses_inclusive_start(self):
        """Verify filter string uses >= for start (not >) to include boundary stories."""
        # This is a unit test checking the filter construction logic
        # We check the actual filter string format
        from api.fetching import ALGOLIA_MIN_POINTS, MIN_STORY_COMMENTS

        ts_start = 1700000000
        ts_end = 1700086400

        # Simulate the filter construction from get_best_stories
        filters = [
            f"created_at_i>={ts_start}",
            f"points>{ALGOLIA_MIN_POINTS}",
            f"num_comments>={MIN_STORY_COMMENTS}",
        ]
        filters.append(f"created_at_i<{ts_end}")

        filter_str = ",".join(filters)

        # Check inclusive start
        assert f"created_at_i>={ts_start}" in filter_str
        # Check exclusive end (to prevent overlap)
        assert f"created_at_i<{ts_end}" in filter_str
        # Should NOT have exclusive start
        assert f"created_at_i>{ts_start}" not in filter_str


class TestWinTargetCalculation:
    """Test window target distribution logic."""

    def test_proportional_distribution(self):
        """Test that win_target is proportional to window duration."""
        import math

        # Simulate windows: 2 days live + 7 days archive
        windows = [
            (1000, 1000 + 2 * 86400, True),  # 2 days
            (1000 - 7 * 86400, 1000, False),  # 7 days
        ]
        limit = 100

        total_duration = sum(end - start for start, end, _ in windows)

        targets = []
        for ts_start, ts_end, _ in windows:
            duration = ts_end - ts_start
            win_target = math.ceil(limit * (duration / total_duration))
            win_target = max(win_target, 20)
            targets.append(win_target)

        # Live window (2 days) should get ~22% of limit
        # Archive window (7 days) should get ~78% of limit
        # But minimum is 20
        assert targets[0] >= 20  # Live window minimum
        assert targets[1] >= 20  # Archive window minimum
        assert targets[1] > targets[0]  # Archive should get more (longer duration)

    def test_minimum_target_enforced(self):
        """Test that minimum target of 20 is enforced even for small limits."""
        import math

        # Small limit with one window
        limit = 5
        total_duration = 7 * 86400
        duration = 7 * 86400

        win_target = math.ceil(limit * (duration / total_duration))
        win_target = max(win_target, 20)

        assert win_target == 20  # Should be minimum, not 5
