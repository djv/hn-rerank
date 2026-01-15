import asyncio
from unittest.mock import patch, MagicMock
from api.fetching import get_best_stories, CANDIDATE_CACHE_PATH

# Constants
MOCK_NOW_TS = 1700000000
DAYS_AGO_30 = MOCK_NOW_TS - (30 * 86400)


async def test_archive_cache_invalidation():
    # Setup
    if CANDIDATE_CACHE_PATH.exists():
        import shutil

        shutil.rmtree(CANDIDATE_CACHE_PATH)
    CANDIDATE_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Testing archive cache invalidation in {CANDIDATE_CACHE_PATH}")

    # We simulate a run that fetches data from 1 month ago.
    # We fix "Now" to a Monday to align anchor exactly.
    # MOCK_NOW_TS is arbitrary. Let's make it a Monday.
    # 2026-01-12 is a Monday. TS: 1768262400 (approx)
    # Let's just mock .weekday() to return 0.

    # Run 1: Fetch
    print("--- Run 1 (Day 0) ---")
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=b"{}",
            json=lambda: {"hits": [{"objectID": i} for i in range(10)]},
        )

        with patch(
            "api.fetching.fetch_story",
            side_effect=lambda client, sid: {"id": sid, "title": "test", "score": 10},
        ):
            with patch("api.fetching.datetime") as mock_datetime:
                mock_datetime.now.return_value.timestamp.return_value = MOCK_NOW_TS
                mock_datetime.now.return_value.weekday.return_value = 0  # Monday

                # We expect windows to cover 30 days.
                # Anchor = Now.
                # Archive 1: -7d to Now.
                # ...
                # Archive 4: -28d to -21d.
                await get_best_stories(limit=10, days=35)

    # Verify cache files exist
    files = list(CANDIDATE_CACHE_PATH.glob("*.json"))
    print(f"Cache files created: {len(files)}")
    if len(files) == 0:
        print("FAILURE: No cache files.")
        return

    # Run 2: Advance time by 8 days ( > 7 day TTL)
    # The windows will shift because Anchor shifts by 7 days.
    # New Anchor = Old Anchor + 7d.
    # Old Archive 1 (T-7 to T) becomes New Archive 2 (T-7 to T) relative to new anchor?
    # New Anchor is T+7.
    # New Archive 1: T to T+7.
    # New Archive 2: T-7 to T.
    # So the time range "T-7 to T" IS requested again.
    # The cache key for "T-7 to T" should exist.
    # BUT, the file mtime is T.
    # Current time is T+8d.
    # Age = 8d. TTL = 7d.
    # Result: Expired. Refetch.

    print("\n--- Run 2 (Day 8 - Expect Refetch of old windows) ---")
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=b"{}",
            json=lambda: {"hits": [{"objectID": i} for i in range(10)]},
        )

        with patch(
            "api.fetching.fetch_story",
            side_effect=lambda client, sid: {"id": sid, "title": "test", "score": 10},
        ):
            with patch("api.fetching.datetime") as mock_datetime:
                mock_datetime.now.return_value.timestamp.return_value = MOCK_NOW_TS + (
                    8 * 86400
                )
                mock_datetime.now.return_value.weekday.return_value = 0  # Still Monday

                # Patch time.time to simulate aging check
                with patch("time.time", return_value=MOCK_NOW_TS + (8 * 86400)):
                    await get_best_stories(limit=10, days=35)

                # Count search calls.
                # We expect multiple windows.
                # If they expired, we see calls.
                search_calls = [
                    c for c in mock_get.call_args_list if "search" in str(c)
                ]
                print(f"Search API calls in Run 2: {len(search_calls)}")

                if len(search_calls) > 0:
                    print("CONFIRMED: Archive windows refetched after 8 days.")
                else:
                    print("UNEXPECTED: Archive windows NOT refetched.")

    # Cleanup
    import shutil

    shutil.rmtree(CANDIDATE_CACHE_PATH)


if __name__ == "__main__":
    asyncio.run(test_archive_cache_invalidation())
