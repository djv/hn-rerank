import asyncio
import logging
from unittest.mock import patch, MagicMock
from api.fetching import get_best_stories, CANDIDATE_CACHE_PATH

# Configure logging to see the "Cache miss" messages
logging.basicConfig(level=logging.INFO)


async def repro():
    # Clear cache
    if CANDIDATE_CACHE_PATH.exists():
        import shutil

        shutil.rmtree(CANDIDATE_CACHE_PATH)
    CANDIDATE_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    print("--- First Call (Should Miss) ---")

    # Mock return data for search
    search_hits = {"hits": [{"objectID": i} for i in range(100, 110)]}

    # Mock dependencies
    with patch("httpx.AsyncClient.get") as mock_get:
        # Search returns hits, Item fetch returns story
        mock_get.side_effect = [
            # Search call
            MagicMock(status_code=200, json=lambda: search_hits, content=b"{}"),
            # Story calls (will be called for each hit)
            *[
                MagicMock(
                    status_code=200,
                    json=lambda: {
                        "type": "story",
                        "id": 100 + i,
                        "title": "t",
                        "points": 10,
                    },
                    content=b"{}",
                )
                for i in range(10)
            ],
        ]

        # Call 1
        await get_best_stories(limit=5, days=1)

        print(f"API Calls after 1st run: {mock_get.call_count}")

    print("\n--- Second Call (Should Hit) ---")

    with patch("httpx.AsyncClient.get") as mock_get:
        # Search returns hits (if called), Item fetch returns story
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: search_hits, content=b"{}"),
            *[
                MagicMock(
                    status_code=200,
                    json=lambda: {
                        "type": "story",
                        "id": 100 + i,
                        "title": "t",
                        "points": 10,
                    },
                    content=b"{}",
                )
                for i in range(10)
            ],
        ]

        # Call 2 (Immediate)
        await get_best_stories(limit=5, days=1)

        print(f"API Calls in 2nd run: {mock_get.call_count}")

        # We expect 0 search calls if cache hit.
        # But we might see story fetches if story cache is separate (it is).
        # We want to check if *search* was called.

        # Check arguments of calls
        search_calls = [c for c in mock_get.call_args_list if "search" in str(c)]
        print(f"Search API calls in 2nd run: {len(search_calls)}")

        if len(search_calls) > 0:
            print("FAILURE: Cache Miss (Search API called again)")
        else:
            print("SUCCESS: Cache Hit (No search API calls)")


if __name__ == "__main__":
    asyncio.run(repro())
