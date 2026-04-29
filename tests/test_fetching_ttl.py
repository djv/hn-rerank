import pytest
from unittest.mock import MagicMock, patch
from api import fetching
from api.constants import CANDIDATE_CACHE_TTL_LONG, CANDIDATE_CACHE_TTL_SHORT


@pytest.mark.asyncio
async def test_live_window_candidate_cache_uses_short_and_long_ttls():
    with (
        patch("api.fetching.get_cached_candidates") as mock_get_cache,
        patch("api.fetching.save_cached_candidates"),
        patch("api.fetching.httpx.AsyncClient") as mock_client_cls,
        patch("api.fetching.fetch_story"),
    ):
        mock_get_cache.return_value = None

        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client

        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"{}"
            resp.json.return_value = {"hits": []}
            return resp

        mock_client.get = mock_get
        mock_client_cls.return_value = mock_client

        await fetching.get_best_stories(limit=10, days=4, include_rss=False)

        ttls = [call.args[1] for call in mock_get_cache.call_args_list]
        assert CANDIDATE_CACHE_TTL_SHORT in ttls
        assert CANDIDATE_CACHE_TTL_LONG in ttls


@pytest.mark.asyncio
async def test_archive_window_uses_bigquery_instead_of_algolia_candidate_cache():
    with (
        patch("api.fetching.get_cached_candidates") as mock_get_cache,
        patch("api.fetching.save_cached_candidates"),
        patch("api.fetching.fetch_bigquery_archive_stories") as mock_bq_archive,
        patch("api.fetching.httpx.AsyncClient") as mock_client_cls,
    ):
        mock_get_cache.return_value = None
        mock_bq_archive.return_value = []

        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client

        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"{}"
            resp.json.return_value = {"hits": []}
            return resp

        mock_client.get = mock_get
        mock_client_cls.return_value = mock_client

        await fetching.get_best_stories(limit=10, days=40, include_rss=False)

        mock_bq_archive.assert_called_once()
        assert all(
            call.args[1] in {CANDIDATE_CACHE_TTL_SHORT, CANDIDATE_CACHE_TTL_LONG}
            for call in mock_get_cache.call_args_list
        )
