import pytest
from unittest.mock import MagicMock, patch
from api import fetching
from api.constants import CANDIDATE_CACHE_TTL_LONG, CANDIDATE_CACHE_TTL_SHORT
from api.config import AppConfig, ArchiveConfig


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

        config = AppConfig(days=4, no_rss=True)
        await fetching.get_best_stories(limit=10, config=config)

        ttls = [call.args[1] for call in mock_get_cache.call_args_list]
        assert CANDIDATE_CACHE_TTL_SHORT in ttls
        assert CANDIDATE_CACHE_TTL_LONG in ttls


@pytest.mark.asyncio
async def test_archive_window_skips_open_index_by_default():
    with (
        patch("api.fetching.get_cached_candidates") as mock_get_cache,
        patch("api.fetching.save_cached_candidates"),
        patch("api.fetching.fetch_open_index_archive_stories") as mock_archive,
        patch("api.fetching.load_cached_archive_stories") as mock_cached_archive,
        patch("api.fetching.httpx.AsyncClient") as mock_client_cls,
    ):
        mock_get_cache.return_value = None
        mock_archive.return_value = []
        mock_cached_archive.return_value = []

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

        config = AppConfig(days=40, no_rss=True)
        await fetching.get_best_stories(limit=10, config=config)

        mock_archive.assert_not_called()
        mock_cached_archive.assert_called_once()
        assert all(
            call.args[1] in {CANDIDATE_CACHE_TTL_SHORT, CANDIDATE_CACHE_TTL_LONG}
            for call in mock_get_cache.call_args_list
        )


@pytest.mark.asyncio
async def test_archive_window_uses_open_index_when_enabled():
    with (
        patch("api.fetching.get_cached_candidates") as mock_get_cache,
        patch("api.fetching.save_cached_candidates"),
        patch("api.fetching.fetch_open_index_archive_stories") as mock_archive,
        patch("api.fetching.load_cached_archive_stories") as mock_cached_archive,
        patch("api.fetching.httpx.AsyncClient") as mock_client_cls,
    ):
        mock_get_cache.return_value = None
        mock_archive.return_value = []
        mock_cached_archive.return_value = []

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

        config = AppConfig(
            days=40,
            no_rss=True,
            archive=ArchiveConfig(open_index_enabled=True),
        )
        await fetching.get_best_stories(limit=10, config=config)

        mock_archive.assert_called_once()
        assert all(
            call.args[1] in {CANDIDATE_CACHE_TTL_SHORT, CANDIDATE_CACHE_TTL_LONG}
            for call in mock_get_cache.call_args_list
        )
