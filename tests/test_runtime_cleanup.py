import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import prepare_data
from api.rerank import _get_embeddings_with_model
from evaluate_quality import RankingEvaluator


@pytest.mark.asyncio
async def test_prepare_data_closes_hn_client_on_early_return():
    with patch("prepare_data.HNClient") as mock_client_class:
        client_cm = mock_client_class.return_value
        client = MagicMock()
        client.fetch_user_data = AsyncMock(
            return_value={"pos": {1}, "upvoted": set(), "hidden": set()}
        )
        client_cm.__aenter__ = AsyncMock(return_value=client)
        client_cm.__aexit__ = AsyncMock(return_value=None)

        await prepare_data.prepare_data("testuser", limit=1)

        client_cm.__aenter__.assert_awaited_once()
        client_cm.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_ranking_evaluator_load_data_closes_hn_client_on_early_return():
    evaluator = RankingEvaluator("testuser")

    with patch("evaluate_quality.HNClient") as mock_client_class:
        client_cm = mock_client_class.return_value
        client = MagicMock()
        client.fetch_user_data = AsyncMock(
            return_value={"pos": {1}, "upvoted": set(), "hidden": set()}
        )
        client_cm.__aenter__ = AsyncMock(return_value=client)
        client_cm.__aexit__ = AsyncMock(return_value=None)

        loaded = await evaluator.load_data()

        assert loaded is False
        client_cm.__aenter__.assert_awaited_once()
        client_cm.__aexit__.assert_awaited_once()


def test_get_embeddings_with_model_closes_npz_cache_file(tmp_path, monkeypatch):
    class _StubModel:
        model_id = "cache-hit-test"

        @staticmethod
        def truncate_to_token_budget(text: str, _max_tokens: int) -> tuple[str, bool]:
            return text, False

    text = "story 1"
    cache_version = "v-test"
    digest = hashlib.sha256(
        f"{cache_version}:{_StubModel.model_id}:{text}".encode()
    ).hexdigest()
    cache_path = tmp_path / f"{digest}.npz"
    cache_path.write_bytes(b"placeholder")

    embedding = np.ones((768,), dtype=np.float32)
    state = {"closed": False}

    class _FakeNpzFile:
        def __enter__(self):
            return {"embedding": embedding}

        def __exit__(self, exc_type, exc, tb):
            state["closed"] = True
            return False

    monkeypatch.setattr("api.rerank.np.load", lambda _path: _FakeNpzFile())

    result = _get_embeddings_with_model(
        [text],
        model=_StubModel(),  # type: ignore[arg-type]
        cache_dir=tmp_path,
        cache_version=cache_version,
        is_query=False,
    )

    assert result.shape == (1, 768)
    assert state["closed"] is True
