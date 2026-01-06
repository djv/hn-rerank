import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import MagicMock, AsyncMock, patch
from tui_app import HNRerankTUI, StoryItem
import numpy as np
import asyncio

# Mock Data Generator
def create_mock_story(sid, score=100):
    return {
        "id": sid,
        "title": f"Story {sid}",
        "score": score,
        "url": f"http://example.com/{sid}",
        "text_content": "Content",
        "comments": ["Comment"]
    }

@pytest.fixture
def tui_mocks():
    # We use a more sophisticated ranker mock to test sorting invariants
    def mock_ranker(candidates, **kwargs):
        # Return indices in order of provided candidates, with decreasing scores
        return [(i, 1.0 - (i * 0.01), 0) for i in range(len(candidates))]

    with patch("api.rerank.init_model"), \
         patch("api.rerank.get_embeddings", return_value=np.zeros((1, 768))), \
         patch("api.rerank.rank_stories", side_effect=mock_ranker), \
         patch("tui_app.get_user_data", new_callable=AsyncMock) as m_user, \
         patch("tui_app.get_best_stories", new_callable=AsyncMock) as m_cand, \
         patch("tui_app.HNClient") as MockClient:

        m_user.return_value = ([create_mock_story(999)], [], {999})
        m_cand.return_value = [create_mock_story(i) for i in range(20)]
        
        client = MockClient.return_value
        client.vote = AsyncMock(return_value=(True, "OK"))
        client.hide = AsyncMock(return_value=(True, "OK"))
        client.close = AsyncMock()
        client.login = AsyncMock(return_value=(True, "OK"))
        
        yield {
            "user": m_user,
            "cand": m_cand,
            "client": client
        }

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None, max_examples=5)
@given(st.just(None))
@pytest.mark.asyncio
async def test_law_of_ranking_integrity(tui_mocks, _):
    """
    Invariant: Stories must always be displayed in descending order of their computed score.
    """
    app = HNRerankTUI("testuser")
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        list_view = app.query_one("#story-list")
        
        scores = [child.score_val for child in list_view.children if isinstance(child, StoryItem)]
        assert scores == sorted(scores, reverse=True)

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much], deadline=None, max_examples=5)
@given(indices=st.just([0, 1]))
@pytest.mark.asyncio
async def test_law_of_expansion_isolation(tui_mocks, indices):
    """
    Invariant: Expanding one story must not affect the state of others.
    """
    app = HNRerankTUI("testuser")
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        list_view = app.query_one("#story-list")
        
        # Expand target indices
        for idx in indices:
            list_view.index = idx
            await pilot.press("enter")
            await pilot.pause(0.1) # Wait for animation/reactivity
            
        # Verify only chosen indices are expanded
        for i, child in enumerate(list_view.children):
            if i in indices:
                assert child.expanded is True, f"Item {i} should be expanded"
            else:
                assert child.expanded is False, f"Item {i} should not be expanded"

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None, max_examples=5)
@given(keys=st.lists(st.sampled_from(["down", "up", "enter", "u", "d"]), min_size=5, max_size=20))
@pytest.mark.asyncio
async def test_system_thermal_stability(tui_mocks, keys):
    """
    Invariant: Randomized high-entropy input must never crash the UI or leave it in a non-responsive state.
    """
    app = HNRerankTUI("testuser")
    with patch("webbrowser.open_new_tab"):
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            for key in keys:
                await pilot.press(key)
                
            assert app.is_running
            assert app.query_one("#story-list").index is not None
