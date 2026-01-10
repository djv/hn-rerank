import pytest
from unittest.mock import AsyncMock, patch
from generate_html import main
import sys


@pytest.mark.asyncio
async def test_generate_html_integration(tmp_path):
    """
    Integration test for generate_html.py.
    Mocks the API and ranking layers to verify the end-to-end HTML generation.
    """
    username = "testuser"
    output_file = tmp_path / "dashboard.html"

    # Mock stories
    mock_story = {
        "id": 1,
        "title": "Integrated Test",
        "url": "https://integrated.test",
        "score": 100,
        "time": 1600000000,
        "comments": ["Integration works"],
        "text_content": "integrated test content",
    }

    # Setup mocks
    with (
        patch("generate_html.HNClient") as mock_client_class,
        patch("generate_html.get_best_stories", new_callable=AsyncMock) as mock_best,
        patch("generate_html.fetch_story", new_callable=AsyncMock) as mock_fetch,
        patch("generate_html.rerank.get_embeddings") as mock_emb,
        patch("generate_html.rerank.compute_recency_weights") as mock_weights,
        patch("generate_html.rerank.rank_stories") as mock_rank,
        patch("generate_html.rerank.generate_story_tldr") as mock_tldr,
        patch("generate_html.rerank.generate_similarity_reason") as mock_reason,
    ):
        # Mock LLM functions
        mock_tldr.return_value = "This is a test TL;DR summary."
        mock_reason.return_value = ""
        # Mock API behavior
        client_instance = mock_client_class.return_value.__aenter__.return_value
        # Mock login check - simulate logged in state
        mock_response = AsyncMock()
        mock_response.text = "logout"  # Contains "logout" = logged in
        client_instance.client.get = AsyncMock(return_value=mock_response)
        client_instance.fetch_user_data.return_value = {
            "pos": set(),
            "upvoted": {1},  # Now using upvoted instead of pos
            "hidden": set(),
        }

        mock_fetch.return_value = mock_story
        mock_best.return_value = [mock_story]

        # Mock Ranking
        import numpy as np

        mock_emb.return_value = np.zeros((1, 768))
        mock_weights.return_value = np.ones(1)
        mock_rank.return_value = [(0, 0.95, 0, 0.95)]  # (idx, score, fav_idx, max_sim)

        # Mock sys.argv
        with patch.object(
            sys, "argv", ["generate_html.py", username, "-o", str(output_file)]
        ):
            await main()

        # Verify output
        assert output_file.exists()
        html_content = output_file.read_text()
        assert "Integrated Test" in html_content
        assert "95%" in html_content
        assert username in html_content
        # TL;DR replaces raw comments
        assert "This is a test TL;DR summary." in html_content
