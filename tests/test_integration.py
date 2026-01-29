import pytest
from unittest.mock import AsyncMock, patch
from generate_html import main
from api.models import Story, RankResult
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
    mock_story = Story(
        id=1,
        title="Integrated Test",
        url="https://integrated.test",
        score=100,
        time=1600000000,
        comments=["Integration works"],
        text_content="integrated test content",
    )

    # Setup mocks
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "fake_key"}),
        patch("generate_html.HNClient") as mock_client_class,
        patch("generate_html.get_best_stories", new_callable=AsyncMock) as mock_best,
        patch("generate_html.fetch_story", new_callable=AsyncMock) as mock_fetch,
        patch("generate_html.rerank.get_embeddings") as mock_emb,
        patch("generate_html.rerank.rank_stories") as mock_rank,
        patch(
            "generate_html.rerank.generate_batch_tldrs", new_callable=AsyncMock
        ) as mock_batch_tldrs,
        patch(
            "generate_html.rerank.generate_batch_cluster_names", new_callable=AsyncMock
        ) as mock_batch_names,
    ):
        # Mock LLM functions
        mock_batch_tldrs.return_value = {1: "This is a test TL;DR summary."}
        mock_batch_names.return_value = {0: "Technology"}

        # Mock API behavior
        client_instance = mock_client_class.return_value.__aenter__.return_value
        # Mock login check - simulate logged in state
        mock_response = AsyncMock()
        mock_response.text = "logout"  # Contains "logout" = logged in
        client_instance.client.get = AsyncMock(return_value=mock_response)
        client_instance.fetch_user_data.return_value = {
            "pos": set(),
            "upvoted": {1},
            "hidden": set(),
            "hidden_urls": set(),
            "favorites": set(),
        }

        mock_fetch.return_value = mock_story
        mock_best.return_value = [mock_story]

        # Mock Ranking
        import numpy as np

        mock_emb.return_value = np.zeros((1, 768))
        mock_rank.return_value = [
            RankResult(index=0, hybrid_score=0.95, best_fav_index=0, max_sim_score=0.95, knn_score=0.90)
        ]

        # Mock sys.argv
        with patch.object(
            sys, "argv", ["generate_html.py", username, "-o", str(output_file)]
        ):
            await main()

        # Verify output
        assert output_file.exists()
        html_content = output_file.read_text()
        assert "Integrated Test" in html_content
        assert "90%" in html_content  # Uses knn_score for display
        assert username in html_content
        # TL;DR replaces raw comments
        assert "This is a test TL;DR summary." in html_content
