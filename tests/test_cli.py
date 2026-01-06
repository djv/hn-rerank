import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
import cli

# We don't need MOCK_HTML anymore because we mock HNClient directly


@pytest.mark.asyncio
async def test_cli_main_flow():
    """Test the full main loop with mocks using HNClient."""

    # Mock HNClient
    with patch("api.client.HNClient") as MockHNClient:
        mock_client = MockHNClient.return_value
        # Mock methods
        mock_client.check_session = AsyncMock(return_value=True)
        mock_client.fetch_favorites = AsyncMock(return_value={101})
        mock_client.fetch_upvoted = AsyncMock(return_value={102})
        mock_client.fetch_hidden = AsyncMock(return_value={103})
        mock_client.close = AsyncMock()

        # Mock fetch_story_texts helper in cli
        with patch("cli.fetch_story_texts", new_callable=AsyncMock) as mock_fetch_texts:
            mock_fetch_texts.return_value = ["Text 1", "Text 2"]

            with patch("cli.get_best_stories", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [
                    {"id": 1, "title": "Match", "text_content": "Match", "score": 100},
                    {"id": 2, "title": "Other", "text_content": "Other", "score": 50},
                ]

                # Mock get_embeddings to avoid ML model
                # Called for pos, neg, candidates (3 times)
                # Return dummy vectors
                with patch("api.rerank.get_embeddings", return_value=np.array([[1.0]])):
                    # Mock rank_stories
                    with patch(
                        "api.rerank.rank_stories",
                        return_value=[(0, 0.9, 0), (1, 0.1, 0)],
                    ):
                        # Mock rich console
                        with patch("cli.console.print"):
                            # Mock Arguments
                            args = MagicMock()
                            args.username = "testuser"
                            args.limit = 100
                            args.top = 10
                            args.days = 30
                            args.diversity = 0.0
                            args.clusters = 0
                            args.model = "test-model"
                            args.login = False
                            args.tui = False

                            await cli.main(args)

                            # Verifications
                            mock_client.fetch_favorites.assert_called_with("testuser")
                            mock_client.fetch_upvoted.assert_called_with("testuser")
                            mock_fetch_texts.assert_called()  # Should be called for pos and neg
