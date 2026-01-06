import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from api import rerank

# Use a temp cache dir for tests
TEST_CACHE_DIR = Path(".cache_test/embeddings")


@pytest.fixture
def temp_cache():
    # Setup
    rerank.CACHE_DIR = TEST_CACHE_DIR
    TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Teardown
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)


def test_cache_creation_and_hit(temp_cache):
    """
    Test that embedding is saved to disk and loaded on second call.
    """
    with patch("api.rerank.get_model") as mock_get_model:
        mock_model = MagicMock()
        # Mock encode to return a specific vector we can check for
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        # Mock string representation for cache key
        mock_model.__str__.return_value = "TestModel"
        mock_get_model.return_value = mock_model

        text = ["unique_text_for_cache"]

        # 1. First Call (Miss)
        vec1 = rerank.get_embeddings(text)
        assert mock_model.encode.call_count == 1
        assert np.allclose(vec1, [[0.1, 0.2, 0.3]])

        # Verify file created
        assert any(TEST_CACHE_DIR.iterdir())

        # 2. Second Call (Hit)
        vec2 = rerank.get_embeddings(text)
        # Should NOT call encode again
        assert mock_model.encode.call_count == 1
        assert np.allclose(vec2, [[0.1, 0.2, 0.3]])
