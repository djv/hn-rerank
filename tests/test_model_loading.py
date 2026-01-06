from unittest.mock import MagicMock, patch
from api import rerank
import numpy as np


def test_model_singleton():
    """Verify init_model sets the global singleton."""
    # Reset singleton
    rerank._model = None

    with patch("api.rerank.ONNXEmbeddingModel") as MockClass:
        mock_instance = MagicMock()
        MockClass.return_value = mock_instance

        # 1. Init
        model1 = rerank.init_model("test-model")
        assert model1 == mock_instance
        MockClass.assert_called_with("test-model")

        # 2. Get (should return same)
        model2 = rerank.get_model()
        assert model2 == model1

        # 3. Init again
        model3 = rerank.init_model("other-model")
        assert model3 == model1
        # Constructor should NOT have been called again
        assert MockClass.call_count == 1


def test_get_embeddings_lazy_load():
    """Verify get_embeddings triggers load if model is None."""
    rerank._model = None

    with patch("api.rerank.ONNXEmbeddingModel") as MockClass:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[1, 2, 3]])
        MockClass.return_value = mock_instance

        # Prevent cache operations
        with patch("api.rerank.get_cache_key") as mock_key:
            mock_key.return_value.exists.return_value = False
            with patch("numpy.save"):
                rerank.get_embeddings(["text"])

        # Should have initialized the default model
        MockClass.assert_called_once()
