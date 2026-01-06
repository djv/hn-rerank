from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_onnx_model():
    """
    Automatically mock ONNXEmbeddingModel for all tests to avoid
    loading the heavy ONNX model or requiring the onnx_model directory.
    """
    with patch("api.rerank.ONNXEmbeddingModel") as MockClass:
        mock_instance = MagicMock()
        MockClass.return_value = mock_instance

        # Default behavior: return random vectors of correct shape
        def side_effect(texts, **kwargs):
            return np.array(
                [
                    np.random.default_rng(len(t)).random(384)  # bge-small dim
                    for t in texts
                ]
            )

        mock_instance.encode.side_effect = side_effect
        yield mock_instance
