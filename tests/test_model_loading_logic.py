
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from api.rerank import ONNXEmbeddingModel

def test_encode_logic():
    # Mock Tokenizer and Session
    with patch("api.rerank.AutoTokenizer") as MockTokenizer, \
         patch("api.rerank.ort.InferenceSession") as MockSession:

        # Setup Tokenizer mock
        tokenizer = MockTokenizer.from_pretrained.return_value
        tokenizer.model_max_length = 512
        # mocked output of tokenizer call
        # inputs: 'input_ids', 'attention_mask', etc.
        # We need to simulate return_tensors='np'
        tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 0]]), # 3rd token masked
        }

        # Setup Session mock
        session = MockSession.return_value
        session.get_inputs.return_value = [MagicMock(name="input_ids"), MagicMock(name="attention_mask")]
        # run returns list of outputs. First is last_hidden_state.
        # Shape: (batch_size, seq_len, hidden_dim)
        # batch=1, seq=3, dim=4
        last_hidden_state = np.array([
            [
                [1.0, 1.0, 1.0, 1.0], # Token 1 (Mask 1)
                [2.0, 2.0, 2.0, 2.0], # Token 2 (Mask 1)
                [9.0, 9.0, 9.0, 9.0], # Token 3 (Mask 0) - should be ignored
            ]
        ])
        session.run.return_value = [last_hidden_state]

        # Initialize model
        model = ONNXEmbeddingModel("dummy_path")

        # Run encode
        texts = ["test sentence"]
        embeddings = model.encode(texts, normalize_embeddings=False)

        # Check logic
        # Sum of valid tokens:
        # Token 1: [1,1,1,1]
        # Token 2: [2,2,2,2]
        # Token 3: Ignored
        # Sum = [3,3,3,3]
        # Sum Mask = 2 (two tokens)
        # Mean = [1.5, 1.5, 1.5, 1.5]

        expected = np.array([[1.5, 1.5, 1.5, 1.5]])
        assert np.allclose(embeddings, expected)

        # Test Normalization
        embeddings_norm = model.encode(texts, normalize_embeddings=True)
        norm = np.linalg.norm(expected)
        expected_norm = expected / norm
        assert np.allclose(embeddings_norm, expected_norm)

def test_encode_empty():
    with patch("api.rerank.AutoTokenizer"), \
         patch("api.rerank.ort.InferenceSession"):
        model = ONNXEmbeddingModel("dummy")
        assert len(model.encode([])) == 0
