import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from api.rerank import ONNXEmbeddingModel, _get_embeddings_with_model


class _BorrowCheckingTokenizer:
    """Tokenizer stub that raises if called concurrently."""

    def __init__(self) -> None:
        self._active = 0
        self._guard = threading.Lock()

    def __call__(
        self,
        texts,
        *,
        truncation: bool,
        max_length: int,
        return_tensors,
        padding: bool | None = None,
    ):
        items = [texts] if isinstance(texts, str) else list(texts)
        _ = truncation, max_length, padding

        with self._guard:
            self._active += 1
            if self._active > 1:
                self._active -= 1
                raise RuntimeError("Already borrowed")

        try:
            # Force overlap windows under threads.
            time.sleep(0.005)
            if return_tensors == "np":
                n = len(items)
                seq = 4
                return {
                    "input_ids": np.ones((n, seq), dtype=np.int64),
                    "attention_mask": np.ones((n, seq), dtype=np.int64),
                }
            return {"input_ids": [1, 2, 3]}
        finally:
            with self._guard:
                self._active -= 1

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        _ = token_ids, skip_special_tokens
        return "decoded"


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummySession:
    def get_inputs(self):
        return [_Node("input_ids"), _Node("attention_mask")]

    def run(self, _, ort_inputs):
        n = ort_inputs["input_ids"].shape[0]
        seq = ort_inputs["input_ids"].shape[1]
        hidden = 8
        return [np.ones((n, seq, hidden), dtype=np.float32)]


def _make_stub_model() -> ONNXEmbeddingModel:
    model = ONNXEmbeddingModel.__new__(ONNXEmbeddingModel)
    model.model_id = "thread-test"
    model.tokenizer = _BorrowCheckingTokenizer()
    model.session = _DummySession()
    model._tokenizer_lock = threading.Lock()
    model._input_names = ["input_ids", "attention_mask"]
    return model


def test_get_embeddings_thread_safe_tokenizer_calls(tmp_path):
    model = _make_stub_model()

    def worker(idx: int):
        texts = [f"story {idx}-{j}" for j in range(3)]
        return _get_embeddings_with_model(
            texts,
            model=model,
            cache_dir=tmp_path,
            cache_version="v-thread-test",
            is_query=False,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        results = [f.result() for f in futures]

    assert len(results) == 8
    for arr in results:
        assert arr.shape == (3, 8)
