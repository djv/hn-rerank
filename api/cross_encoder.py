from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase

class CrossEncoderModel:
    def __init__(self, model_dir: str = "onnx_ce_model") -> None:
        self.model_dir: str = model_dir
        model_path = Path(model_dir) / "model.onnx"
        if not model_path.exists():
            # Try to find it relative to current file if it is in a worktree
            alt_path = Path(__file__).parent.parent / model_dir / "model.onnx"
            if alt_path.exists():
                 model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Cross-encoder model not found in {model_dir} (checked {model_path} and {alt_path})."
                )

        from transformers import AutoTokenizer

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(model_path.parent)
        )

        providers = ["CPUExecutionProvider"]
        self.session: ort.InferenceSession = ort.InferenceSession(
            str(model_path), providers=providers
        )
        
        self._tokenizer_lock = threading.Lock()
        self._input_names: list[str] = [node.name for node in self.session.get_inputs()]

    def score(
        self,
        queries: list[str],
        documents: list[str],
        batch_size: int = 16,
    ) -> NDArray[np.float32]:
        """Compute cross-encoder scores for query-document pairs."""
        if len(queries) != len(documents):
            raise ValueError("Number of queries and documents must match.")
        
        all_scores: list[NDArray[np.float32]] = []
        total_items: int = len(queries)

        for i in range(0, total_items, batch_size):
            batch_queries = queries[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            
            # Cross-encoders take pairs of sentences
            with self._tokenizer_lock:
                inputs: BatchEncoding = self.tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )
                
                ort_inputs: dict[str, NDArray[np.int64]] = {
                    k: v.astype(np.int64)
                    for k, v in inputs.items()
                    if k in self._input_names
                }

            outputs = self.session.run(None, ort_inputs)
            # Cross-encoder output is usually (batch_size, 1) or (batch_size, 2)
            # For ms-marco-MiniLM-L-6-v2, it is (batch_size, 1)
            logits: NDArray[np.float32] = cast(NDArray[np.float32], outputs[0])
            all_scores.append(logits.flatten())

        if not all_scores:
            return np.array([], dtype=np.float32)
        return np.concatenate(all_scores)
