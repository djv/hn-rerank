from __future__ import annotations

import numpy as np


def unit_rows(rows: list[list[float]]) -> np.ndarray:
    """Return row-normalized float32 embeddings."""
    arr = np.array(rows, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
