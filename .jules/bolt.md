## 2025-05-22 - [Vectorization of HN Score Calculation]
**Learning:** Python loops over simple mathematical operations are significantly slower than numpy vectorization. Replacing list comprehension with numpy arrays yielded ~2x speedup.
**Action:** Always look for opportunities to vectorize iterative mathematical calculations, especially in scoring/ranking logic. Be careful with `np.where` eager evaluation; use masked operations or safe math functions (e.g. `np.maximum`) to avoid runtime warnings.
