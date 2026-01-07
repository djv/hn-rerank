# Technical Debt & Audit Report

## Security
- **Cookie Storage**: Session cookies are stored in plain text in `.cache/user/cookies.json`.
    - *Risk*: Low for local usage, but critical if the directory is shared or committed.
    - *Mitigation*: Ensure `.gitignore` covers `.cache/` (Verified: it does).
- **Password Handling**: `getpass` is used, which is good. Password is not stored, only session cookies.

## Performance
- **Embedding**: We compute embeddings for *all* candidates every run if not cached.
    - *Optimization*: The ONNX runtime is fast, but for 1000+ candidates, a vector database (like Chroma or FAISS) would be more scalable than `numpy` arrays.
- **Scraping**: `BeautifulSoup` is synchronous CPU-bound work running in an async loop.
    - *Optimization*: For massive scale, this should run in a `ProcessPoolExecutor`, but for <1000 items, the overhead is negligible compared to network I/O.

## Code Quality
- **Type Safety**: The codebase uses `typing` extensively (`mypy`/`pyright` compliant).
- **Testing**: `pytest` coverage exists for core logic.
- **Linting**: `ruff` checks pass.

## Future Refactoring
- **Signal Support**: Currently, favorited *comments* are treated as invalid stories and negatively cached.
    - *Improvement*: Implement a `fetch_comment` logic to support comment-based signals.
- **Config**: Constants are hardcoded in `api/constants.py`.
    - *Improvement*: Move to `config.toml` or environment variables for easier user tweaking without code edits.
