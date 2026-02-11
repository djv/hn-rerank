from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically using a temp file + rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def evict_old_cache_files(
    cache_dir: Path, pattern: str, max_files: int
) -> None:
    """Remove oldest cache files if over max_files limit (LRU by mtime)."""
    if max_files <= 0:
        return
    cache_files = list(cache_dir.glob(pattern))
    if len(cache_files) <= max_files:
        return
    cache_files.sort(key=lambda p: p.stat().st_mtime)
    for f in cache_files[: len(cache_files) - max_files]:
        with suppress(OSError):
            f.unlink()
