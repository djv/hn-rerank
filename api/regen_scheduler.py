"""Fire-and-forget dashboard regeneration on feedback.

When request_regen() is called and no regen is currently running,
spawn the dashboard build process.  If a regen is already running,
skip — the running process reflects the current feedback state.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
LOG_FILE = Path(ROOT_DIR) / ".cache" / "regen.log"

# Use the venv Python directly (no uv wrapper needed — we're already in it).
COMMAND = [sys.executable, "generate_html.py"]

_regen_process: subprocess.Popen[bytes] | None = None
_regen_lock = threading.Lock()
logger = logging.getLogger("regen_scheduler")


def request_regen() -> None:
    global _regen_process

    with _regen_lock:
        if _regen_process is not None:
            ret = _regen_process.poll()
            if ret is None:
                return
            _regen_process = None

    logger.info("Starting dashboard regeneration")
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "ab") as log_fh:
            log_fh.write(
                f"\n--- regen {datetime.now():%Y-%m-%d %H:%M:%S} ---\n".encode()
            )
            log_fh.flush()
            _regen_process = subprocess.Popen(
                COMMAND,
                cwd=ROOT_DIR,
                stdout=log_fh,
                stderr=log_fh,
            )
    except Exception:
        logger.exception("Failed to start dashboard regeneration")
        try:
            with open(LOG_FILE, "a") as log_fh:
                log_fh.write(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] FATAL: "
                    f"{traceback.format_exc()}\n"
                )
        except Exception:
            pass
