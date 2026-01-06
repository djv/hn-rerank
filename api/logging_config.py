import logging
import sys
from pathlib import Path

LOG_DIR = Path.home() / ".config" / "hn_rerank" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

def setup_logging(name: str = "hn_rerank", level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (only for API, TUI might want to suppress this or redirect to stderr)
    # We'll write to stderr so it doesn't break TUI stdout if piped
    console_handler = logging.StreamHandler(sys.stderr)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()
