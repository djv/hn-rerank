import json
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".config" / "hn_rerank"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(key: str, value: str):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config[key] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_username() -> Optional[str]:
    return load_config().get("username")
