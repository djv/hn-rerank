from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from url_normalize import url_normalize


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        normalized = url_normalize(url)
    except Exception:
        normalized = url

    parts = urlsplit(normalized)
    if not parts.scheme or not parts.netloc:
        return normalized
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
