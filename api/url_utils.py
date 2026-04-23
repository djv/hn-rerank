from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from url_normalize import url_normalize

_TRACKING_QUERY_PARAMS = {
    "utm",
    "fbclid",
    "gclid",
    "dclid",
    "gbraid",
    "wbraid",
    "mc_cid",
    "mc_eid",
    "_hsenc",
    "_hsmi",
    "mkt_tok",
}


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        normalized_raw = url_normalize(url)
        normalized = normalized_raw if isinstance(normalized_raw, str) else url
    except Exception:
        normalized = url

    parts = urlsplit(normalized)
    if not parts.scheme or not parts.netloc:
        return normalized
    path = parts.path.rstrip("/")
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_QUERY_PARAMS
    ]
    filtered_query.sort(key=lambda item: item)
    query = urlencode(filtered_query, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, path, query, ""))
