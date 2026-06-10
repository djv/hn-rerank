"""Aggregate impression/click stats from telemetry SQLite for use as ranking features."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(".cache/user_feedback/feedback_events.sqlite")


@dataclass(frozen=True)
class StoryImpressionStats:
    impression_count: int = 0
    click_count: int = 0
    click_ratio: float = 0.0
    days_since_last_impression: float = 30.0


@dataclass(frozen=True)
class DomainImpressionStats:
    domain_ctr: float = 0.0
    domain_impression_count: int = 0


_story_stats: dict[int, StoryImpressionStats] | None = None
_domain_stats: dict[str, DomainImpressionStats] | None = None


def extract_domain_with_fallback(url: str | None, is_hn: bool = False) -> str | None:
    """Pull domain from a URL string, with fallback for text-only HN posts.

    Accepts None for url (which occurs when SQLite url IS NULL), cleanly
    falling back to "hn.text" if is_hn is True.
    """
    if not url:
        return "hn.text" if is_hn else None
    from api.url_utils import extract_domain
    domain = extract_domain(url)
    if not domain:
        return "hn.text" if is_hn else None
    return domain


def fetch_impression_stats() -> tuple[dict[int, StoryImpressionStats], dict[str, DomainImpressionStats]]:
    """Query telemetry_events and return per-story + per-domain stats.

    Returns ({story_id: stats}, {domain: stats}). Missing keys = all-zeros.
    """
    if not DB_PATH.exists():
        return {}, {}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # Per-story aggregation
        story_rows = conn.execute(
            """
            SELECT story_id,
                   SUM(CASE WHEN event='impression' THEN 1 ELSE 0 END) as impressions,
                   SUM(CASE WHEN event='click' THEN 1 ELSE 0 END) as clicks,
                   MAX(server_ts) as last_seen
            FROM telemetry_events
            GROUP BY story_id
            """
        ).fetchall()

        now = time.time()
        per_story: dict[int, StoryImpressionStats] = {}
        for row in story_rows:
            imps = int(row["impressions"] or 0)
            clks = int(row["clicks"] or 0)
            last = float(row["last_seen"] or 0.0)
            days_since = min((now - last) / 86400.0, 30.0) if last > 0 else 30.0
            per_story[row["story_id"]] = StoryImpressionStats(
                impression_count=imps,
                click_count=clks,
                click_ratio=clks / max(imps, 1),
                days_since_last_impression=days_since,
            )

        # Domain-level aggregation: group by URL first, then merge domains in Python
        domain_rows = conn.execute(
            """
            SELECT url,
                   story_source,
                   SUM(CASE WHEN event='click' THEN 1 ELSE 0 END) as clicks,
                   SUM(CASE WHEN event='impression' THEN 1 ELSE 0 END) as impressions
            FROM telemetry_events
            GROUP BY url, story_source
            """
        ).fetchall()

        domain_clicks: dict[str, int] = {}
        domain_impressions: dict[str, int] = {}
        for row in domain_rows:
            url = row["url"]
            is_hn = row["story_source"] == "hn"
            domain = extract_domain_with_fallback(url, is_hn=is_hn)
            if domain:
                domain_clicks[domain] = domain_clicks.get(domain, 0) + int(row["clicks"] or 0)
                domain_impressions[domain] = domain_impressions.get(domain, 0) + int(row["impressions"] or 0)

        per_domain: dict[str, DomainImpressionStats] = {}
        for domain in domain_impressions:
            imps = domain_impressions[domain]
            clks = domain_clicks.get(domain, 0)
            per_domain[domain] = DomainImpressionStats(
                domain_ctr=clks / max(imps, 1),
                domain_impression_count=imps,
            )

        return per_story, per_domain
    finally:
        conn.close()


def load_telemetry_stats() -> tuple[dict[int, StoryImpressionStats], dict[str, DomainImpressionStats]]:
    """Lazy thread-safe cache load of telemetry stats for the current run.
    
    This is called independently by each metadata feature function (e.g., 6 times
    per rank_stories invocation). The cache ensures we only query SQLite once
    and makes the independent calls extremely fast dict-lookups.
    """
    global _story_stats, _domain_stats
    if _story_stats is not None and _domain_stats is not None:
        return _story_stats, _domain_stats
    _story_stats, _domain_stats = fetch_impression_stats()
    return _story_stats, _domain_stats


def reset_telemetry_stats_cache() -> None:
    """Clear cached stats (primarily for testing purposes)."""
    global _story_stats, _domain_stats
    _story_stats = None
    _domain_stats = None
