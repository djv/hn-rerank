"""SQLite-backed event logging for dashboard card views.

Handles impression and click events in the telemetry_events table.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DB_PATH = Path(".cache/user_feedback/feedback_events.sqlite")
KNOWN_ACQUISITION_KINDS = frozenset({"exploit", "uncertainty", "disagreement", "novel"})
KNOWN_EVENTS = frozenset({"impression", "click"})


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_event_schema(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    if conn is None:
        conn = connect_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            event TEXT NOT NULL CHECK (event IN ('impression', 'click')),
            server_ts REAL NOT NULL,
            feedback_key TEXT NOT NULL,
            story_id INTEGER NOT NULL,
            story_source TEXT NOT NULL,
            title TEXT NOT NULL,
            rank_index INTEGER NOT NULL,
            model_score REAL NOT NULL,
            knn_score REAL NOT NULL,
            max_sim_score REAL NOT NULL,
            max_cluster_score REAL NOT NULL,
            acquisition_kind TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            url TEXT
        )
    """)
    # Idempotent migration check for url column
    cursor = conn.execute("PRAGMA table_info(telemetry_events)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "url" not in columns:
        conn.execute("ALTER TABLE telemetry_events ADD COLUMN url TEXT")

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_telemetry_event_ts
            ON telemetry_events(event, server_ts)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_telemetry_config
            ON telemetry_events(config_hash, acquisition_kind, event)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_telemetry_key_ts
            ON telemetry_events(feedback_key, server_ts)
    """)
    conn.commit()
    return conn


def _insert_one(conn: sqlite3.Connection, rec: ImpressionRecord) -> None:
    d = asdict(rec)
    conn.execute(
        """INSERT INTO telemetry_events
           (event_id, event, server_ts, feedback_key, story_id, story_source,
            title, rank_index, model_score, knn_score, max_sim_score,
            max_cluster_score, acquisition_kind, config_hash, url)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            uuid.uuid4().hex,
            d["event"],
            d["timestamp"],
            d["feedback_key"],
            d["story_id"],
            d["story_source"],
            d["title"],
            d["rank_index"],
            d["model_score"],
            d["knn_score"],
            d["max_sim_score"],
            d["max_cluster_score"],
            d["acquisition_kind"],
            d["config_hash"],
            d["url"],
        ),
    )


def append_impressions(records: list[ImpressionRecord]) -> None:
    if not records:
        return
    conn = init_event_schema()
    try:
        for rec in records:
            _insert_one(conn, rec)
        conn.commit()
    finally:
        conn.close()


@dataclass
class ImpressionRecord:
    timestamp: float
    feedback_key: str
    story_id: int
    story_source: str
    title: str
    rank_index: int
    model_score: float
    knn_score: float
    max_sim_score: float
    max_cluster_score: float
    acquisition_kind: str
    config_hash: str
    url: str | None = None
    event: str = "impression"


def impression_from_payload(item: dict[str, Any]) -> ImpressionRecord | None:
    story_id = _int(item, "story_id")
    rank_index = _int(item, "rank_index")
    model_score = _float(item, "model_score")
    knn_score = _float(item, "knn_score")
    max_sim_score = _float(item, "max_sim_score")
    max_cluster_score = _float(item, "max_cluster_score")
    acquisition_kind = str(item.get("acquisition_kind", "exploit"))
    config_hash = str(item.get("config_hash", ""))
    event = str(item.get("event", "impression"))
    url = item.get("url")
    if url is not None:
        url = str(url).strip() or None

    if story_id is None or rank_index is None:
        return None
    if model_score is None:
        return None
    if acquisition_kind not in KNOWN_ACQUISITION_KINDS:
        acquisition_kind = "exploit"
    if event not in KNOWN_EVENTS:
        event = "impression"

    title = str(item.get("title", "")).strip()
    if not title:
        title = "Untitled"

    return ImpressionRecord(
        timestamp=time.time(),
        feedback_key=str(item.get("feedback_key", "")),
        story_id=story_id,
        story_source=str(item.get("story_source", "hn")),
        title=title,
        rank_index=rank_index,
        model_score=model_score,
        knn_score=knn_score or 0.0,
        max_sim_score=max_sim_score or 0.0,
        max_cluster_score=max_cluster_score or 0.0,
        acquisition_kind=acquisition_kind,
        config_hash=config_hash[:12],
        url=url,
        event=event,
    )


def _int(d: dict[str, Any], key: str) -> int | None:
    v = d.get(key)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return None


def _float(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None
