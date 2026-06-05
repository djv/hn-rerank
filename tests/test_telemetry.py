"""Tests for telemetry schema changes and URL payload parsing."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from api.impressions import (
    ImpressionRecord,
    append_impressions,
    connect_db,
    init_event_schema,
    impression_from_payload,
)

@pytest.fixture
def patch_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("api.impressions.DB_PATH", db_path):
        yield

def test_url_column_exists(patch_db):
    conn = init_event_schema()
    cursor = conn.execute("PRAGMA table_info(telemetry_events)")
    columns = [row["name"] for row in cursor.fetchall()]
    assert "url" in columns
    conn.close()

def test_url_column_idempotent(patch_db):
    conn = init_event_schema()
    # Call a second time, it should execute ALTER TABLE but ignore the column presence
    conn = init_event_schema(conn)
    cursor = conn.execute("PRAGMA table_info(telemetry_events)")
    columns = [row["name"] for row in cursor.fetchall()]
    assert "url" in columns
    # Verify we still only have one 'url' column
    assert columns.count("url") == 1
    conn.close()

def test_url_roundtrip(patch_db):
    rec = ImpressionRecord(
        timestamp=123.45,
        feedback_key="http://example.com/foo",
        story_id=456,
        story_source="rss",
        title="Example Title",
        rank_index=2,
        model_score=0.9,
        knn_score=0.8,
        max_sim_score=0.85,
        max_cluster_score=0.0,
        acquisition_kind="exploit",
        config_hash="hash123",
        url="http://example.com/foo",
        event="impression",
    )
    append_impressions([rec])
    
    conn = connect_db()
    rows = conn.execute("SELECT feedback_key, url, event FROM telemetry_events").fetchall()
    assert len(rows) == 1
    assert rows[0]["feedback_key"] == "http://example.com/foo"
    assert rows[0]["url"] == "http://example.com/foo"
    assert rows[0]["event"] == "impression"
    conn.close()

def test_url_insert_without_url(patch_db):
    rec = ImpressionRecord(
        timestamp=123.45,
        feedback_key="hn:123",
        story_id=123,
        story_source="hn",
        title="HN Title",
        rank_index=0,
        model_score=0.9,
        knn_score=0.8,
        max_sim_score=0.85,
        max_cluster_score=0.0,
        acquisition_kind="exploit",
        config_hash="hash123",
        event="impression",
        # url omitted (defaults to None)
    )
    append_impressions([rec])
    
    conn = connect_db()
    rows = conn.execute("SELECT feedback_key, url FROM telemetry_events").fetchall()
    assert len(rows) == 1
    assert rows[0]["feedback_key"] == "hn:123"
    assert rows[0]["url"] is None
    conn.close()

def test_parser_handles_url():
    payload = {
        "event": "impression",
        "feedback_key": "http://example.com",
        "story_id": 999,
        "story_source": "reddit",
        "title": "Reddit Story",
        "rank_index": 5,
        "model_score": 0.5,
        "knn_score": 0.5,
        "max_sim_score": 0.5,
        "max_cluster_score": 0.0,
        "acquisition_kind": "exploit",
        "config_hash": "hash",
        "url": "https://example.com/original",
    }
    rec = impression_from_payload(payload)
    assert rec is not None
    assert rec.url == "https://example.com/original"

def test_parser_handles_missing_url():
    payload = {
        "event": "impression",
        "feedback_key": "hn:123",
        "story_id": 123,
        "story_source": "hn",
        "title": "HN Story",
        "rank_index": 5,
        "model_score": 0.5,
        "knn_score": 0.5,
        "max_sim_score": 0.5,
        "max_cluster_score": 0.0,
        "acquisition_kind": "exploit",
        "config_hash": "hash",
        # url missing
    }
    rec = impression_from_payload(payload)
    assert rec is not None
    assert rec.url is None
