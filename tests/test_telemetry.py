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
    with (
        patch("api.impressions.DB_PATH", db_path),
        patch("api.telemetry_features.DB_PATH", db_path),
    ):
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
    rows = conn.execute(
        "SELECT feedback_key, url, event FROM telemetry_events"
    ).fetchall()
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


def test_click_has_url(patch_db):
    """Click event payload includes url field and roundtrips through the DB."""
    rec = ImpressionRecord(
        timestamp=200.0,
        feedback_key="http://example.com/article",
        story_id=789,
        story_source="rss",
        title="Click Test",
        rank_index=3,
        model_score=0.7,
        knn_score=0.6,
        max_sim_score=0.5,
        max_cluster_score=0.0,
        acquisition_kind="exploit",
        config_hash="clickhash",
        url="http://example.com/article",
        event="click",
    )
    append_impressions([rec])

    conn = connect_db()
    rows = conn.execute(
        "SELECT event, url FROM telemetry_events WHERE story_id = 789"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["event"] == "click"
    assert rows[0]["url"] == "http://example.com/article"
    conn.close()


def test_telemetry_features_aggregation(patch_db):
    from api.telemetry_features import (
        fetch_impression_stats,
        load_telemetry_stats,
        reset_telemetry_stats_cache,
    )
    from api.models import Story
    from api.rerank import METADATA_FEATURES
    import time

    reset_telemetry_stats_cache()

    # 1. Empty DB stats
    story_stats, domain_stats = fetch_impression_stats()
    assert story_stats == {}
    assert domain_stats == {}

    # 2. Populate DB
    now = time.time()
    recs = [
        # Story 1: 2 impressions, 1 click
        ImpressionRecord(
            timestamp=now - 3600,
            feedback_key="rss:1",
            story_id=1,
            story_source="rss",
            title="Story 1",
            rank_index=0,
            model_score=0.5,
            knn_score=0.5,
            max_sim_score=0.5,
            max_cluster_score=0.0,
            acquisition_kind="exploit",
            config_hash="test",
            url="https://github.com/foo/bar",
            event="impression",
        ),
        ImpressionRecord(
            timestamp=now - 1800,
            feedback_key="rss:1",
            story_id=1,
            story_source="rss",
            title="Story 1",
            rank_index=0,
            model_score=0.5,
            knn_score=0.5,
            max_sim_score=0.5,
            max_cluster_score=0.0,
            acquisition_kind="exploit",
            config_hash="test",
            url="https://github.com/foo/bar",
            event="impression",
        ),
        ImpressionRecord(
            timestamp=now - 600,
            feedback_key="rss:1",
            story_id=1,
            story_source="rss",
            title="Story 1",
            rank_index=0,
            model_score=0.5,
            knn_score=0.5,
            max_sim_score=0.5,
            max_cluster_score=0.0,
            acquisition_kind="exploit",
            config_hash="test",
            url="https://github.com/foo/bar",
            event="click",
        ),
        # Story 2 (HN post): 1 impression, 0 clicks, no URL
        ImpressionRecord(
            timestamp=now - 7200,
            feedback_key="hn:2",
            story_id=2,
            story_source="hn",
            title="Story 2",
            rank_index=1,
            model_score=0.5,
            knn_score=0.5,
            max_sim_score=0.5,
            max_cluster_score=0.0,
            acquisition_kind="exploit",
            config_hash="test",
            url="",
            event="impression",
        ),
    ]
    append_impressions(recs)

    # 3. Test stats extraction
    story_stats, domain_stats = fetch_impression_stats()
    assert 1 in story_stats
    assert story_stats[1].impression_count == 2
    assert story_stats[1].click_count == 1
    assert story_stats[1].click_ratio == 0.5
    assert story_stats[1].days_since_last_impression < 1.0

    assert 2 in story_stats
    assert story_stats[2].impression_count == 1
    assert story_stats[2].click_count == 0
    assert story_stats[2].click_ratio == 0.0

    # Domain levels: github.com and hn.text
    assert "github.com" in domain_stats
    assert domain_stats["github.com"].domain_impression_count == 2
    assert domain_stats["github.com"].domain_ctr == 0.5

    assert "hn.text" in domain_stats
    assert domain_stats["hn.text"].domain_impression_count == 1
    assert domain_stats["hn.text"].domain_ctr == 0.0

    # 4. Lazy Cache / global reload test
    with patch("api.telemetry_features.fetch_impression_stats") as mock_fetch:
        mock_fetch.return_value = (story_stats, domain_stats)
        s1, d1 = load_telemetry_stats()
        s2, d2 = load_telemetry_stats()
        assert s1 is s2
        assert mock_fetch.call_count == 1

    # 5. Metadata features array integration and bounding
    s_cands = [
        Story(
            id=1,
            title="S1",
            url="https://github.com/foo/bar",
            score=10,
            time=int(now),
            text_content="t1",
            source="rss",
            comment_count=0,
        ),
        Story(
            id=2,
            title="S2",
            url="",
            score=20,
            time=int(now),
            text_content="t2",
            source="hn",
            comment_count=0,
        ),
        Story(
            id=99,
            title="S99",
            url="https://newdomain.com/abc",
            score=0,
            time=int(now),
            text_content="t99",
            source="rss",
            comment_count=0,
        ),
    ]

    # Warm caching
    reset_telemetry_stats_cache()
    load_telemetry_stats()

    # Test impression_count
    arr = METADATA_FEATURES["impression_count"](s_cands, now)
    assert arr.shape == (3, 1)
    # Story 1: 2 impressions. log1p(2) / log1p(200) = ~0.207
    assert 0.20 < arr[0][0] < 0.22
    # Story 2: 1 impression. log1p(1) / log1p(200) = ~0.13
    assert 0.12 < arr[1][0] < 0.14
    # Story 99: 0 impressions = 0.0
    assert arr[2][0] == 0.0

    # Test click_count
    arr = METADATA_FEATURES["click_count"](s_cands, now)
    assert arr.shape == (3, 1)
    # Story 1: 1 click. log1p(1) / log1p(20) = ~0.227
    assert 0.22 < arr[0][0] < 0.24
    assert arr[1][0] == 0.0
    assert arr[2][0] == 0.0

    # Test click_ratio
    arr = METADATA_FEATURES["click_ratio"](s_cands, now)
    assert arr[0][0] == 0.5
    assert arr[1][0] == 0.0
    assert arr[2][0] == 0.0

    # Test days_since_last_impression
    arr = METADATA_FEATURES["days_since_last_impression"](s_cands, now)
    # Story 1: seen recently (< 1.0 day). (30 - days_since) / 30 = ~0.99
    assert 0.95 < arr[0][0] <= 1.0
    # Story 99: never seen = 0.0
    assert arr[2][0] == 0.0

    # Test domain_ctr
    arr = METADATA_FEATURES["domain_ctr"](s_cands, now)
    # github.com: 0.5
    assert arr[0][0] == 0.5
    # hn.text: 0.0
    assert arr[1][0] == 0.0
    # newdomain.com: 0.0
    assert arr[2][0] == 0.0

    # Test domain_impression_count
    arr = METADATA_FEATURES["domain_impression_count"](s_cands, now)
    # github.com: 2 impressions. log1p(2) / log1p(500) = ~0.176
    assert 0.17 < arr[0][0] < 0.19
    # hn.text: 1 impression. log1p(1) / log1p(500) = ~0.111
    assert 0.10 < arr[1][0] < 0.12
    # newdomain.com: 0.0
    assert arr[2][0] == 0.0

    reset_telemetry_stats_cache()
