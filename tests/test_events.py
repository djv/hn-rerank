"""Tests for SQLite-backed event logging."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from api.impressions import (
    ImpressionRecord,
    append_impressions,
    connect_db,
    impression_from_payload,
)
from api.regen_scheduler import request_regen


@pytest.fixture
def patch_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("api.impressions.DB_PATH", db_path):
        yield


def test_append_preserves_prior_rows(patch_db):
    r1 = ImpressionRecord(
        timestamp=100.0,
        feedback_key="a",
        story_id=1,
        story_source="hn",
        title="Title A",
        rank_index=0,
        model_score=0.9,
        knn_score=0.8,
        max_sim_score=0.85,
        max_cluster_score=0.0,
        acquisition_kind="exploit",
        config_hash="abc123",
        event="impression",
    )
    r2 = ImpressionRecord(
        timestamp=200.0,
        feedback_key="b",
        story_id=2,
        story_source="hn",
        title="Title B",
        rank_index=1,
        model_score=0.8,
        knn_score=0.7,
        max_sim_score=0.75,
        max_cluster_score=0.0,
        acquisition_kind="exploit",
        config_hash="abc123",
        event="click",
    )
    append_impressions([r1])
    append_impressions([r2])

    conn = connect_db()
    rows = conn.execute(
        "SELECT feedback_key, event FROM telemetry_events ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["feedback_key"] == "a"
    assert rows[0]["event"] == "impression"
    assert rows[1]["feedback_key"] == "b"
    assert rows[1]["event"] == "click"
    conn.close()


def test_batch_insert(patch_db):
    records = [
        ImpressionRecord(
            timestamp=float(i),
            feedback_key=f"k{i}",
            story_id=i,
            story_source="hn",
            title=f"Title {i}",
            rank_index=i,
            model_score=0.9 - i * 0.1,
            knn_score=0.8,
            max_sim_score=0.85,
            max_cluster_score=0.0,
            acquisition_kind="exploit",
            config_hash="abc123",
            event="impression",
        )
        for i in range(5)
    ]
    append_impressions(records)

    conn = connect_db()
    count = conn.execute("SELECT COUNT(*) as c FROM telemetry_events").fetchone()["c"]
    assert count == 5
    conn.close()


def test_empty_batch_does_nothing(patch_db, tmp_path):
    append_impressions([])
    db_path = tmp_path / "test.db"
    assert not db_path.exists()


def test_fresh_db_no_dir_creates_path(patch_db, tmp_path):
    deep_path = tmp_path / "a" / "b" / "c" / "test.db"
    with patch("api.impressions.DB_PATH", deep_path):
        conn = connect_db()
        assert deep_path.parent.exists()
        conn.close()


def test_payload_valid_impression():
    payload = {
        "event": "impression",
        "feedback_key": "hn:123",
        "story_id": 123,
        "story_source": "hn",
        "title": "Test Story",
        "rank_index": 3,
        "model_score": 0.85,
        "knn_score": 0.70,
        "max_sim_score": 0.80,
        "max_cluster_score": 0.0,
        "acquisition_kind": "novel",
        "config_hash": "abc123",
    }
    rec = impression_from_payload(payload)
    assert rec is not None
    assert rec.story_id == 123
    assert rec.event == "impression"
    assert rec.acquisition_kind == "novel"
    assert rec.config_hash == "abc123"


def test_payload_click():
    payload = {
        "story_id": 1,
        "rank_index": 0,
        "model_score": 0.5,
        "title": "Test",
        "event": "click",
    }
    rec = impression_from_payload(payload)
    assert rec is not None
    assert rec.event == "click"


def test_payload_bad_data():
    assert impression_from_payload({}) is None
    assert impression_from_payload({"story_id": "bad", "rank_index": 0}) is None
    assert impression_from_payload({"story_id": 1, "rank_index": "bad"}) is None
    assert impression_from_payload({"story_id": 1, "rank_index": 0}) is None


def test_request_regen_spawns_process():
    import api.regen_scheduler as rs

    rs._regen_process = None
    with patch("subprocess.Popen") as mock_popen:
        request_regen()
    mock_popen.assert_called_once()


def test_request_regen_skips_when_running():
    import api.regen_scheduler as rs

    rs._regen_process = None
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = None
        request_regen()
        first_call = mock_popen.call_count
        request_regen()
        assert mock_popen.call_count == first_call
    rs._regen_process = None


def test_concurrent_appends_no_data_loss(patch_db):
    import concurrent.futures

    threads = 8
    records_per = 10

    def worker(n: int):
        batch = [
            ImpressionRecord(
                timestamp=float(n * records_per + i),
                feedback_key=f"w{n}_r{i}",
                story_id=n * records_per + i,
                story_source="hn",
                title=f"Worker {n} Row {i}",
                rank_index=i,
                model_score=0.5,
                knn_score=0.5,
                max_sim_score=0.5,
                max_cluster_score=0.0,
                acquisition_kind="exploit",
                config_hash="test",
                event="impression",
            )
            for i in range(records_per)
        ]
        append_impressions(batch)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(worker, n) for n in range(threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    conn = connect_db()
    count = conn.execute("SELECT COUNT(*) as c FROM telemetry_events").fetchone()["c"]
    assert count == threads * records_per
    conn.close()
