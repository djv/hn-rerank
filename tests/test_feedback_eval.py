"""Tests for the full-feedback eval pipeline (build_dataset_from_feedback)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.models import Story


def _make_record(
    i: int,
    action: str = "up",
) -> dict:
    return {
        "key": f"hn:{i}",
        "action": action,
        "id": i,
        "source": "hn",
        "title": f"Story {i}",
        "url": None,
        "discussion_url": f"https://news.ycombinator.com/item?id={i}",
        "text_content": f"Content for story {i}.",
        "time": 100000 + i,
        "score": 10 + i,
        "comment_count": 0,
        "hn_mirror_status": "success",
        "hn_mirror_error": None,
        "updated_at": 200000.0 + i,
        "acquisition_kind": "exploit",
    }


@pytest.fixture
def feedback_file(tmp_path: Path) -> Path:
    """Write a synthetic feedback file with 110 up, 40 down, 30 neutral."""
    records: dict[str, dict] = {}
    for i in range(110):
        r = _make_record(i, "up")
        records[r["key"]] = r
    for i in range(110, 150):
        r = _make_record(i, "down")
        records[r["key"]] = r
    for i in range(150, 180):
        r = _make_record(i, "neutral")
        records[r["key"]] = r
    payload = {"version": 1, "records": records}
    path = tmp_path / "dashboard_feedback.json"
    path.write_text(json.dumps(payload))
    return path


def test_feedback_has_expected_counts(feedback_file: Path) -> None:
    """Sanity check for the fixture."""
    from api.feedback import load_feedback

    records = load_feedback(feedback_file)
    ups = [r for r in records.values() if r.action == "up"]
    downs = [r for r in records.values() if r.action == "down"]
    neutrals = [r for r in records.values() if r.action == "neutral"]
    assert len(records) == 180
    assert len(ups) == 110
    assert len(downs) == 40
    assert len(neutrals) == 30


def test_build_dataset_from_feedback(feedback_file: Path) -> None:
    """build_dataset_from_feedback stores all ups; candidates = neg fallback."""
    from evaluate_quality import RankingEvaluator

    ev = RankingEvaluator(username="test")
    ok = ev.build_dataset_from_feedback(feedback_file, holdout=0.20)
    assert ok
    ds = ev.dataset
    assert ds is not None
    assert len(ds.test_stories) == 22
    assert len(ds.train_stories) == 88
    # No snapshot provided → fallback: candidates = neg_stories
    assert len(ds.candidates) == 70
    assert len(ds.neg_stories) == 70
    train_ids = {s.id for s in ds.train_stories}
    test_ids = {s.id for s in ds.test_stories}
    assert train_ids.isdisjoint(test_ids)
    neg_ids = {s.id for s in ds.neg_stories}
    assert neg_ids.isdisjoint(train_ids)
    assert ds.train_embeddings.shape[0] == 88
    assert ds.train_embeddings.shape[1] > 0


def test_build_dataset_from_feedback_with_snapshot_candidates(
    feedback_file: Path, tmp_path: Path
) -> None:
    """Cache snapshot candidates are used as the eval pool."""
    from evaluate_quality import RankingEvaluator

    snap_path = tmp_path / "baseline.json"
    snap = {
        "format_version": 1,
        "username": "test",
        "saved_at": 0,
        "metadata": {},
        "train_stories": [],
        "test_stories": [],
        "neg_stories": [],
        "candidates": [
            Story(id=200, title="c", url=None, score=0, time=0).to_dict(),
            Story(id=201, title="d", url=None, score=0, time=0).to_dict(),
        ],
        "test_ids": [],
    }
    snap_path.write_text(json.dumps(snap))

    ev = RankingEvaluator(username="test")
    ok = ev.build_dataset_from_feedback(
        feedback_file,
        snapshot_candidates_path=snap_path,
        holdout=0.20,
    )
    assert ok
    ds = ev.dataset
    assert ds is not None
    assert len(ds.candidates) == 2
    assert len(ds.neg_stories) == 70


def test_build_dataset_not_enough_ups(tmp_path: Path) -> None:
    """Fewer than 10 ups returns False."""
    from evaluate_quality import RankingEvaluator

    records: dict[str, dict] = {}
    for i in range(5):
        r = _make_record(i, "up")
        records[r["key"]] = r
    path = tmp_path / "dashboard_feedback.json"
    path.write_text(json.dumps({"version": 1, "records": records}))

    ev = RankingEvaluator(username="test")
    ok = ev.build_dataset_from_feedback(path, holdout=0.20)
    assert not ok


def test_build_dataset_no_feedback_file(tmp_path: Path) -> None:
    """Missing feedback file returns False with graceful error."""
    from evaluate_quality import RankingEvaluator

    ev = RankingEvaluator(username="test")
    ok = ev.build_dataset_from_feedback(tmp_path / "nonexistent.json")
    assert not ok
