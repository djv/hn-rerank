import json

import pytest

from api.feedback import (
    FEEDBACK_STORE_VERSION,
    apply_feedback_payload,
    feedback_action_for_story,
    feedback_key,
    load_feedback,
    record_from_payload,
    save_feedback,
)


def test_feedback_key_prefers_normalized_url():
    assert (
        feedback_key("hn", 123, "https://example.com/story?utm_source=x")
        == "https://example.com/story"
    )
    assert feedback_key("hn", 123, None) == "hn:123"


def test_feedback_store_latest_vote_wins_and_clear_removes(tmp_path):
    path = tmp_path / "feedback.json"
    payload = {
        "id": 123,
        "source": "hn",
        "title": "Story",
        "url": "https://example.com/story",
        "discussion_url": "https://news.ycombinator.com/item?id=123",
        "text_content": "Story text",
        "time": 1700000000,
        "action": "up",
    }

    records, record = apply_feedback_payload(payload, path=path)

    assert record is not None
    assert records[record.key].action == "up"
    assert feedback_action_for_story(
        records,
        source="hn",
        story_id=123,
        url="https://example.com/story",
    ) == "up"

    payload["action"] = "down"
    records, record = apply_feedback_payload(payload, path=path)

    assert record is not None
    assert len(records) == 1
    assert records[record.key].action == "down"

    payload["action"] = "clear"
    records, record = apply_feedback_payload(payload, path=path)

    assert record is None
    assert records == {}


def test_feedback_store_ignores_malformed_json(tmp_path):
    path = tmp_path / "feedback.json"
    path.write_text("{not-json")

    assert load_feedback(path) == {}


def test_feedback_store_round_trips_records(tmp_path):
    path = tmp_path / "feedback.json"
    _, record = apply_feedback_payload(
        {
            "id": -1,
            "source": "rss",
            "title": "External",
            "url": "https://example.com/external",
            "discussion_url": None,
            "text_content": "External text",
            "time": 1700000000,
            "action": "down",
        },
        path=path,
        mirror_status="none",
    )
    assert record is not None

    save_feedback({record.key: record}, path)
    raw = json.loads(path.read_text())
    loaded = load_feedback(path)

    assert raw["version"] == FEEDBACK_STORE_VERSION
    assert loaded[record.key].to_story().title == "External"
    assert loaded[record.key].action == "down"


def test_feedback_store_round_trips_rank_diagnostics(tmp_path):
    path = tmp_path / "feedback.json"
    _, record = apply_feedback_payload(
        {
            "id": 123,
            "source": "hn",
            "title": "Ranked",
            "url": "https://example.com/ranked",
            "discussion_url": "https://news.ycombinator.com/item?id=123",
            "text_content": "Ranked text",
            "time": 1700000000,
            "action": "up",
            "hybrid_score": 0.9,
            "semantic_score": 0.8,
            "hn_score": 0.7,
            "freshness_boost": 0.1,
            "knn_score": 0.6,
            "max_sim_score": 0.5,
            "max_cluster_score": 0.4,
            "cross_encoder_score": 0.3,
        },
        path=path,
    )
    assert record is not None

    loaded = load_feedback(path)[record.key]
    rank_result = loaded.to_rank_result()

    assert rank_result is not None
    assert rank_result.hybrid_score == pytest.approx(0.9)
    assert rank_result.semantic_score == pytest.approx(0.8)
    assert rank_result.cross_encoder_score == pytest.approx(0.3)


def test_feedback_payload_requires_record_for_clear():
    with pytest.raises(ValueError):
        record_from_payload(
            {
                "id": 1,
                "source": "hn",
                "title": "Story",
                "url": None,
                "discussion_url": None,
                "action": "clear",
            }
        )
