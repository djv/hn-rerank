import json
from typing import cast

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.provisional import urls

from api.feedback import (
    FEEDBACK_STORE_VERSION,
    FeedbackAction,
    FeedbackPayload,
    FeedbackRecord,
    apply_feedback_payload,
    feedback_action_for_story,
    feedback_key,
    load_feedback,
    record_from_payload,
    save_feedback,
)
from api.url_utils import normalize_url


def test_feedback_key_prefers_normalized_url():
    assert (
        feedback_key("hn", 123, "https://example.com/story?utm_source=x")
        == "https://example.com/story"
    )
    assert feedback_key("hn", 123, None) == "hn:123"


def test_feedback_store_latest_vote_wins_and_clear_removes(tmp_path):
    path = tmp_path / "feedback.json"
    payload: FeedbackPayload = {
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
    assert (
        feedback_action_for_story(
            records,
            source="hn",
            story_id=123,
            url="https://example.com/story",
        )
        == "up"
    )

    payload["action"] = "neutral"
    records, record = apply_feedback_payload(payload, path=path)

    assert record is not None
    assert len(records) == 1
    assert records[record.key].action == "neutral"

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


def test_feedback_store_ignores_runtime_rank_diagnostics_on_write(tmp_path):
    path = tmp_path / "feedback.json"
    payload: dict[str, object] = {
        "id": 123,
        "source": "hn",
        "title": "Ranked",
        "url": "https://example.com/ranked",
        "discussion_url": "https://news.ycombinator.com/item?id=123",
        "text_content": "Ranked text",
        "time": 1700000000,
        "score": 321,
        "comment_count": 45,
        "action": "up",
        "knn_score": 0.6,
        "max_sim_score": 0.5,
        "max_cluster_score": 0.4,
    }
    _, record = apply_feedback_payload(
        cast(FeedbackPayload, payload),
        path=path,
    )
    assert record is not None

    loaded = load_feedback(path)[record.key]
    story = loaded.to_story()

    assert story.score == 321
    assert story.comment_count == 45


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


# =============================================================================
# Property-based tests
# =============================================================================

_ACTION_VOTE = st.sampled_from(["up", "neutral", "down"])
_URL_STRATEGY = st.one_of(st.none(), urls())
_SOURCE_STRATEGY = st.sampled_from(["hn", "rss", "lobsters", "tildes", "lesswrong"])

_VOTE_PAYLOAD = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=1, max_value=10**9),
        "source": _SOURCE_STRATEGY,
        "title": st.text(min_size=1, max_size=200),
        "url": _URL_STRATEGY,
        "discussion_url": st.one_of(st.none(), urls().map(str)),
        "text_content": st.text(max_size=5000),
        "time": st.integers(min_value=0, max_value=2**31 - 1),
        "action": _ACTION_VOTE,
    }
)


@st.composite
def vote_sequence(draw):
    """A non-empty list of vote payloads targeting the same story."""
    n = draw(st.integers(min_value=1, max_value=10))
    base = draw(_VOTE_PAYLOAD)
    actions = [draw(_ACTION_VOTE) for _ in range(n)]
    return [{**base, "action": a} for a in actions]


# ---- 1. feedback_key uses normalized URL when URL is non-empty ----
@settings(max_examples=200, deadline=None)
@given(
    source=_SOURCE_STRATEGY,
    story_id=st.integers(min_value=0, max_value=10**9),
    url=_URL_STRATEGY,
)
def test_feedback_key_uses_normalized_url_when_present(
    source: str, story_id: int, url: str | None
) -> None:
    """If URL is non-empty and normalizes to a non-empty string, key is normalized URL."""
    key = feedback_key(source, story_id, url)
    if url:
        normalized = normalize_url(url)
        if normalized:
            assert key == normalized, f"expected {normalized!r} got {key!r}"
        else:
            assert key == f"{source}:{story_id}"
    else:
        assert key == f"{source}:{story_id}"


# ---- 2. feedback_key uses source:id fallback when URL is None ----
@settings(max_examples=100, deadline=None)
@given(
    source=_SOURCE_STRATEGY,
    story_id=st.integers(min_value=0, max_value=10**9),
)
def test_feedback_key_fallback_for_none_url(source: str, story_id: int) -> None:
    """If URL is None, the key is f'{source}:{story_id}'."""
    assert feedback_key(source, story_id, None) == f"{source}:{story_id}"


# ---- 3. Latest vote wins for any sequence ----
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payloads=vote_sequence())
def test_feedback_latest_vote_wins(tmp_path, payloads) -> None:
    """For any sequence of N votes on the same key, the final state matches the last vote."""
    path = tmp_path / "feedback.json"
    for payload in payloads:
        apply_feedback_payload(payload, path=path)
    final = load_feedback(path)
    expected_key = feedback_key(
        payloads[0]["source"],
        payloads[0]["id"],
        payloads[0]["url"],
    )
    assert expected_key in final, (
        f"key {expected_key} not found in {list(final.keys())}"
    )
    assert final[expected_key].action == payloads[-1]["action"]


# ---- 4. clear removes the record ----
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payloads=vote_sequence())
def test_feedback_clear_removes_record(tmp_path, payloads) -> None:
    """After applying a clear payload, the record is gone."""
    path = tmp_path / "feedback.json"
    for p in payloads:
        apply_feedback_payload(p, path=path)
    key = feedback_key(
        payloads[0]["source"],
        payloads[0]["id"],
        payloads[0]["url"],
    )
    assert key in load_feedback(path), "key should exist after votes"
    clear_payload = cast(FeedbackPayload, {**payloads[-1], "action": "clear"})
    records, _ = apply_feedback_payload(clear_payload, path=path)
    assert key not in records, "key should be removed after clear"
    assert key not in load_feedback(path)


# ---- 5. clear is idempotent ----
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payloads=vote_sequence())
def test_feedback_clear_is_idempotent(tmp_path, payloads) -> None:
    """Clearing a non-existent key is a no-op."""
    path = tmp_path / "feedback.json"
    clear_payload = cast(FeedbackPayload, {**payloads[0], "action": "clear"})
    records, returned = apply_feedback_payload(clear_payload, path=path)
    assert records == {}
    assert returned is None


# ---- 6. round-trip preserves records ----
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payloads=vote_sequence())
def test_feedback_round_trip_preserves_records(tmp_path, payloads) -> None:
    """save_feedback(records) then load_feedback() returns equivalent records."""
    path = tmp_path / "feedback.json"
    all_records: dict[str, FeedbackRecord] = {}
    for p in payloads:
        records, _ = apply_feedback_payload(p, path=path)
        all_records = records
    save_feedback(all_records, path)
    reloaded = load_feedback(path)
    assert set(reloaded.keys()) == set(all_records.keys())
    for k in reloaded:
        assert reloaded[k].action == all_records[k].action
        assert reloaded[k].title == all_records[k].title
        assert reloaded[k].id == all_records[k].id


# ---- 7. Different keys are independent ----
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(a=_VOTE_PAYLOAD, b=_VOTE_PAYLOAD)
def test_feedback_records_for_different_keys_independent(tmp_path, a, b) -> None:
    """A vote on key A doesn't affect the record for key B."""
    a = {**a, "id": 1, "url": "https://example.com/a"}
    b = {**b, "id": 2, "url": "https://example.com/b"}
    path = tmp_path / "feedback.json"
    apply_feedback_payload(cast(FeedbackPayload, a), path=path)
    apply_feedback_payload(cast(FeedbackPayload, b), path=path)
    records = load_feedback(path)
    key_a = feedback_key(
        cast(str, a["source"]),
        cast(int, a["id"]),
        cast(str | None, a["url"]),
    )
    key_b = feedback_key(
        cast(str, b["source"]),
        cast(int, b["id"]),
        cast(str | None, b["url"]),
    )
    assert key_a in records
    assert key_b in records
    assert records[key_a].action == a["action"]
    assert records[key_b].action == b["action"]


# ---- 8. to_story text_content fallback ----
@settings(max_examples=100, deadline=None)
@given(
    action=_ACTION_VOTE,
    title=st.text(min_size=1, max_size=200),
    text_content=st.text(min_size=1, max_size=2000),
)
def test_feedback_to_story_text_content_fallback(
    action: str, title: str, text_content: str
) -> None:
    """record.to_story().text_content equals title when text_content is empty."""
    rec_action = cast(FeedbackAction, action)
    record = FeedbackRecord(
        key="k",
        action=rec_action,
        id=1,
        source="hn",
        title=title,
        url=None,
        discussion_url=None,
        text_content="",
        time=1_700_000_000,
    )
    story = record.to_story()
    assert story.text_content == title
    record2 = FeedbackRecord(
        key="k",
        action=rec_action,
        id=1,
        source="hn",
        title=title,
        url=None,
        discussion_url=None,
        text_content=text_content,
        time=1_700_000_000,
    )
    story2 = record2.to_story()
    assert story2.text_content == text_content
