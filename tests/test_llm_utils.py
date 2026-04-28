import pytest

from api.llm_utils import (
    _coerce_tldr_value,
    _build_tldr_prompt,
    _finalize_cluster_name,
    _load_tldr_cache,
    _parse_retry_after,
    _safe_json_loads,
    build_messages,
    build_payload,
)


def test_build_messages_from_string():
    messages = build_messages("Hello")
    assert messages == [{"role": "user", "content": "Hello"}]


def test_build_messages_from_parts():
    contents = [
        {"parts": [{"text": "Hello"}, {"text": " world"}]},
        {"parts": [{"text": "Second"}]},
    ]
    messages = build_messages(contents)
    assert messages == [
        {"role": "user", "content": "Hello world"},
        {"role": "user", "content": "Second"},
    ]


def test_build_payload_json_response_format():
    payload = build_payload(
        model="test-model",
        contents="Hi",
        config={"response_mime_type": "application/json"},
    )
    assert payload["response_format"] == {"type": "json_object"}


def test_build_payload_max_tokens():
    payload = build_payload(
        model="test-model",
        contents="Hi",
        config={"max_output_tokens": 123},
    )
    assert payload["max_tokens"] == 123


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Software Engineering", "Software Engineering"),
        ("Graph Neural Networks for Drug Discovery Pipelines", "Graph Neural Networks for Drug Discovery"),
        ("Research and", "Research"),
        ("", None),
        ("Bad\nName", None),
        ("{\"name\": \"Bad\"}", None),
    ],
)
def test_finalize_cluster_name(raw: str, expected: str | None):
    assert _finalize_cluster_name(raw) == expected


def test_safe_json_loads_extracts_markdown_object():
    assert _safe_json_loads('```json\n{"tldr": "Works"}\n```') == {"tldr": "Works"}


def test_safe_json_loads_falls_back_to_empty_dict():
    assert _safe_json_loads("not json") == {}


def test_parse_retry_after_seconds():
    assert _parse_retry_after("2.5") == pytest.approx(2.5)
    assert _parse_retry_after("") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Plain summary", "Plain summary"),
        ({"summary": "Summary from object"}, "Summary from object"),
        ({"title": "Ignored", "summary": "Summary wins"}, "Summary wins"),
        ({"tldr": "Alternate key"}, "Alternate key"),
        ({"title": "No summary"}, ""),
        ({"summary": 123}, ""),
    ],
)
def test_coerce_tldr_value(raw: object, expected: str):
    assert _coerce_tldr_value(raw) == expected


def test_load_tldr_cache_normalizes_dict_values(tmp_path, monkeypatch):
    cache_path = tmp_path / "tldrs.json"
    cache_path.write_text(
        '{"1": {"title": "Story", "summary": "Only this text"}, "2": "Already clean"}'
    )
    monkeypatch.setattr("api.llm_utils.TLDR_CACHE_PATH", cache_path)

    assert _load_tldr_cache() == {
        "1": "Only this text",
        "2": "Already clean",
    }


def test_build_tldr_prompt_requires_flat_json_strings():
    prompt = _build_tldr_prompt(
        [
            "### STORY ID: 123 ###\nTitle: Example\nContent: Body\nComments:\n- One",
        ]
    )

    assert "Return a flat JSON object mapping each requested story ID string to a plain string summary." in prompt
    assert "Do NOT return nested objects, arrays, or metadata fields." in prompt
    assert '"12345": {' in prompt
    assert '"summary": "Nested objects are not allowed."' in prompt
    assert '"12345": "Tool X implements lock-free concurrency' in prompt
