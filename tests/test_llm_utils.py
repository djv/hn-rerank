from api.llm_utils import build_messages, build_payload


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
