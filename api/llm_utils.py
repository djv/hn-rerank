from __future__ import annotations

from typing import cast

from api.constants import LLM_TEMPERATURE


def build_messages(contents: object | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, object], item)
            parts = item_dict.get("parts")
            if not isinstance(parts, list):
                continue
            texts: list[str] = []
            for part in parts:
                if isinstance(part, dict):
                    part_dict = cast(dict[str, object], part)
                    text_val = part_dict.get("text")
                    if isinstance(text_val, str):
                        texts.append(text_val)
            if texts:
                messages.append({"role": "user", "content": "".join(texts)})
    return messages


def build_payload(
    model: str,
    contents: object | None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "messages": build_messages(contents),
        "temperature": config.get("temperature", LLM_TEMPERATURE)
        if config
        else LLM_TEMPERATURE,
    }

    if config:
        max_tokens = config.get("max_tokens", config.get("max_output_tokens"))
        if isinstance(max_tokens, (int, float)) and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)

    if config and config.get("response_mime_type") == "application/json":
        payload["response_format"] = {"type": "json_object"}

    return payload
