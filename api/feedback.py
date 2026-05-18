"""Dashboard feedback persistence and signal helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from api.cache_utils import atomic_write_json
from api.models import Story, StorySource
from api.url_utils import normalize_url

FeedbackAction = Literal["up", "down"]
FeedbackMirrorStatus = Literal["none", "success", "failed"]

FEEDBACK_STORE_PATH = Path(".cache/user_feedback/dashboard_feedback.json")
FEEDBACK_STORE_VERSION = 1


class FeedbackRecordDict(TypedDict):
    key: str
    action: FeedbackAction
    id: int
    source: StorySource
    title: str
    url: str | None
    discussion_url: str | None
    text_content: str
    time: int
    hn_mirror_status: FeedbackMirrorStatus
    hn_mirror_error: str | None
    updated_at: float


class FeedbackStoreFile(TypedDict):
    version: int
    records: dict[str, FeedbackRecordDict]


class FeedbackPayload(TypedDict):
    id: int
    source: StorySource
    title: str
    url: str | None
    discussion_url: str | None
    text_content: NotRequired[str]
    time: NotRequired[int]
    action: Literal["up", "down", "clear"]


@dataclass(frozen=True)
class FeedbackRecord:
    key: str
    action: FeedbackAction
    id: int
    source: StorySource
    title: str
    url: str | None
    discussion_url: str | None
    text_content: str
    time: int
    hn_mirror_status: FeedbackMirrorStatus = "none"
    hn_mirror_error: str | None = None
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, data: FeedbackRecordDict) -> FeedbackRecord:
        return cls(
            key=str(data["key"]),
            action=cast(FeedbackAction, data["action"]),
            id=int(data["id"]),
            source=cast(StorySource, data.get("source", "hn")),
            title=str(data.get("title", "")),
            url=data.get("url"),
            discussion_url=data.get("discussion_url"),
            text_content=str(data.get("text_content", "")),
            time=int(data.get("time", 0)),
            hn_mirror_status=cast(
                FeedbackMirrorStatus, data.get("hn_mirror_status", "none")
            ),
            hn_mirror_error=data.get("hn_mirror_error"),
            updated_at=float(data.get("updated_at", 0.0)),
        )

    def to_dict(self) -> FeedbackRecordDict:
        return {
            "key": self.key,
            "action": self.action,
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "discussion_url": self.discussion_url,
            "text_content": self.text_content,
            "time": self.time,
            "hn_mirror_status": self.hn_mirror_status,
            "hn_mirror_error": self.hn_mirror_error,
            "updated_at": self.updated_at,
        }

    def to_story(self) -> Story:
        text_content = self.text_content or self.title
        return Story(
            id=self.id,
            title=self.title,
            url=self.url,
            score=0,
            time=self.time,
            discussion_url=self.discussion_url,
            comments=[],
            text_content=text_content,
            source=self.source,
        )


def feedback_key(source: str, story_id: int, url: str | None) -> str:
    if url:
        normalized = normalize_url(url)
        if normalized:
            return normalized
    return f"{source}:{story_id}"


def record_from_payload(
    payload: FeedbackPayload,
    *,
    mirror_status: FeedbackMirrorStatus = "none",
    mirror_error: str | None = None,
) -> FeedbackRecord:
    action = payload["action"]
    if action == "clear":
        raise ValueError("clear payloads do not produce records")

    source = payload.get("source", "hn")
    story_id = int(payload["id"])
    url = payload.get("url")
    key = feedback_key(source, story_id, url)
    title = str(payload.get("title", "")).strip()
    text_content = str(payload.get("text_content", "")).strip() or title
    return FeedbackRecord(
        key=key,
        action=action,
        id=story_id,
        source=source,
        title=title,
        url=url,
        discussion_url=payload.get("discussion_url"),
        text_content=text_content,
        time=int(payload.get("time", 0)),
        hn_mirror_status=mirror_status,
        hn_mirror_error=mirror_error,
        updated_at=time.time(),
    )


def load_feedback(path: Path = FEEDBACK_STORE_PATH) -> dict[str, FeedbackRecord]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict) or raw.get("version") != FEEDBACK_STORE_VERSION:
        return {}
    records = raw.get("records")
    if not isinstance(records, dict):
        return {}

    parsed: dict[str, FeedbackRecord] = {}
    for key, value in records.items():
        if not isinstance(value, dict):
            continue
        try:
            record = FeedbackRecord.from_dict(cast(FeedbackRecordDict, value))
        except (KeyError, TypeError, ValueError):
            continue
        parsed[str(key)] = record
    return parsed


def save_feedback(
    records: dict[str, FeedbackRecord],
    path: Path = FEEDBACK_STORE_PATH,
) -> None:
    payload: FeedbackStoreFile = {
        "version": FEEDBACK_STORE_VERSION,
        "records": {key: record.to_dict() for key, record in records.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def apply_feedback_payload(
    payload: FeedbackPayload,
    *,
    path: Path = FEEDBACK_STORE_PATH,
    mirror_status: FeedbackMirrorStatus = "none",
    mirror_error: str | None = None,
) -> tuple[dict[str, FeedbackRecord], FeedbackRecord | None]:
    story_id = int(payload["id"])
    source = payload.get("source", "hn")
    key = feedback_key(source, story_id, payload.get("url"))
    records = load_feedback(path)
    if payload["action"] == "clear":
        records.pop(key, None)
        save_feedback(records, path)
        return records, None

    record = record_from_payload(
        payload,
        mirror_status=mirror_status,
        mirror_error=mirror_error,
    )
    records[record.key] = record
    save_feedback(records, path)
    return records, record


def feedback_action_for_story(
    records: dict[str, FeedbackRecord],
    *,
    source: str,
    story_id: int,
    url: str | None,
) -> FeedbackAction | None:
    record = records.get(feedback_key(source, story_id, url))
    return record.action if record else None

