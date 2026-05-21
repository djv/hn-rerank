"""Dashboard feedback persistence and signal helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from api.cache_utils import atomic_write_json
from api.models import RankResult, Story, StorySource
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
    hybrid_score: NotRequired[float | None]
    semantic_score: NotRequired[float | None]
    hn_score: NotRequired[float | None]
    freshness_boost: NotRequired[float | None]
    knn_score: NotRequired[float | None]
    max_sim_score: NotRequired[float | None]
    max_cluster_score: NotRequired[float | None]
    cross_encoder_score: NotRequired[float | None]


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
    hybrid_score: NotRequired[float]
    semantic_score: NotRequired[float]
    hn_score: NotRequired[float]
    freshness_boost: NotRequired[float]
    knn_score: NotRequired[float]
    max_sim_score: NotRequired[float]
    max_cluster_score: NotRequired[float]
    cross_encoder_score: NotRequired[float]


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
    hybrid_score: float | None = None
    semantic_score: float | None = None
    hn_score: float | None = None
    freshness_boost: float | None = None
    knn_score: float | None = None
    max_sim_score: float | None = None
    max_cluster_score: float | None = None
    cross_encoder_score: float | None = None

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
            hybrid_score=_optional_float(data.get("hybrid_score")),
            semantic_score=_optional_float(data.get("semantic_score")),
            hn_score=_optional_float(data.get("hn_score")),
            freshness_boost=_optional_float(data.get("freshness_boost")),
            knn_score=_optional_float(data.get("knn_score")),
            max_sim_score=_optional_float(data.get("max_sim_score")),
            max_cluster_score=_optional_float(data.get("max_cluster_score")),
            cross_encoder_score=_optional_float(data.get("cross_encoder_score")),
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
            "hybrid_score": self.hybrid_score,
            "semantic_score": self.semantic_score,
            "hn_score": self.hn_score,
            "freshness_boost": self.freshness_boost,
            "knn_score": self.knn_score,
            "max_sim_score": self.max_sim_score,
            "max_cluster_score": self.max_cluster_score,
            "cross_encoder_score": self.cross_encoder_score,
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

    def to_rank_result(self) -> RankResult | None:
        required = (
            self.hybrid_score,
            self.semantic_score,
            self.hn_score,
            self.freshness_boost,
            self.knn_score,
            self.max_sim_score,
            self.max_cluster_score,
            self.cross_encoder_score,
        )
        if any(value is None for value in required):
            return None
        return RankResult(
            index=-1,
            hybrid_score=float(self.hybrid_score),
            best_fav_index=-1,
            max_sim_score=float(self.max_sim_score),
            knn_score=float(self.knn_score),
            max_cluster_score=float(self.max_cluster_score),
            semantic_score=float(self.semantic_score),
            hn_score=float(self.hn_score),
            freshness_boost=float(self.freshness_boost),
            cross_encoder_score=float(self.cross_encoder_score),
        )


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


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
        hybrid_score=_optional_float(payload.get("hybrid_score")),
        semantic_score=_optional_float(payload.get("semantic_score")),
        hn_score=_optional_float(payload.get("hn_score")),
        freshness_boost=_optional_float(payload.get("freshness_boost")),
        knn_score=_optional_float(payload.get("knn_score")),
        max_sim_score=_optional_float(payload.get("max_sim_score")),
        max_cluster_score=_optional_float(payload.get("max_cluster_score")),
        cross_encoder_score=_optional_float(payload.get("cross_encoder_score")),
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
