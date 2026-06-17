"""Typed data models for HN reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict, cast

from api.external_sources import source_badge_label as _source_badge_label

StorySource = Literal[
    "hn",
    "rss",
    "lobsters",
    "tildes",
    "lesswrong",
    "slashdot",
    "github_trending",
    "reddit",
    "reddit_machinelearning",
    "reddit_programming",
    "reddit_compsci",
    "digg",
    "haskell_discourse",
]


def source_badge_label(source: StorySource) -> str | None:
    return _source_badge_label(source)


class StoryDict(TypedDict):
    """Serialized Story payload for caching and API boundaries."""

    id: int
    title: str
    url: str | None
    score: int
    time: int
    discussion_url: str | None
    comments: list[str]
    text_content: str
    source: StorySource
    comment_count: NotRequired[int | None]
    feedback_updated_at: NotRequired[float]


class StoryForTldr(TypedDict):
    """Minimum fields needed for TL;DR generation."""

    id: int
    title: str
    comments: list[str]
    text_content: str


class StoryDisplayDict(StoryForTldr):
    """Serialized StoryDisplay payload for templates and LLMs."""

    match_percent: int
    cluster_name: str
    points: int
    time_ago: str
    time: int
    url: str | None
    hn_url: str | None
    reason: str
    reason_url: str
    source: StorySource
    tldr: str
    rank_index: NotRequired[int]
    model_score: NotRequired[float]
    knn_score: NotRequired[float]
    max_sim_score: NotRequired[float]
    max_cluster_score: NotRequired[float]
    comment_count: NotRequired[int | None]
    feedback_action: NotRequired[Literal["up", "neutral", "down"] | None]
    acquisition_kind: NotRequired[str]


@dataclass
class Story:
    """A Hacker News story with extracted content."""

    id: int
    title: str
    url: str | None
    score: int
    time: int
    discussion_url: str | None = None
    comments: list[str] = field(default_factory=list)
    text_content: str = ""
    source: StorySource = "hn"
    comment_count: int | None = None
    feedback_updated_at: float = 0.0

    @classmethod
    def from_dict(cls, d: StoryDict) -> Story:
        """Create Story from dict (e.g., from cache/API response)."""
        return cls(
            id=int(d.get("id", 0)),
            title=str(d.get("title", "")),
            url=d.get("url"),
            score=int(d.get("score", 0)),
            time=int(d.get("time", 0)),
            discussion_url=d.get("discussion_url"),
            comments=d.get("comments", []),
            text_content=str(d.get("text_content", "")),
            source=cast("StorySource", d.get("source", "hn")),
            comment_count=d.get("comment_count"),
            feedback_updated_at=d.get("feedback_updated_at", 0.0),
        )

    def to_dict(self) -> StoryDict:
        """Serialize to dict for caching."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "score": self.score,
            "time": self.time,
            "discussion_url": self.discussion_url,
            "comments": self.comments,
            "text_content": self.text_content,
            "source": self.source,
            "comment_count": self.comment_count,
            "feedback_updated_at": self.feedback_updated_at,
        }

    @property
    def is_hn(self) -> bool:
        return self.source == "hn"

    @property
    def is_external(self) -> bool:
        return not self.is_hn

    @property
    def badge_label(self) -> str | None:
        return source_badge_label(self.source)


@dataclass
class RankResult:
    """Result of ranking a single candidate story."""

    index: int  # Index in the candidates list
    model_score: float  # Canonical ranking score
    best_fav_index: int  # Index of most similar positive signal (-1 if none)
    max_sim_score: float  # Similarity to best matching positive signal
    knn_score: float  # Mean similarity to top-k neighbors (for display)
    max_cluster_score: float = 0.0
    p_down: float = 0.0
    p_neutral: float = 0.0
    p_up: float = 0.0
    entropy: float = 0.0
    acquisition_kind: str = "exploit"


@dataclass
class StoryDisplay:
    """Story formatted for HTML display."""

    id: int
    match_percent: int
    cluster_name: str
    points: int
    time_ago: str
    time: int
    url: str | None
    title: str
    hn_url: str | None
    reason: str  # Title of matched positive signal
    reason_url: str  # URL to matched positive signal
    comments: list[str]
    source: StorySource = "hn"
    tldr: str = ""
    text_content: str = ""
    rank_index: int = 0
    model_score: float = 0.0
    knn_score: float = 0.0
    max_sim_score: float = 0.0
    max_cluster_score: float = 0.0
    comment_count: int | None = None
    feedback_action: Literal["up", "neutral", "down"] | None = None
    acquisition_kind: str = "exploit"

    def to_dict(self) -> StoryDisplayDict:
        """Convert to dict for template rendering."""
        return {
            "id": self.id,
            "match_percent": self.match_percent,
            "cluster_name": self.cluster_name,
            "points": self.points,
            "time_ago": self.time_ago,
            "time": self.time,
            "url": self.url,
            "title": self.title,
            "hn_url": self.hn_url,
            "reason": self.reason,
            "reason_url": self.reason_url,
            "comments": self.comments,
            "source": self.source,
            "tldr": self.tldr,
            "text_content": self.text_content,
            "rank_index": self.rank_index,
            "model_score": self.model_score,
            "knn_score": self.knn_score,
            "max_sim_score": self.max_sim_score,
            "max_cluster_score": self.max_cluster_score,
            "comment_count": self.comment_count,
            "feedback_action": self.feedback_action,
            "acquisition_kind": self.acquisition_kind,
        }

    @property
    def is_hn(self) -> bool:
        return self.source == "hn"

    @property
    def is_external(self) -> bool:
        return not self.is_hn

    @property
    def badge_label(self) -> str | None:
        return source_badge_label(self.source)
