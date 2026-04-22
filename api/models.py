"""Typed data models for HN reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

StorySource = Literal["hn", "rss", "lobsters", "tildes"]

_SOURCE_BADGE_LABELS: dict[StorySource, str | None] = {
    "hn": None,
    "rss": "RSS",
    "lobsters": "Lobsters",
    "tildes": "Tildes",
}


def source_badge_label(source: StorySource) -> str | None:
    return _SOURCE_BADGE_LABELS[source]


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


class StoryForTldr(TypedDict):
    """Minimum fields needed for TL;DR generation."""

    id: int
    title: str
    comments: list[str]


class StoryDisplayDict(StoryForTldr):
    """Serialized StoryDisplay payload for templates and LLMs."""

    match_percent: int
    cluster_name: str
    points: int
    time_ago: str
    url: str | None
    hn_url: str | None
    reason: str
    reason_url: str
    tldr: str
    source: StorySource


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
            comments=list(d.get("comments", [])),
            text_content=str(d.get("text_content", "")),
            source=d.get("source", "hn"),
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
    hybrid_score: float  # Combined semantic + HN score
    best_fav_index: int  # Index of most similar positive signal (-1 if none)
    max_sim_score: float  # Similarity to best matching positive signal
    knn_score: float  # Mean similarity to top-k neighbors (for display)
    semantic_score: float = 0.0  # Raw semantic score (classifier or k-NN)
    hn_score: float = 0.0  # HN score contribution (log-scaled)
    freshness_boost: float = 0.0  # Freshness boost applied to hybrid score


@dataclass
class StoryDisplay:
    """Story formatted for HTML display."""

    id: int
    match_percent: int
    cluster_name: str
    points: int
    time_ago: str
    url: str | None
    title: str
    hn_url: str | None
    reason: str  # Title of matched positive signal
    reason_url: str  # URL to matched positive signal
    comments: list[str]
    source: StorySource = "hn"
    tldr: str = ""

    def to_dict(self) -> StoryDisplayDict:
        """Convert to dict for template rendering."""
        return {
            "id": self.id,
            "match_percent": self.match_percent,
            "cluster_name": self.cluster_name,
            "points": self.points,
            "time_ago": self.time_ago,
            "url": self.url,
            "title": self.title,
            "hn_url": self.hn_url,
            "reason": self.reason,
            "reason_url": self.reason_url,
            "comments": self.comments,
            "source": self.source,
            "tldr": self.tldr,
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
