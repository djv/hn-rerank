"""Typed data models for HN reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Story:
    """A Hacker News story with extracted content."""

    id: int
    title: str
    url: Optional[str]
    score: int
    time: int
    comments: list[str] = field(default_factory=list)
    text_content: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Story:
        """Create Story from dict (e.g., from cache/API response)."""
        return cls(
            id=int(d.get("id", 0)),
            title=str(d.get("title", "")),
            url=d.get("url"),
            score=int(d.get("score", 0)),
            time=int(d.get("time", 0)),
            comments=list(d.get("comments", [])),
            text_content=str(d.get("text_content", "")),
        )

    def to_dict(self) -> dict:
        """Serialize to dict for caching."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "score": self.score,
            "time": self.time,
            "comments": self.comments,
            "text_content": self.text_content,
        }


@dataclass
class RankResult:
    """Result of ranking a single candidate story."""

    index: int  # Index in the candidates list
    hybrid_score: float  # Combined semantic + HN score
    best_fav_index: int  # Index of most similar positive signal (-1 if none)
    max_sim_score: float  # Similarity to best matching positive signal
    knn_score: float  # Mean similarity to top-k neighbors (for display)


@dataclass
class StoryDisplay:
    """Story formatted for HTML display."""

    id: int
    match_percent: int
    cluster_name: str
    points: int
    time_ago: str
    url: Optional[str]
    title: str
    hn_url: str
    reason: str  # Title of matched positive signal
    reason_url: str  # URL to matched positive signal
    comments: list[str]
    tldr: str = ""

    def to_dict(self) -> dict:
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
            "tldr": self.tldr,
        }
