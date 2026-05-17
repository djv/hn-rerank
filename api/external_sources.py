"""Registry for non-HN story sources."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ExternalSourceSpec:
    """Configuration for detecting and handling an external story source."""

    source: str
    badge_label: str | None
    domains: tuple[str, ...] = ()
    path_prefixes: tuple[str, ...] = ()
    curated: bool = False
    parser: str = "feed"

    def matches_url(self, value: str) -> bool:
        if not value:
            return False
        parsed = urlparse(value)
        host = parsed.netloc.lower()
        if not host:
            return any(domain in value.lower() for domain in self.domains)
        if host not in self.domains:
            return False
        if not self.path_prefixes:
            return True
        path = parsed.path.lower()
        return any(path.startswith(prefix.lower()) for prefix in self.path_prefixes)


SOURCE_SPECS: tuple[ExternalSourceSpec, ...] = (
    ExternalSourceSpec(source="hn", badge_label=None),
    ExternalSourceSpec(source="rss", badge_label="RSS"),
    ExternalSourceSpec(
        source="lobsters",
        badge_label="Lobsters",
        domains=("lobste.rs",),
        curated=True,
    ),
    ExternalSourceSpec(
        source="tildes",
        badge_label="Tildes",
        domains=("tildes.net",),
        curated=True,
    ),
    ExternalSourceSpec(
        source="lesswrong",
        badge_label="LessWrong",
        domains=("www.lesswrong.com", "lesswrong.com"),
        curated=True,
    ),
    ExternalSourceSpec(
        source="reddit_machinelearning",
        badge_label="r/MachineLearning",
        domains=("www.reddit.com", "reddit.com", "old.reddit.com"),
        path_prefixes=("/r/MachineLearning",),
        curated=True,
    ),
    ExternalSourceSpec(
        source="reddit_programming",
        badge_label="r/programming",
        domains=("www.reddit.com", "reddit.com", "old.reddit.com"),
        path_prefixes=("/r/programming",),
        curated=True,
    ),
    ExternalSourceSpec(
        source="reddit_compsci",
        badge_label="r/compsci",
        domains=("www.reddit.com", "reddit.com", "old.reddit.com"),
        path_prefixes=("/r/compsci",),
        curated=True,
    ),
    ExternalSourceSpec(
        source="reddit",
        badge_label="Reddit",
        domains=("www.reddit.com", "reddit.com", "old.reddit.com"),
        path_prefixes=("/r/",),
        curated=True,
    ),
    ExternalSourceSpec(
        source="digg",
        badge_label="Digg",
        domains=("digg.com", "www.digg.com"),
        path_prefixes=("/ai",),
        curated=True,
        parser="digg_ai",
    ),
)

SOURCE_BADGE_LABELS = {spec.source: spec.badge_label for spec in SOURCE_SPECS}
_SOURCE_BY_NAME = {spec.source: spec for spec in SOURCE_SPECS}
_DETECTABLE_SPECS = tuple(
    spec for spec in SOURCE_SPECS if spec.source not in {"hn", "rss"}
)


def source_badge_label(source: str) -> str | None:
    return SOURCE_BADGE_LABELS.get(source, "RSS")


def source_spec(source: str) -> ExternalSourceSpec:
    return _SOURCE_BY_NAME[source]


def detect_source(feed_url: str, link: str = "") -> ExternalSourceSpec:
    for spec in _DETECTABLE_SPECS:
        if spec.matches_url(feed_url) or spec.matches_url(link):
            return spec
    return _SOURCE_BY_NAME["rss"]
