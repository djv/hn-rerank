"""Property-based tests for API data model serialization and invariants."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from api.models import (
    RankResult,
    Story,
    StoryDisplay,
    StorySource,
    source_badge_label,
)


ALL_SOURCES: tuple[StorySource, ...] = (
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
)

source_strategy = st.sampled_from(ALL_SOURCES)

url_strategy = st.one_of(
    st.none(),
    st.from_regex(r"https?://[a-z0-9.-]+\.[a-z]{2,3}(/[\w./-]*)?", fullmatch=True),
)


@st.composite
def stories(draw):
    story_id = draw(st.integers(min_value=-(10**6), max_value=10**9))
    return Story(
        id=story_id,
        title=draw(st.text(min_size=0, max_size=200)),
        url=draw(url_strategy),
        score=draw(st.integers(min_value=-1000, max_value=100_000)),
        time=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        discussion_url=draw(st.one_of(st.none(), url_strategy)),
        comments=draw(st.lists(st.text(max_size=500), max_size=50)),
        text_content=draw(st.text(max_size=10_000)),
        source=draw(source_strategy),
        comment_count=draw(
            st.one_of(st.none(), st.integers(min_value=0, max_value=100_000))
        ),
    )


# ---- 1. Story round-trip preserves all fields ----
@settings(max_examples=200, deadline=None)
@given(s=stories())
def test_story_round_trip_preserves_fields(s: Story) -> None:
    """Story.from_dict(s.to_dict()) == s for any valid Story."""
    d = s.to_dict()
    s2 = Story.from_dict(d)
    assert s == s2


# ---- 2. is_external iff source != hn ----
@settings(max_examples=100, deadline=None)
@given(source=source_strategy)
def test_story_is_external_iff_source_not_hn(source: StorySource) -> None:
    """is_external is True for all sources except 'hn', False for 'hn'."""
    s = Story(id=1, title="x", url=None, score=0, time=0, source=source)
    assert s.is_external == (source != "hn")
    assert s.is_hn == (source == "hn")
    assert s.is_hn != s.is_external or (source == "hn" and not s.is_external)


# ---- 3. badge_label matches source_badge_label ----
@settings(max_examples=100, deadline=None)
@given(source=source_strategy)
def test_story_badge_label_matches_source(source: StorySource) -> None:
    """badge_label matches source_badge_label(source)."""
    s = Story(id=1, title="x", url=None, score=0, time=0, source=source)
    assert s.badge_label == source_badge_label(source)


# ---- 4. StoryDisplay.to_dict has expected keys ----
@settings(max_examples=50, deadline=None)
@given(
    story_id=st.integers(min_value=0, max_value=10**9),
    match_percent=st.integers(min_value=0, max_value=100),
    source=source_strategy,
)
def test_story_display_to_dict_keys(
    story_id: int, match_percent: int, source: StorySource
) -> None:
    """StoryDisplay.to_dict() returns all expected keys with correct types."""
    sd = StoryDisplay(
        id=story_id,
        match_percent=match_percent,
        cluster_name="Topic",
        points=10,
        time_ago="1h",
        time=1_700_000_000,
        url=None,
        title="T",
        hn_url=None,
        reason="R",
        reason_url="https://example.com",
        comments=[],
        source=source,
    )
    d = sd.to_dict()
    required = {
        "id",
        "match_percent",
        "cluster_name",
        "points",
        "time_ago",
        "time",
        "url",
        "title",
        "hn_url",
        "reason",
        "reason_url",
        "comments",
        "source",
        "tldr",
        "text_content",
        "rank_index",
        "model_score",
        "knn_score",
        "max_sim_score",
        "max_cluster_score",
        "comment_count",
        "feedback_action",
        "acquisition_kind",
    }
    assert required <= set(d.keys()), f"missing: {required - set(d.keys())}"
    assert isinstance(d["id"], int)
    assert isinstance(d["title"], str)
    assert isinstance(d["source"], str)


# ---- 5. RankResult construction with arbitrary values ----
@settings(max_examples=100, deadline=None)
@given(
    index=st.integers(min_value=-1, max_value=10**6),
    model_score=st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    best_fav_index=st.integers(min_value=-1, max_value=1000),
    max_sim_score=st.floats(
        min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    knn_score=st.floats(
        min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    p_up=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_rank_result_constructs_with_arbitrary_floats(
    index: int,
    model_score: float,
    best_fav_index: int,
    max_sim_score: float,
    knn_score: float,
    p_up: float,
) -> None:
    """RankResult can be constructed with any sane float/integer values."""
    r = RankResult(
        index=index,
        model_score=model_score,
        best_fav_index=best_fav_index,
        max_sim_score=max_sim_score,
        knn_score=knn_score,
        p_up=p_up,
    )
    assert r.index == index
    assert r.model_score == model_score
    assert r.p_up == p_up
