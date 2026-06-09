import time

import httpx
import numpy as np
import pytest
import respx

import generate_html
from generate_html import (
    _INDEX_TEMPLATE,
    apply_feedback_signal_overrides,
    build_candidate_cluster_map,
    generate_story_html,
    get_cluster_id_for_result,
    get_relative_time,
    refresh_hn_story_metadata,
    resolve_cluster_name,
    select_ranked_results,
    split_feedback_records,
)
from api.feedback import FeedbackRecord
from api.models import RankResult, Story, StoryDisplay


def test_generate_story_html_special_chars():
    """
    Ensure that story titles or comments containing curly braces don't crash the generator.
    This validates the robustness of the string formatting logic.
    """
    story = StoryDisplay(
        id=123,
        match_percent=95,
        cluster_name="",
        points=100,
        time_ago="2h",
        time=123,
        url="https://example.com",
        title="React {hooks} vs [brackets]",
        hn_url="https://news.ycombinator.com/item?id=123",
        reason="Interest in {technology}",
        reason_url="",
        comments=["This is a {comment}"],
    )

    # This should not raise a KeyError or ValueError
    try:
        html = generate_story_html(story)
        assert "React {hooks}" in html
        # Comments are no longer displayed directly; TL;DR replaces them
    except (KeyError, ValueError) as e:
        pytest.fail(f"generate_story_html crashed with special characters: {e}")


def test_generate_story_html_missing_fields():
    """Test handling of stories with missing or None fields."""
    story = StoryDisplay(
        id=456,
        match_percent=50,
        cluster_name="",
        points=10,
        time_ago="1d",
        time=456,
        url=None,  # Missing URL
        title="Untitled",
        hn_url="https://news.ycombinator.com/item?id=456",
        reason="",  # Empty reason
        reason_url="",
        comments=[],
    )
    html = generate_story_html(story)
    assert "Untitled" in html
    assert "https://news.ycombinator.com/item?id=456" in html
    # Should use HN URL as primary link if URL is missing
    assert 'href="https://news.ycombinator.com/item?id=456"' in html


def test_generate_story_html_includes_comment_count_after_icon():
    story = StoryDisplay(
        id=457,
        match_percent=81,
        cluster_name="",
        points=90,
        time_ago="3h",
        time=457,
        url="https://example.com/story",
        title="Story with discussion",
        hn_url="https://news.ycombinator.com/item?id=457",
        reason="",
        reason_url="",
        comments=[],
        comment_count=42,
    )

    html = generate_story_html(story)

    assert 'aria-label="Open comments for Story with discussion"' in html
    assert 'title="Comments">💬 42</a>' in html


def test_generate_story_html_includes_feedback_controls_and_metadata():
    story = StoryDisplay(
        id=123,
        match_percent=90,
        cluster_name="",
        points=42,
        time_ago="1h",
        time=1700000000,
        url="https://example.com/story",
        title="Feedback story",
        hn_url="https://news.ycombinator.com/item?id=123",
        reason="",
        reason_url="",
        comments=[],
        text_content="Feedback story text",
        model_score=0.91,
        knn_score=0.65,
        max_sim_score=0.77,
        max_cluster_score=0.9,
        feedback_action="up",
    )

    html = generate_story_html(story)

    assert 'data-feedback-button="up"' in html
    assert 'data-feedback-button="neutral"' in html
    assert 'data-feedback-button="down"' in html
    assert 'data-story-id="123"' in html
    assert 'data-story-source="hn"' in html
    assert 'data-story-score="42"' in html
    assert 'data-story-comment-count=""' in html
    assert 'data-feedback-action="up"' in html


def test_generate_story_html_hides_unknown_comment_count():
    story = StoryDisplay(
        id=458,
        match_percent=81,
        cluster_name="",
        points=90,
        time_ago="3h",
        time=458,
        url="https://example.com/story",
        title="Story with unknown discussion count",
        hn_url="https://news.ycombinator.com/item?id=458",
        reason="",
        reason_url="",
        comments=[],
    )

    html = generate_story_html(story)

    assert 'title="Comments">💬</a>' in html
    assert "💬 0" not in html


def test_generate_story_html_shows_external_count_without_discussion_link():
    story = StoryDisplay(
        id=-459,
        match_percent=81,
        cluster_name="",
        points=0,
        time_ago="3h",
        time=459,
        url="https://digg.com/ai/story",
        title="External story with count",
        hn_url=None,
        reason="",
        reason_url="",
        comments=[],
        source="digg",
        comment_count=31,
    )

    html = generate_story_html(story)

    assert (
        '<span class="text-[10px] text-stone-400 font-mono shrink-0" title="Comments">💬 31</span>'
        in html
    )
    assert 'title="Comments">💬 31</a>' not in html


def test_resolve_cluster_name_fallback():
    cluster_names = {0: "Systems"}

    assert resolve_cluster_name(cluster_names, 0) == "Systems"
    assert resolve_cluster_name(cluster_names, 1) == "Group 2"
    assert resolve_cluster_name(cluster_names, -1) == ""


def test_resolve_cluster_name_empty_name_fallback_for_rss():
    cluster_names = {2: ""}

    assert resolve_cluster_name(cluster_names, 2) == ""
    assert (
        resolve_cluster_name(cluster_names, 2, allow_empty_fallback=True) == "Group 3"
    )


def test_generate_story_html_includes_cluster_chip():
    story = StoryDisplay(
        id=789,
        match_percent=72,
        cluster_name=resolve_cluster_name({}, 0),
        points=50,
        time_ago="3h",
        time=789,
        url="https://example.com",
        title="Clustered story",
        hn_url="https://news.ycombinator.com/item?id=789",
        reason="",
        reason_url="",
        comments=[],
    )
    html = generate_story_html(story)

    assert "cluster-chip" in html
    assert "Group 1" in html


class TestRelativeTime:
    """Edge case tests for get_relative_time function."""

    def test_zero_timestamp(self):
        assert get_relative_time(0) == ""

    def test_now(self):
        # Timestamp = now should return "now"
        assert get_relative_time(int(time.time())) == "now"

    def test_seconds(self):
        # < 60 seconds ago
        assert get_relative_time(int(time.time()) - 30) == "now"

    def test_minutes(self):
        # 1 minute ago
        assert get_relative_time(int(time.time()) - 60) == "1m"
        # 59 minutes ago
        assert get_relative_time(int(time.time()) - 3599) == "59m"

    def test_hours(self):
        # 1 hour ago
        assert get_relative_time(int(time.time()) - 3600) == "1h"
        # 23 hours ago
        assert get_relative_time(int(time.time()) - 86399) == "23h"

    def test_days(self):
        # 1 day ago
        assert get_relative_time(int(time.time()) - 86400) == "1d"
        # 30 days ago
        assert get_relative_time(int(time.time()) - 86400 * 30) == "30d"


class TestHtmlEscaping:
    """Tests for HTML injection prevention."""

    def test_title_escaped(self):
        story = StoryDisplay(
            id=1,
            match_percent=80,
            cluster_name="",
            points=50,
            time_ago="1h",
            time=1,
            url="https://example.com",
            title="<script>alert('xss')</script>",
            hn_url="https://news.ycombinator.com/item?id=1",
            reason="",
            reason_url="",
            comments=[],
        )
        html = generate_story_html(story)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_tldr_escaped(self):
        """TL;DR content should be HTML-escaped."""
        story = StoryDisplay(
            id=1,
            match_percent=80,
            cluster_name="",
            points=50,
            time_ago="1h",
            time=1,
            url="https://example.com",
            title="Safe Title",
            hn_url="https://news.ycombinator.com/item?id=1",
            reason="",
            reason_url="",
            comments=[],
            tldr="<script>alert('xss')</script>",
        )
        html = generate_story_html(story)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_reason_not_rendered(self):
        story = StoryDisplay(
            id=1,
            match_percent=80,
            cluster_name="",
            points=50,
            time_ago="1h",
            time=1,
            url="https://example.com",
            title="Safe Title",
            hn_url="https://news.ycombinator.com/item?id=1",
            reason="<b>Bold</b> reason",
            reason_url="",
            comments=[],
        )
        html = generate_story_html(story)
        assert "<b>Bold</b> reason" not in html
        assert "&lt;b&gt;Bold&lt;/b&gt; reason" not in html


def test_generate_story_html_without_hn_url_hides_comment_link():
    story = StoryDisplay(
        id=-1,
        match_percent=60,
        cluster_name="",
        points=0,
        time_ago="1h",
        time=1,
        url="https://example.com/rss",
        title="RSS Story",
        hn_url=None,
        reason="",
        reason_url="",
        comments=[],
        source="rss",
    )
    html = generate_story_html(story)
    assert "RSS Story" in html
    assert 'title="Comments"' not in html
    assert 'aria-label="Open comments for RSS Story"' not in html
    assert 'aria-label="Open story for RSS Story"' in html
    assert 'href="https://example.com/rss"' in html


def test_generate_story_html_rss_badge():
    story = StoryDisplay(
        id=-42,
        match_percent=80,
        cluster_name="",
        points=0,
        time_ago="1h",
        time=42,
        url="https://example.com/rss",
        title="RSS Story",
        hn_url=None,
        reason="",
        reason_url="",
        comments=[],
        source="rss",
    )
    html = generate_story_html(story)
    assert "RSS" in html


def test_generate_story_html_lesswrong_badge():
    story = StoryDisplay(
        id=-43,
        match_percent=80,
        cluster_name="",
        points=0,
        time_ago="1h",
        time=43,
        url="https://www.lesswrong.com/posts/abc/example",
        title="LessWrong Story",
        hn_url=None,
        reason="",
        reason_url="",
        comments=[],
        source="lesswrong",
    )
    html = generate_story_html(story)
    assert "LessWrong" in html


def test_generate_story_html_reddit_badge():
    story = StoryDisplay(
        id=-44,
        match_percent=80,
        cluster_name="",
        points=0,
        time_ago="1h",
        time=44,
        url="https://arxiv.org/abs/2604.21691",
        title="Reddit ML Story",
        hn_url="https://www.reddit.com/r/MachineLearning/comments/1sun588/post/",
        reason="",
        reason_url="",
        comments=[],
        source="reddit_machinelearning",
    )
    html = generate_story_html(story)
    assert "r/MachineLearning" in html
    assert 'href="https://arxiv.org/abs/2604.21691"' in html
    assert (
        'href="https://www.reddit.com/r/MachineLearning/comments/1sun588/post/"' in html
    )


def test_generate_story_html_external_comments_link():
    story = StoryDisplay(
        id=-7,
        match_percent=80,
        cluster_name="",
        points=0,
        time_ago="1h",
        time=7,
        url="https://example.com/article",
        title="Lobsters Story",
        hn_url="https://lobste.rs/s/example/post",
        reason="",
        reason_url="",
        comments=[],
        source="lobsters",
    )
    html = generate_story_html(story)
    assert 'href="https://example.com/article"' in html
    assert 'href="https://lobste.rs/s/example/post"' in html
    assert 'aria-label="Open comments for Lobsters Story"' in html
    assert 'title="Comments"' in html
    assert "Lobsters" in html


def test_generate_story_html_makes_card_clickable_to_comments():
    story = StoryDisplay(
        id=901,
        match_percent=80,
        cluster_name="",
        points=12,
        time_ago="2h",
        time=1710000000,
        url="https://example.com/story",
        title="Clickable card story",
        hn_url="https://news.ycombinator.com/item?id=901",
        reason="",
        reason_url="",
        comments=[],
    )

    html = generate_story_html(story)

    assert 'class="absolute inset-0 z-10 rounded-lg"' in html
    assert 'href="https://news.ycombinator.com/item?id=901"' in html
    assert 'aria-label="Open comments for Clickable card story"' in html


def test_generate_story_html_includes_sort_metadata():
    story = StoryDisplay(
        id=900,
        match_percent=80,
        cluster_name="",
        points=12,
        time_ago="2h",
        time=1710000000,
        url="https://example.com/story",
        title="Sortable story",
        hn_url="https://news.ycombinator.com/item?id=900",
        reason="",
        reason_url="",
        comments=[],
        rank_index=3,
    )

    html = generate_story_html(story)

    assert 'data-rank-index="3"' in html
    assert 'data-story-time="1710000000"' in html


def test_index_template_includes_sort_control_and_defaults_to_similarity():
    html = _INDEX_TEMPLATE.render(
        username="alice",
        n_clusters=4,
        timestamp="2026-05-05 00:00:00",
        stories_html="",
    )

    assert 'id="sort-mode"' in html
    assert '<option value="current" selected>Similarity</option>' in html
    assert '<option value="date">Date</option>' in html
    assert "renderSort(sortMode.value);" in html


def test_index_template_removes_feedback_cards_after_vote():
    html = _INDEX_TEMPLATE.render(
        username="test",
        n_clusters=1,
        timestamp="now",
        stories_html="",
    )

    assert "feedback-removing" in html
    assert "const removeCard = (card) =>" in html
    assert "window.setTimeout(() => card.remove(), 220);" in html
    assert "if (nextAction !== 'clear')" in html


def test_index_template_hides_previously_acted_feedback_cards_on_load():
    html = _INDEX_TEMPLATE.render(
        username="test",
        n_clusters=1,
        timestamp="now",
        stories_html="",
    )

    assert "const ACTED_KEYS_KEY = 'hnRerankActedFeedbackKeys';" in html
    assert "const hidePreviouslyActedCards = () =>" in html
    assert "action === 'up' || action === 'neutral' || action === 'down'" in html
    assert "actedKeys.has(card.dataset.feedbackKey)" in html
    assert "hidePreviouslyActedCards();" in html


def test_index_template_persists_successful_feedback_keys_locally():
    html = _INDEX_TEMPLATE.render(
        username="test",
        n_clusters=1,
        timestamp="now",
        stories_html="",
    )

    assert "const rememberActedKey = (key) =>" in html
    assert "const forgetActedKey = (key) =>" in html
    assert "rememberActedKey(card.dataset.feedbackKey);" in html
    assert "forgetActedKey(card.dataset.feedbackKey);" in html
    assert "localStorage.setItem(ACTED_KEYS_KEY" in html


def test_index_template_syncs_server_feedback_without_prompting_on_load():
    html = _INDEX_TEMPLATE.render(
        username="test",
        n_clusters=1,
        config_hash="abc123",
        timestamp="now",
        stories_html="",
    )

    assert "const syncServerFeedback = async () =>" in html
    assert "const token = localStorage.getItem(TOKEN_KEY);" in html
    assert "method: 'GET'" in html
    assert "'X-HN-RERANK-FEEDBACK-TOKEN': token" in html
    assert "for (const [key, record] of Object.entries(payload.records))" in html
    assert "['up', 'neutral', 'down'].includes(record.action)" in html
    assert (
        "syncServerFeedback().then(() => setupIntersectionObserverImpressions());"
        in html
    )
    assert "setupIntersectionObserverImpressions" in html
    assert "IMPRESSIONS_URL" in html
    assert "CONFIG_HASH" in html


def test_build_candidate_cluster_map_respects_threshold_for_external(monkeypatch):
    cands = [
        Story(
            id=-1,
            title="RSS Story",
            url="https://example.com/rss",
            score=0,
            time=1,
            text_content="rss content",
            source="rss",
        ),
        Story(
            id=123,
            title="HN Story",
            url="https://example.com/hn",
            score=10,
            time=1,
            text_content="hn content",
        ),
    ]
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(
        generate_html.rerank,
        "get_embeddings",
        lambda _texts, **_kwargs: np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )

    cluster_map = build_candidate_cluster_map(
        cands,
        centroids,
        threshold=1.01,
    )

    assert cluster_map[0] == -1
    assert cluster_map[1] == -1


def _mk_rank(
    idx: int,
    score: float,
) -> RankResult:
    return RankResult(
        index=idx,
        model_score=score,
        best_fav_index=-1,
        max_sim_score=0.0,
        knn_score=0.0,
    )


@pytest.mark.asyncio
@respx.mock
async def test_refresh_hn_story_metadata_updates_comment_count_and_score():
    story = Story(
        id=48179130,
        title="Cached HN story",
        url="https://example.com/story",
        score=10,
        time=1,
        text_content="cached story",
        source="hn",
        comment_count=3,
    )
    progress: list[tuple[int, int]] = []
    respx.get("https://hacker-news.firebaseio.com/v0/item/48179130.json").mock(
        return_value=httpx.Response(
            200,
            json={"id": 48179130, "score": 44, "descendants": 17},
        )
    )

    await refresh_hn_story_metadata(
        stories=[story],
        progress_callback=lambda curr, total: progress.append((curr, total)),
    )

    assert story.comment_count == 17
    assert story.score == 44
    assert progress == [(1, 1)]


@pytest.mark.parametrize(
    ("num_ext", "num_hn", "count", "expected_ext"),
    [
        (6, 4, 6, 6),  # enough external to fill quota
        (8, 2, 6, 6),  # external floor dominates
        (1, 9, 6, 1),  # limited external, rest HN
    ],
)
def test_select_ranked_results_external_quota_scenarios(
    num_ext: int, num_hn: int, count: int, expected_ext: int
):
    cands = [
        Story(
            id=-(i + 1),
            title=f"RSS {i}",
            url=None,
            score=0,
            time=1,
            text_content="",
            source="rss",
        )
        if i < num_ext
        else Story(
            id=i - num_ext + 1,
            title=f"HN {i - num_ext}",
            url=None,
            score=0,
            time=1,
            text_content="",
        )
        for i in range(num_ext + num_hn)
    ]
    ranked = [_mk_rank(i, 1.0 - (i * 0.01)) for i in range(len(cands))]

    selected = select_ranked_results(
        ranked,
        cands,
        cluster_labels=None,
        cluster_names={},
        cand_cluster_map={},
        count=count,
    )

    ext_count = sum(1 for r in selected if cands[r.index].is_external)
    hn_count = len(selected) - ext_count
    assert len(selected) == count
    assert ext_count == expected_ext
    assert hn_count == count - expected_ext


def test_select_ranked_results_no_longer_prioritizes_cluster_coverage():
    cands = [
        Story(id=1, title="HN 0", url=None, score=0, time=1, text_content=""),
        Story(id=2, title="HN 1", url=None, score=0, time=1, text_content=""),
        Story(id=3, title="HN 2", url=None, score=0, time=1, text_content=""),
    ]
    ranked = [_mk_rank(0, 0.99), _mk_rank(1, 0.98), _mk_rank(2, 0.97)]

    selected = select_ranked_results(
        ranked,
        cands,
        cluster_labels=np.array([0, 1], dtype=np.int32),
        cluster_names={0: "Alpha", 1: "Beta"},
        cand_cluster_map={0: 0, 1: 0, 2: 1},
        count=2,
    )

    assert [result.index for result in selected] == [0, 1]


def test_select_ranked_results_uses_model_scores():
    cands = [
        Story(id=1, title="HN 0", url=None, score=0, time=1, text_content=""),
        Story(id=2, title="HN 1", url=None, score=0, time=1, text_content=""),
    ]
    ranked = [
        _mk_rank(0, 0.9),
        _mk_rank(1, 0.8),
    ]

    selected = select_ranked_results(
        ranked,
        cands,
        cluster_labels=None,
        cluster_names={},
        cand_cluster_map={},
        count=2,
    )

    assert [result.index for result in selected] == [0, 1]


def test_split_feedback_records_builds_signals_and_exclusions():
    records = {
        "hn:1": FeedbackRecord(
            key="hn:1",
            action="up",
            id=1,
            source="hn",
            title="HN up",
            url=None,
            discussion_url="https://news.ycombinator.com/item?id=1",
            text_content="HN up",
            time=1700000000,
        ),
        "https://example.com/neutral": FeedbackRecord(
            key="https://example.com/neutral",
            action="neutral",
            id=-3,
            source="rss",
            title="External neutral",
            url="https://example.com/neutral",
            discussion_url=None,
            text_content="External neutral",
            time=1700000000,
        ),
        "https://example.com/down": FeedbackRecord(
            key="https://example.com/down",
            action="down",
            id=-2,
            source="rss",
            title="External down",
            url="https://example.com/down",
            discussion_url=None,
            text_content="External down",
            time=1700000000,
        ),
    }

    positive, negative, hn_ids, urls = split_feedback_records(records)

    assert [story.title for story in positive] == ["HN up"]
    assert [story.title for story in negative] == ["External down"]
    assert hn_ids == {1}
    assert urls == {"https://example.com/down", "https://example.com/neutral"}


def test_apply_feedback_signal_overrides_preserves_hn_and_overrides_conflicts():
    data = {
        "pos": {1, 2},
        "upvoted": {1},
        "hidden": {3, 4},
        "hidden_urls": set(),
        "favorites": {2},
        "favorites_urls": set(),
        "upvoted_urls": set(),
    }
    feedback_positive = [
        Story(id=3, title="Hidden but liked", url=None, score=0, time=1, source="hn")
    ]
    feedback_negative = [
        Story(
            id=2, title="Favorite but disliked", url=None, score=0, time=1, source="hn"
        )
    ]

    pos_ids, neg_ids = apply_feedback_signal_overrides(
        data,
        feedback_positive,
        feedback_negative,
        signal_limit=10,
        use_hidden_signal=True,
    )

    assert set(pos_ids) == {1, 3}
    assert set(neg_ids) == {2, 4}


def test_apply_feedback_signal_overrides_prioritizes_dashboard_votes_at_limit():
    data = {
        "pos": {1, 2, 3},
        "upvoted": {1},
        "hidden": {4, 5, 6},
        "hidden_urls": set(),
        "favorites": {2, 3},
        "favorites_urls": set(),
        "upvoted_urls": set(),
    }
    feedback_positive = [
        Story(id=99, title="Dashboard up", url=None, score=0, time=1, source="hn")
    ]
    feedback_negative = [
        Story(id=98, title="Dashboard down", url=None, score=0, time=1, source="hn")
    ]

    pos_ids, neg_ids = apply_feedback_signal_overrides(
        data,
        feedback_positive,
        feedback_negative,
        signal_limit=1,
        use_hidden_signal=True,
    )

    assert pos_ids == [99]
    assert neg_ids == [98]


def test_get_cluster_id_prefers_candidate_assignment():
    result = RankResult(
        index=0,
        model_score=1.0,
        best_fav_index=0,
        max_sim_score=0.99,
        knn_score=0.99,
    )
    cluster_labels = np.array([3], dtype=np.int32)
    cand_cluster_map = {0: 7}

    cid = get_cluster_id_for_result(result, cluster_labels, cand_cluster_map, 0.85)
    assert cid == 7


def test_get_cluster_id_falls_back_to_best_fav_when_candidate_unassigned():
    result = RankResult(
        index=0,
        model_score=1.0,
        best_fav_index=0,
        max_sim_score=0.99,
        knn_score=0.99,
    )
    cluster_labels = np.array([3], dtype=np.int32)
    cand_cluster_map = {0: -1}

    cid = get_cluster_id_for_result(result, cluster_labels, cand_cluster_map, 0.85)
    assert cid == 3
