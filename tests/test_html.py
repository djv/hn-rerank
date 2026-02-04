import time
import pytest
from generate_html import generate_story_html, get_relative_time, resolve_cluster_name
from api.models import StoryDisplay


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


def test_resolve_cluster_name_fallback():
    cluster_names = {0: "Systems"}

    assert resolve_cluster_name(cluster_names, 0) == "Systems"
    assert resolve_cluster_name(cluster_names, 1) == "Group 2"
    assert resolve_cluster_name(cluster_names, -1) == ""


def test_generate_story_html_includes_cluster_chip():
    story = StoryDisplay(
        id=789,
        match_percent=72,
        cluster_name=resolve_cluster_name({}, 0),
        points=50,
        time_ago="3h",
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
        url="https://example.com/rss",
        title="RSS Story",
        hn_url=None,
        reason="",
        reason_url="",
        comments=[],
    )
    html = generate_story_html(story)
    assert "RSS Story" in html
    assert "title=\"Comments\"" not in html
