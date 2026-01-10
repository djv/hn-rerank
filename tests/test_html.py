import time
import pytest
from generate_html import generate_story_html, get_relative_time


def test_generate_story_html_special_chars():
    """
    Ensure that story titles or comments containing curly braces don't crash the generator.
    This validates the robustness of the string formatting logic.
    """
    story = {
        "match_percent": 95,
        "points": 100,
        "time_ago": "2h",
        "url": "https://example.com",
        "title": "React {hooks} vs [brackets]",
        "hn_url": "https://news.ycombinator.com/item?id=123",
        "reason": "Interest in {technology}",
        "comments": ["This is a {comment}"],
    }

    # This should not raise a KeyError or ValueError
    try:
        html = generate_story_html(story)
        assert "React {hooks}" in html
        assert "Interest in {technology}" in html
        # Comments are no longer displayed directly; TL;DR replaces them
    except (KeyError, ValueError) as e:
        pytest.fail(f"generate_story_html crashed with special characters: {e}")


def test_generate_story_html_missing_fields():
    """Test handling of stories with missing or None fields."""
    story = {
        "match_percent": 50,
        "points": 10,
        "time_ago": "1d",
        "url": None,  # Missing URL
        "title": "Untitled",
        "hn_url": "https://news.ycombinator.com/item?id=456",
        "reason": "",  # Empty reason
        "comments": [],
    }
    html = generate_story_html(story)
    assert "Untitled" in html
    assert "https://news.ycombinator.com/item?id=456" in html
    # Should use HN URL as primary link if URL is missing
    assert 'href="https://news.ycombinator.com/item?id=456"' in html


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
        story = {
            "match_percent": 80,
            "points": 50,
            "time_ago": "1h",
            "url": "https://example.com",
            "title": "<script>alert('xss')</script>",
            "hn_url": "https://news.ycombinator.com/item?id=1",
            "reason": None,
            "comments": [],
        }
        html = generate_story_html(story)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_tldr_escaped(self):
        """TL;DR content should be HTML-escaped."""
        story = {
            "match_percent": 80,
            "points": 50,
            "time_ago": "1h",
            "url": "https://example.com",
            "title": "Safe Title",
            "hn_url": "https://news.ycombinator.com/item?id=1",
            "reason": None,
            "comments": [],
            "tldr": "<script>alert('xss')</script>",
        }
        html = generate_story_html(story)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_reason_escaped(self):
        story = {
            "match_percent": 80,
            "points": 50,
            "time_ago": "1h",
            "url": "https://example.com",
            "title": "Safe Title",
            "hn_url": "https://news.ycombinator.com/item?id=1",
            "reason": "<b>Bold</b> reason",
            "comments": [],
        }
        html = generate_story_html(story)
        assert "<b>" not in html
        assert "&lt;b&gt;" in html
