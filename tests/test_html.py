import pytest
from generate_html import generate_story_html

def test_generate_story_html_special_chars():
    """
    Ensure that story titles or comments containing curly braces don't crash the generator.
    This validates the robustness of the string formatting logic.
    """
    story = {
        "match_percent": 95,
        "points": 100,
        "url": "https://example.com",
        "title": "React {hooks} vs [brackets]",
        "hn_url": "https://news.ycombinator.com/item?id=123",
        "reason": "Interest in {technology}",
        "comments": ["This is a {comment}"]
    }
    
    # This should not raise a KeyError or ValueError
    try:
        html = generate_story_html(story)
        assert "React {hooks}" in html
        assert "Interest in {technology}" in html
        assert "This is a {comment}" in html
    except (KeyError, ValueError) as e:
        pytest.fail(f"generate_story_html crashed with special characters: {e}")

def test_generate_story_html_missing_fields():
    """Test handling of stories with missing or None fields."""
    story = {
        "match_percent": 50,
        "points": 10,
        "url": None, # Missing URL
        "title": "Untitled",
        "hn_url": "https://news.ycombinator.com/item?id=456",
        "reason": "", # Empty reason
        "comments": []
    }
    html = generate_story_html(story)
    assert "Untitled" in html
    assert "https://news.ycombinator.com/item?id=456" in html
    # Should use HN URL as primary link if URL is missing
    assert 'href="https://news.ycombinator.com/item?id=456"' in html
