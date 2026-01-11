import pytest
import re
from generate_html import generate_story_html

def test_story_card_accessibility_improvements():
    """
    Verify UX and accessibility improvements in the Story Card HTML.
    Checks for:
    - High contrast text color on HN button.
    - ARIA labels for icon-only buttons.
    - Security attributes (noopener noreferrer) on external links.
    - Improved touch targets (larger padding).
    """
    story = {
        "match_percent": 90,
        "points": 150,
        "time_ago": "3h",
        "url": "https://example.com/story",
        "title": "Accessible Design Patterns",
        "hn_url": "https://news.ycombinator.com/item?id=999",
        "reason": "UX Interests",
        "comments": [],
    }

    html = generate_story_html(story)

    # 1. Check for improved contrast on the HN button icon
    # Extract the class attribute of the HN link
    # The link has title="HN"
    hn_link_match = re.search(r'<a [^>]*title="HN"[^>]*>', html)
    assert hn_link_match, "HN link with title='HN' not found"
    hn_link_tag = hn_link_match.group(0)

    assert "text-stone-600" in hn_link_tag, "HN button should use higher contrast text color (text-stone-600)"
    assert "text-stone-400" not in hn_link_tag, "HN button should not use low contrast text-stone-400"

    # 2. Check for ARIA label on the HN button
    assert 'aria-label="View on Hacker News"' in hn_link_tag, "HN button missing aria-label"

    # 3. Check for rel="noopener noreferrer" on external links
    # Check the story title link
    # Find the link that contains the title
    title_link_match = re.search(r'<a [^>]*>Accessible Design Patterns</a>', html)
    assert title_link_match, "Story title link not found"
    title_link_tag = title_link_match.group(0)

    assert 'rel="noopener noreferrer"' in title_link_tag, "Story title link should have rel='noopener noreferrer'"
    assert 'href="https://example.com/story"' in title_link_tag

    # Check the HN link also has it
    assert 'rel="noopener noreferrer"' in hn_link_tag, "HN link should have rel='noopener noreferrer'"

    # 4. Check for larger touch target padding on HN link
    assert "p-2" in hn_link_tag, "HN button should have larger padding (p-2) for touch accessibility"
