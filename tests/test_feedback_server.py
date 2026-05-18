import pytest

from scripts.feedback_server import (
    FeedbackHandler,
    _extract_hn_action_path,
    _mirror_hn_action,
)


def test_extract_hn_action_path_uses_authenticated_links():
    html = """
    <a href="vote?id=123&amp;how=up&amp;auth=abc&amp;goto=item%3Fid%3D123">up</a>
    <a href="hide?id=123&amp;auth=abc&amp;goto=item%3Fid%3D123">hide</a>
    """

    assert (
        _extract_hn_action_path(html, 123, "up")
        == "vote?id=123&how=up&auth=abc&goto=item%3Fid%3D123"
    )
    assert (
        _extract_hn_action_path(html, 123, "down")
        == "hide?id=123&auth=abc&goto=item%3Fid%3D123"
    )


def test_feedback_handler_validates_required_payload_fields():
    with pytest.raises(ValueError, match="action"):
        FeedbackHandler._validate_payload(
            {
                "id": 1,
                "source": "hn",
                "title": "Story",
                "url": None,
                "discussion_url": None,
                "action": "invalid",
            }
        )

    with pytest.raises(ValueError, match="id"):
        FeedbackHandler._validate_payload(
            {
                "id": "1",
                "source": "hn",
                "title": "Story",
                "url": None,
                "discussion_url": None,
                "action": "up",
            }
        )


@pytest.mark.asyncio
async def test_feedback_mirror_skips_non_hn_sources():
    status, error = await _mirror_hn_action(
        {
            "id": -1,
            "source": "rss",
            "title": "External",
            "url": "https://example.com",
            "discussion_url": None,
            "action": "up",
        }
    )

    assert status == "none"
    assert error is None
