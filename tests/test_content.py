from __future__ import annotations

import httpx
import pytest

from api.content import (
    _fallback_extract_text,
    compose_story_text,
    fetch_full_text,
    strip_html,
    truncate_text,
)


def test_strip_html_unescapes_and_normalizes_spacing():
    assert strip_html("<p>Hello&nbsp; <strong>world</strong> !</p>") == "Hello world!"


def test_truncate_text_prefers_word_boundary():
    assert truncate_text("alpha beta gamma delta", 12) == "alpha beta..."


def test_fallback_extract_text_prefers_article_and_drops_chrome():
    html = """
    <html>
      <body>
        <nav>Navigation</nav>
        <article><h1>Title</h1><p>Main text.</p></article>
        <footer>Footer</footer>
      </body>
    </html>
    """
    assert _fallback_extract_text(html) == "Title Main text."


def test_compose_story_text_combines_cleaned_sources():
    text = compose_story_text(
        title="<b>Title</b>",
        self_text="Self&nbsp;text",
        article_text="<p>Article text</p>",
        comments=["First&nbsp;comment", "<i>Second</i> comment"],
    )
    assert text == "Title. Self text Article text First comment Second comment"


@pytest.mark.asyncio
async def test_fetch_full_text_rejects_non_html(tmp_path, monkeypatch):
    monkeypatch.setattr("api.content.CONTENT_CACHE_PATH", tmp_path)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"not html",
            headers={"content-type": "application/json"},
            request=request,
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        assert await fetch_full_text(client, "https://example.com/data.json") == ""
