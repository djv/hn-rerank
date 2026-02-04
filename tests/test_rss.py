import pytest
import respx
from httpx import Response

from api import rss
from api.rss import _extract_opml_feed_urls, _parse_feed_entries, fetch_rss_stories


def test_extract_opml_feed_urls():
    opml = """
    <opml version="2.0">
      <body>
        <outline text="Blogs" title="Blogs">
          <outline type="rss" text="Example" xmlUrl="https://example.com/feed.xml"/>
          <outline type="rss" text="Other" xmlUrl="https://other.com/rss"/>
        </outline>
      </body>
    </opml>
    """
    urls = _extract_opml_feed_urls(opml)
    assert urls == ["https://example.com/feed.xml", "https://other.com/rss"]


def test_parse_rss_entries():
    xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Post One</title>
          <link>https://example.com/post-1</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description><![CDATA[<p>Hello <b>world</b>.</p>]]></description>
        </item>
      </channel>
    </rss>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://example.com/feed.xml",
        max_items=5,
        min_ts=0,
    )
    assert len(stories) == 1
    story = stories[0]
    assert story.title == "Post One"
    assert story.url == "https://example.com/post-1"
    assert story.time > 0
    assert "Post One" in story.text_content
    assert "Hello world." in story.text_content


def test_parse_rss_prefers_content_encoded():
    xml = """
    <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
      <channel>
        <item>
          <title>Encoded Post</title>
          <link>https://example.com/post-encoded</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Short description</description>
          <content:encoded><![CDATA[<p>Longer <b>encoded</b> content.</p>]]></content:encoded>
        </item>
      </channel>
    </rss>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://example.com/feed.xml",
        max_items=5,
        min_ts=0,
    )
    assert len(stories) == 1
    story = stories[0]
    assert "Longer encoded content." in story.text_content


def test_parse_atom_entries():
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Atom Post</title>
        <link rel="alternate" href="https://example.com/atom" />
        <updated>2026-02-01T12:34:56Z</updated>
        <summary>Atom summary</summary>
      </entry>
    </feed>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://example.com/atom.xml",
        max_items=5,
        min_ts=0,
    )
    assert len(stories) == 1
    story = stories[0]
    assert story.title == "Atom Post"
    assert story.url == "https://example.com/atom"
    assert story.time > 0
    assert "Atom summary" in story.text_content


@pytest.mark.asyncio
@respx.mock
async def test_fetch_rss_stories_filters_old_and_excluded_urls(monkeypatch):
    opml_url = "https://example.com/feeds.opml"
    feed_url = "https://example.com/feed.xml"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [])

    opml = f"""
    <opml version="2.0">
      <body>
        <outline text="Blogs" title="Blogs">
          <outline type="rss" text="Example" xmlUrl="{feed_url}"/>
        </outline>
      </body>
    </opml>
    """

    rss_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Recent Post</title>
          <link>https://example.com/recent</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Recent summary</description>
        </item>
        <item>
          <title>Old Post</title>
          <link>https://example.com/old</link>
          <pubDate>Mon, 02 Feb 2020 12:00:00 GMT</pubDate>
          <description>Old summary</description>
        </item>
      </channel>
    </rss>
    """

    respx.get(opml_url).mock(return_value=Response(200, text=opml))
    respx.get(feed_url).mock(return_value=Response(200, text=rss_xml))

    stories = await fetch_rss_stories(
        opml_url=opml_url,
        days=7,
        max_feeds=5,
        per_feed=10,
        exclude_urls={"https://example.com/recent"},
        fetch_full_content=False,
    )

    assert stories == []
