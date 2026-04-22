import pytest
import respx
from httpx import Response
from pathlib import Path

from api import rss
from api.rss import _extract_opml_feed_urls, _parse_feed_entries, fetch_rss_stories


@pytest.fixture(autouse=True)
def isolate_rss_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "rss-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(rss, "RSS_CACHE_PATH", Path(cache_dir))


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
        <language>en</language>
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
    <feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">
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


def test_parse_feed_entries_allows_supported_feed_language():
    xml = """
    <rss version="2.0">
      <channel>
        <language>fr-FR</language>
        <item>
          <title>Bonjour le monde</title>
          <link>https://example.com/bonjour</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Un article en français sur les langages de programmation et les systèmes distribués.</description>
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
    assert stories[0].title == "Bonjour le monde"



def test_parse_feed_entries_filters_unsupported_feed_language_by_detection():
    xml = """
    <rss version="2.0">
      <channel>
        <title>Nederlandse technologieblog voor ontwikkelaars</title>
        <description>Deze feed bespreekt programmeertalen, softwareontwikkeling, beveiliging en infrastructuur in het Nederlands.</description>
        <item>
          <title>Waarom deze programmeertaal populair blijft bij ontwikkelaars</title>
          <link>https://example.com/nederlands-1</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Dit artikel legt uit waarom ontwikkelaars graag werken met betrouwbare tooling en duidelijke documentatie.</description>
        </item>
        <item>
          <title>Nieuwe release verbetert prestaties van de compiler</title>
          <link>https://example.com/nederlands-2</link>
          <pubDate>Mon, 02 Feb 2026 13:00:00 GMT</pubDate>
          <description>De verbeteringen richten zich op optimalisaties, foutmeldingen en snellere builds voor grote projecten.</description>
        </item>
      </channel>
    </rss>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://voorbeeld.nl/feed.xml",
        max_items=5,
        min_ts=0,
    )
    assert stories == []


def test_parse_feed_entries_preserves_discussion_url_for_curated_sources():
    xml = """
    <rss version="2.0">
      <channel>
        <language>en</language>
        <item>
          <title>Lobsters Post</title>
          <link>https://example.com/post</link>
          <comments>https://lobste.rs/s/example/post</comments>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Lobsters summary</description>
        </item>
      </channel>
    </rss>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://lobste.rs/rss",
        max_items=5,
        min_ts=0,
    )
    assert len(stories) == 1
    story = stories[0]
    assert story.source == "lobsters"
    assert story.discussion_url == "https://lobste.rs/s/example/post"


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


@pytest.mark.asyncio
@respx.mock
async def test_fetch_rss_stories_uses_extra_feeds_when_opml_is_empty(monkeypatch):
    opml_url = "https://example.com/feeds.opml"
    feed_url = "https://lobste.rs/rss"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [feed_url])

    rss_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Lobsters Post</title>
          <link>https://example.com/post</link>
          <comments>https://lobste.rs/s/example/post</comments>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Lobsters summary</description>
        </item>
      </channel>
    </rss>
    """

    respx.get(opml_url).mock(return_value=Response(200, text=""))
    respx.get(feed_url).mock(return_value=Response(200, text=rss_xml))

    stories = await fetch_rss_stories(
        opml_url=opml_url,
        days=3650,
        max_feeds=5,
        per_feed=10,
        fetch_full_content=False,
    )

    assert len(stories) == 1
    assert stories[0].title == "Lobsters Post"
    assert stories[0].url == "https://example.com/post"
    assert stories[0].discussion_url == "https://lobste.rs/s/example/post"



@pytest.mark.asyncio
@respx.mock
async def test_fetch_rss_stories_uses_higher_limit_for_curated_news_sources(monkeypatch):
    opml_url = "https://example.com/feeds.opml"
    feed_url = "https://lobste.rs/rss"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [feed_url])

    items = []
    for i in range(60):
        items.append(
            f"""
            <item>
              <title>Lobsters Post {i}</title>
              <link>https://example.com/post-{i}</link>
              <comments>https://lobste.rs/s/example/post-{i}</comments>
              <pubDate>Mon, 02 Feb 2026 12:{i:02d}:00 GMT</pubDate>
              <description>Lobsters summary {i}</description>
            </item>
            """
        )
    rss_xml = f"""
    <rss version="2.0">
      <channel>
        {''.join(items)}
      </channel>
    </rss>
    """

    respx.get(opml_url).mock(return_value=Response(200, text=""))
    respx.get(feed_url).mock(return_value=Response(200, text=rss_xml))

    stories = await fetch_rss_stories(
        opml_url=opml_url,
        days=3650,
        max_feeds=5,
        per_feed=5,
        fetch_full_content=False,
    )

    assert len(stories) == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    assert stories[0].source == "lobsters"
    assert stories[-1].title == f"Lobsters Post {rss.RSS_CURATED_NEWS_PER_FEED_LIMIT - 1}"
