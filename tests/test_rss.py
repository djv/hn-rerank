import logging
import pytest
import respx
from httpx import Response
from pathlib import Path

from api import rss
from api.rss import (
    _extract_opml_feed_urls,
    _feed_item_limit,
    _parse_date,
    _parse_digg_ai_page,
    _parse_feed_entries,
    fetch_rss_stories,
)


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


def test_parse_date_logs_debug_on_unparseable_input(caplog):
    with caplog.at_level(logging.DEBUG):
        parsed = _parse_date("definitely not a date")

    assert parsed is None
    assert "Failed to parse RSS date" in caplog.text


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


def test_parse_feed_entries_skips_undated_items():
    xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Undated essay</title>
          <link>https://example.com/essay</link>
          <description>No date metadata here</description>
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
    assert stories == []


def test_parse_digg_ai_page_extracts_embedded_stories():
    html = r"""
    <script>self.__next_f.push([1,"{\"id\":\"b2f851a5-0a26-459e-ac2e-35bb611ba02a\",\"shortId\":\"svp61dsd\",\"title\":\"OpenAI launches ChatGPT personal finance preview\",\"tldr\":\"OpenAI is testing personal finance features for ChatGPT Pro users.\",\"createdAt\":\"2026-05-15T16:12:17.887072+00:00\",\"firstPostAt\":\"2026-05-15T16:01:19+00:00\",\"lastFrozenPostAt\":\"2026-05-16T03:55:22+00:00\",\"postCount\":31}"])</script>
    """

    stories = _parse_digg_ai_page(
        html,
        feed_url="https://digg.com/ai",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    story = stories[0]
    assert story.source == "digg"
    assert story.badge_label == "Digg"
    assert story.url == "https://digg.com/ai/b2f851a5-0a26-459e-ac2e-35bb611ba02a"
    assert story.score == 0
    assert story.comment_count == 31
    assert story.time == 1778861537
    assert "OpenAI launches ChatGPT personal finance preview" in story.text_content
    assert "personal finance features" in story.text_content


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


def test_parse_feed_entries_marks_lesswrong_source():
    xml = """
    <rss version="2.0">
      <channel>
        <language>en</language>
        <item>
          <title>LessWrong Post</title>
          <link>https://www.lesswrong.com/posts/abc/example</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>LessWrong summary</description>
        </item>
      </channel>
    </rss>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://www.lesswrong.com/feed.xml",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    assert stories[0].source == "lesswrong"
    assert stories[0].badge_label == "LessWrong"


def test_parse_feed_entries_marks_slashdot_source_and_comments():
    xml = """
    <rss version="2.0" xmlns:slash="http://purl.org/rss/1.0/modules/slash/">
      <channel>
        <language>en</language>
        <item>
          <title>Slashdot Post</title>
          <link>https://slashdot.org/story/26/02/02/123456/example</link>
          <comments>https://slashdot.org/story/26/02/02/123456/example#comments</comments>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Slashdot summary</description>
          <slash:comments>42</slash:comments>
        </item>
      </channel>
    </rss>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://rss.slashdot.org/Slashdot/slashdotMain",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    story = stories[0]
    assert story.source == "slashdot"
    assert story.badge_label == "Slashdot"
    assert story.url == "https://slashdot.org/story/26/02/02/123456/example"
    assert story.discussion_url == "https://slashdot.org/story/26/02/02/123456/example#comments"
    assert story.score == 0
    assert story.comment_count == 42


def test_parse_feed_entries_marks_tildes_source_and_score():
    xml = """
    <rss version="2.0">
      <channel>
        <language>en</language>
        <item>
          <title>Tildes Post</title>
          <link>https://tildes.net/~tech/123/example</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>&lt;p&gt;Votes: 12&lt;/p&gt;&lt;p&gt;Tildes summary&lt;/p&gt;</description>
        </item>
      </channel>
    </rss>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://tildes.net/topics.rss",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    assert stories[0].source == "tildes"
    assert stories[0].badge_label == "Tildes"
    assert stories[0].score == 12


def test_parse_feed_entries_marks_reddit_source_and_external_link():
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Machine Learning</title>
      <subtitle>Machine learning research discussion and papers.</subtitle>
      <entry>
        <title>There Will Be a Scientific Theory of Deep Learning [R]</title>
        <link href="https://www.reddit.com/r/MachineLearning/comments/1sun588/post/" />
        <published>2026-04-24T17:58:00+00:00</published>
        <content type="html">
          &lt;div class=&quot;md&quot;&gt;&lt;p&gt;Research discussion summary.&lt;/p&gt;&lt;/div&gt;
          &lt;span&gt;&lt;a href=&quot;https://arxiv.org/abs/2604.21691&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt;
          &lt;span&gt;&lt;a href=&quot;https://www.reddit.com/r/MachineLearning/comments/1sun588/post/&quot;&gt;[comments]&lt;/a&gt;&lt;/span&gt;
        </content>
      </entry>
    </feed>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    assert stories[0].source == "reddit_machinelearning"
    assert stories[0].badge_label == "r/MachineLearning"
    assert stories[0].url == "https://arxiv.org/abs/2604.21691"
    assert (
        stories[0].discussion_url
        == "https://www.reddit.com/r/MachineLearning/comments/1sun588/post/"
    )


def test_parse_feed_entries_marks_other_subreddits_as_reddit():
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>programming</title>
      <subtitle>Computer programming news and technical software discussion.</subtitle>
      <entry>
        <title>Interesting Programming Link</title>
        <link href="https://www.reddit.com/r/programming/comments/abc/post/" />
        <published>2026-04-24T17:58:00+00:00</published>
        <content type="html">
          &lt;span&gt;&lt;a href=&quot;https://example.com/programming&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt;
          &lt;span&gt;&lt;a href=&quot;https://www.reddit.com/r/programming/comments/abc/post/&quot;&gt;[comments]&lt;/a&gt;&lt;/span&gt;
        </content>
      </entry>
    </feed>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://www.reddit.com/r/programming/top/.rss?t=week&limit=25",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    assert stories[0].source == "reddit_programming"
    assert stories[0].badge_label == "r/programming"
    assert stories[0].url == "https://example.com/programming"
    assert (
        stories[0].discussion_url
        == "https://www.reddit.com/r/programming/comments/abc/post/"
    )


def test_parse_feed_entries_reddit_self_post_uses_comments_url():
    comments_url = "https://www.reddit.com/r/MachineLearning/comments/1syjlc2/post/"
    xml = f"""
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Machine Learning</title>
      <subtitle>Machine learning research discussion and papers.</subtitle>
      <entry>
        <title>Why is reasoning not done in vector space? [D]</title>
        <link href="{comments_url}" />
        <published>2026-04-29T00:46:15+00:00</published>
        <content type="html">
          &lt;div class=&quot;md&quot;&gt;&lt;p&gt;Discussion of vector-space reasoning.&lt;/p&gt;&lt;/div&gt;
          &lt;span&gt;&lt;a href=&quot;{comments_url}&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt;
          &lt;span&gt;&lt;a href=&quot;{comments_url}&quot;&gt;[comments]&lt;/a&gt;&lt;/span&gt;
        </content>
      </entry>
    </feed>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    assert stories[0].source == "reddit_machinelearning"
    assert stories[0].url == comments_url
    assert stories[0].discussion_url == comments_url


def test_parse_feed_entries_extracts_reddit_score():
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Reddit Post With Score</title>
        <link href="https://www.reddit.com/r/MachineLearning/comments/123/" />
        <updated>2026-04-24T17:58:00Z</updated>
        <content type="html">
          &lt;span&gt;&lt;a href=&quot;https://example.com/&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt;
          &lt;span&gt;&lt;a href=&quot;https://www.reddit.com/r/MachineLearning/comments/123/&quot;&gt;[comments]&lt;/a&gt;&lt;/span&gt; (score: 789)
        </content>
      </entry>
    </feed>
    """
    stories = _parse_feed_entries(
        xml,
        feed_url="https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25",
        max_items=5,
        min_ts=0,
    )

    assert len(stories) == 1
    assert stories[0].score == 789


def test_feed_item_limit_uses_registry_curated_flag():
    assert _feed_item_limit("https://example.com/feed.xml", 5) == rss.RSS_PER_FEED_LIMIT
    assert (
        _feed_item_limit("https://lobste.rs/top/rss", 5)
        == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    )
    assert (
        _feed_item_limit("https://tildes.net/topics.rss", 5)
        == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    )
    assert (
        _feed_item_limit("https://www.lesswrong.com/feed.xml?view=frontpage", 5)
        == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    )
    assert (
        _feed_item_limit("https://rss.slashdot.org/Slashdot/slashdotMain", 5)
        == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    )
    assert (
        _feed_item_limit(
            "https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25", 5
        )
        == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    )
    assert (
        _feed_item_limit(
            "https://www.reddit.com/r/programming/top/.rss?t=week&limit=25", 5
        )
        == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    )
    assert (
        _feed_item_limit(
            "https://www.reddit.com/r/compsci/top/.rss?t=week&limit=25", 5
        )
        == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    )
    assert _feed_item_limit("https://digg.com/ai", 5) == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT


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
        <language>en</language>
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
async def test_fetch_rss_stories_parses_digg_ai_source(monkeypatch):
    opml_url = "https://example.com/feeds.opml"
    feed_url = "https://digg.com/ai"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [feed_url])

    html = r"""
    <script>self.__next_f.push([1,"{\"id\":\"b2f851a5-0a26-459e-ac2e-35bb611ba02a\",\"shortId\":\"svp61dsd\",\"title\":\"Digg AI Post\",\"tldr\":\"Digg AI summary.\",\"createdAt\":\"2026-05-15T16:12:17.887072+00:00\",\"firstPostAt\":\"2026-05-15T16:01:19+00:00\",\"lastFrozenPostAt\":\"2026-05-16T03:55:22+00:00\",\"postCount\":31}"])</script>
    """

    respx.get(opml_url).mock(return_value=Response(200, text=""))
    respx.get(feed_url).mock(return_value=Response(200, text=html))

    stories = await fetch_rss_stories(
        opml_url=opml_url,
        days=3650,
        max_feeds=5,
        per_feed=10,
        fetch_full_content=False,
    )

    assert len(stories) == 1
    assert stories[0].source == "digg"
    assert stories[0].title == "Digg AI Post"
    assert stories[0].comment_count == 31
    assert stories[0].url == "https://digg.com/ai/b2f851a5-0a26-459e-ac2e-35bb611ba02a"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_rss_stories_skips_excluded_feeds(monkeypatch):
    opml_url = "https://example.com/feeds-with-exclusions.opml"
    excluded_feed = "https://rachelbythebay.com/w/atom.xml"
    allowed_feed = "https://example.com/allowed-feed.xml"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [])
    monkeypatch.setattr(rss, "RSS_EXCLUDED_FEEDS", {excluded_feed})

    opml = f"""
    <opml version="2.0">
      <body>
        <outline type="rss" text="Excluded" xmlUrl="{excluded_feed}"/>
        <outline type="rss" text="Allowed" xmlUrl="{allowed_feed}"/>
      </body>
    </opml>
    """
    rss_xml = """
    <rss version="2.0">
      <channel>
        <language>en</language>
        <item>
          <title>Allowed Post</title>
          <link>https://example.com/post</link>
          <pubDate>Mon, 02 Feb 2026 12:00:00 GMT</pubDate>
          <description>Allowed summary</description>
        </item>
      </channel>
    </rss>
    """

    respx.get(opml_url).mock(return_value=Response(200, text=opml))
    excluded_route = respx.get(excluded_feed).mock(return_value=Response(500))
    respx.get(allowed_feed).mock(return_value=Response(200, text=rss_xml))

    stories = await fetch_rss_stories(
        opml_url=opml_url,
        days=3650,
        max_feeds=5,
        per_feed=10,
        fetch_full_content=False,
    )

    assert [story.title for story in stories] == ["Allowed Post"]
    assert not excluded_route.called



@pytest.mark.asyncio
@respx.mock
async def test_fetch_rss_stories_uses_higher_default_limit_for_regular_rss(monkeypatch):
    opml_url = "https://example.com/feeds.opml"
    feed_url = "https://example.com/feed.xml"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [feed_url])

    items = []
    for i in range(30):
        items.append(
            f"""
            <item>
              <title>Regular Feed Post {i}</title>
              <link>https://example.com/posts/{i}</link>
              <pubDate>Mon, 02 Feb 2026 12:{i:02d}:00 GMT</pubDate>
              <description>Regular feed summary {i}</description>
            </item>
            """
        )
    rss_xml = f"""
    <rss version="2.0">
      <channel>
        <language>en</language>
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

    assert len(stories) == 30
    assert stories[0].source == "rss"
    assert stories[-1].title == "Regular Feed Post 29"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_rss_stories_uses_higher_limit_for_curated_news_sources(monkeypatch):
    opml_url = "https://example.com/feeds.opml"
    feed_url = "https://www.lesswrong.com/feed.xml"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [feed_url])

    items = []
    for i in range(60):
        items.append(
            f"""
            <item>
              <title>LessWrong Post {i}</title>
              <link>https://www.lesswrong.com/posts/{i}/post-{i}</link>
              <pubDate>Mon, 02 Feb 2026 12:{i:02d}:00 GMT</pubDate>
              <description>LessWrong summary {i}</description>
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
    assert stories[0].source == "lesswrong"
    assert stories[-1].title == f"LessWrong Post {rss.RSS_CURATED_NEWS_PER_FEED_LIMIT - 1}"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_rss_stories_uses_higher_limit_for_reddit(monkeypatch):
    opml_url = "https://example.com/feeds.opml"
    feed_url = "https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25"
    monkeypatch.setattr(rss, "RSS_EXTRA_FEEDS", [feed_url])

    entries = []
    for i in range(60):
        comments_url = f"https://www.reddit.com/r/MachineLearning/comments/{i}/post-{i}/"
        entries.append(
            f"""
            <entry>
              <title>Reddit ML Post {i}</title>
              <link href="{comments_url}" />
              <published>2026-04-24T12:{i:02d}:00+00:00</published>
              <content type="html">
                &lt;div class=&quot;md&quot;&gt;&lt;p&gt;Machine learning research post {i}.&lt;/p&gt;&lt;/div&gt;
                &lt;span&gt;&lt;a href=&quot;https://example.com/paper-{i}&quot;&gt;[link]&lt;/a&gt;&lt;/span&gt;
                &lt;span&gt;&lt;a href=&quot;{comments_url}&quot;&gt;[comments]&lt;/a&gt;&lt;/span&gt;
              </content>
            </entry>
            """
        )
    atom_xml = f"""
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Machine Learning</title>
      <subtitle>Machine learning research discussion and papers.</subtitle>
      {''.join(entries)}
    </feed>
    """

    respx.get(opml_url).mock(return_value=Response(200, text=""))
    respx.get(feed_url).mock(return_value=Response(200, text=atom_xml))

    stories = await fetch_rss_stories(
        opml_url=opml_url,
        days=3650,
        max_feeds=5,
        per_feed=5,
        fetch_full_content=False,
    )

    assert len(stories) == rss.RSS_CURATED_NEWS_PER_FEED_LIMIT
    assert stories[0].source == "reddit_machinelearning"
    assert stories[0].url == "https://example.com/paper-0"
    assert stories[-1].title == f"Reddit ML Post {rss.RSS_CURATED_NEWS_PER_FEED_LIMIT - 1}"
