import pytest
from server import _fetch_article_body

_ARTICLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Article</title></head>
<body>
<article>
<h1>Test Headline</h1>
<p>This is the body of the test article. It contains enough text to
pass the 200-character minimum threshold required by the extractor.
We need to make sure this paragraph is long enough so trafilatura
and the BS4 fallback both consider it meaningful content.</p>
<p>Here is some additional text to pad out the length even further.
Traffic comes from many sources and the extractor needs enough
content to be confident it has found the real article body and not
some sidebar or footer text. This should be more than enough.</p>
</article>
<footer>Footer text that should be stripped.</footer>
</body>
</html>
"""

_JUNK_HTML_WITH_TEXT = """\
<!DOCTYPE html>
<html>
<body>
<div id="comments">
<p>This is a comment-like block of text that trafilatura may skip
because it is inside a generic div without article or main tags.
The text is long enough to be considered content but the structure
is too loose for trafilatura to extract confidently.</p>
<p>More text here to ensure we cross the 200-character boundary so
the BeautifulSoup fallback can pick it up when trafilatura gives up.
This simulates poorly-structured pages that still contain readable
content that we want to extract.</p>
</div>
</body>
</html>
"""

_EMPTY_HTML = """\
<!DOCTYPE html>
<html>
<body>
<p>Short.</p>
</body>
</html>
"""


async def _serve(handler, status=200, body=b"", delay=0.0):
    import asyncio
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    class TestHandler(BaseHTTPRequestHandler):
        _call_count = 0

        def do_GET(self):
            type(self)._call_count += 1
            if delay:
                asyncio.run(asyncio.sleep(delay))
            if callable(status):
                s, b = status(self)
            else:
                s, b = status, body
            self.send_response(s)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}/{handler}"
    try:
        result = await _fetch_article_body(url)
    finally:
        server.shutdown()
    return result, TestHandler._call_count


@pytest.mark.asyncio
async def test_fetch_success():
    body = _ARTICLE_HTML.encode()
    result, calls = await _serve("success", 200, body)
    assert result is not None
    assert "Test Headline" in result
    assert "body of the test article" in result
    assert "Footer text" not in result
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_404():
    result, calls = await _serve("notfound", 404, b"")
    assert result is None
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_retry_on_503():
    state = {"attempts": 0}

    def status_handler(inst):
        state["attempts"] += 1
        if state["attempts"] == 1:
            return 503, b""
        return 200, _ARTICLE_HTML.encode()

    result, calls = await _serve("retry", status_handler)
    assert result is not None
    assert "Test Headline" in result
    assert calls == 2


@pytest.mark.asyncio
async def test_fetch_retry_exhausted():
    state = {"attempts": 0}

    def status_handler(inst):
        state["attempts"] += 1
        return 503, b""

    result, calls = await _serve("exhausted", status_handler)
    assert result is None
    assert calls == 2


@pytest.mark.asyncio
async def test_fetch_bs4_fallback():
    body = _JUNK_HTML_WITH_TEXT.encode()
    result, calls = await _serve("junk", 200, body)
    assert result is not None
    assert "comment-like block" in result
    assert "BeautifulSoup fallback" in result
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_empty_body():
    body = _EMPTY_HTML.encode()
    result, calls = await _serve("empty", 200, body)
    assert result is None
    assert calls == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_fetch_real_urls_from_db():
    """Smoke test: fetch article bodies for real HN URLs from the production DB."""
    import sqlite3
    from pathlib import Path

    db_path = Path("/home/dev/hn-rewrite/hn_rewrite.db")
    if not db_path.exists():
        pytest.skip("production DB not found")

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT url FROM stories WHERE url IS NOT NULL AND url != ''"
        " AND score > 50 AND source = 'hn' ORDER BY score DESC LIMIT 3"
    ).fetchall()
    conn.close()

    if not rows:
        pytest.skip("no high-score stories with URLs in DB")

    for (url,) in rows:
        result = await _fetch_article_body(url)
        # Some sites are JS-rendered SPAs, PDFs, etc. — None is acceptable;
        # the test passes as long as no exception is thrown.
        if result is not None:
            assert len(result) >= 200, f"body too short ({len(result)} chars) for {url}"
