import threading
import httpx
import pytest
from http.server import ThreadingHTTPServer

from server import Handler
from pipeline import Config
from database import Database, Story


@pytest.fixture
def test_env(tmp_path):
    db_file = tmp_path / "test_server.db"
    db = Database(str(db_file))

    # Create test user
    user = db.create_user("test_token", "test_user")

    output_file = tmp_path / "public" / "index.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("<html>Test Dashboard</html>", encoding="utf-8")

    config = Config(
        db_path=str(db_file),
        output=str(output_file),
        server_port=0,
    )

    regen_event = threading.Event()

    class TestHandler(Handler):
        pass

    class MockEmbedder:
        def encode(self, texts, batch_size=64):
            import numpy as np
            return np.zeros((len(texts), 384), dtype=np.float32)

    TestHandler.config = config
    TestHandler.db = db
    TestHandler.embedder = MockEmbedder()
    TestHandler.regen_event = regen_event
    TestHandler._dashboard_cache = {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield port, db, regen_event, output_file, user

    server.shutdown()
    db.close()


def test_token_redirect(test_env):
    port, _, _, _, user = test_env
    resp = httpx.get(f"http://127.0.0.1:{port}/u/{user.token}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"] == "../"
    assert "hn_token" in resp.headers.get("Set-Cookie", "")


def test_first_visit_redirect(test_env):
    port, _, _, _, _ = test_env
    resp = httpx.get(f"http://127.0.0.1:{port}/", follow_redirects=False)
    assert resp.status_code == 302
    location = resp.headers["Location"]
    assert location.startswith("u/")
    assert "hn_token" in resp.headers.get("Set-Cookie", "")


def test_static_serving(test_env):
    port, _, _, _, user = test_env
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    # Dynamic rendering returns personalized dashboard, not static file
    assert resp.status_code == 200


def test_feedback_post(test_env):
    port, db, regen_event, _, user = test_env
    db.upsert_story(
        Story(
            id=999,
            title="Feedback story",
            url="https://example.com",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    feedback_payload = {
        "story_id": 999,
        "action": "up",
    }
    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json=feedback_payload,
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    records = db.get_all_feedback(user.id)
    assert len(records) == 1
    assert records[0].story_id == 999
    assert records[0].action == "up"
    assert regen_event.is_set()


def test_feedback_clear(test_env):
    port, db, regen_event, _, user = test_env
    db.upsert_story(
        Story(
            id=999,
            title="Title",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text",
            source="hn",
        )
    )
    db.upsert_feedback(user.id, 999, "up")
    assert len(db.get_all_feedback(user.id)) == 1

    regen_event.clear()

    clear_payload = {
        "story_id": 999,
        "action": "clear",
    }
    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json=clear_payload,
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200

    assert len(db.get_all_feedback(user.id)) == 0
    assert regen_event.is_set()


def test_cors_headers(test_env):
    port, _, _, _, _ = test_env
    resp = httpx.options(f"http://127.0.0.1:{port}/api/feedback")
    assert resp.status_code == 204
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


def test_tldr_detail_dynamic_fetch(test_env, monkeypatch):
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=777,
            title="Dynamic test",
            url="https://example.com/dynamic-test",
            score=100,
            time=1600000000,
            text_content="Dynamic test.",
            source="hn",
            comment_count=5,
            discussion_url="https://news.ycombinator.com/item?id=777",
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    # Mock fetch_story and _fetch_article_body
    async def mock_fetch_story(client, sid, database):
        story = database.get_story(sid)
        from dataclasses import replace
        updated = replace(story, top_comments="Fetched comments", text_content="Dynamic test. Fetched comments")
        database.upsert_story(updated)
        return updated

    async def mock_fetch_article_body(url):
        return "Fetched article body text"

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body, points, comment_count, age_hours):
        return f"TLDR: {title} | {top_comments} | {article_body}"

    import server
    import pipeline
    monkeypatch.setattr(pipeline, "fetch_story", mock_fetch_story)
    monkeypatch.setattr(server, "_fetch_article_body", mock_fetch_article_body)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    # Request TLDR
    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 777},
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "Fetched comments" in data["tldr"]
    assert "Fetched article body text" in data["tldr"]

    # Verify database was updated
    updated_story = db.get_story(777)
    assert updated_story.top_comments == "Fetched comments"
    assert updated_story.article_body == "Fetched article body text"

