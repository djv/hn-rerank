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

    TestHandler.config = config
    TestHandler.db = db
    TestHandler.regen_event = regen_event

    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield port, db, regen_event, output_file

    server.shutdown()
    server.server_close()
    db.close()


def test_static_serving(test_env):
    port, _, _, _ = test_env
    resp = httpx.get(f"http://127.0.0.1:{port}/")
    assert resp.status_code == 200
    assert "Test Dashboard" in resp.text


def test_feedback_post(test_env):
    port, db, regen_event, _ = test_env
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
        "title": "Feedback story",
        "url": "https://example.com",
        "text_content": "Feedback body text",
        "source": "hn",
    }
    resp = httpx.post(f"http://127.0.0.1:{port}/api/feedback", json=feedback_payload)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    records = db.get_all_feedback()
    assert len(records) == 1
    assert records[0].story_id == 999
    assert records[0].action == "up"
    assert regen_event.is_set()


def test_feedback_clear(test_env):
    port, db, regen_event, _ = test_env
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
    db.upsert_feedback(999, "up")
    assert len(db.get_all_feedback()) == 1

    regen_event.clear()

    clear_payload = {
        "story_id": 999,
        "action": "clear",
    }
    resp = httpx.post(f"http://127.0.0.1:{port}/api/feedback", json=clear_payload)
    assert resp.status_code == 200

    assert len(db.get_all_feedback()) == 0
    assert regen_event.is_set()


def test_cors_headers(test_env):
    port, _, _, _ = test_env
    resp = httpx.options(f"http://127.0.0.1:{port}/api/feedback")
    assert resp.status_code == 204
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")
