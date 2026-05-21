#!/usr/bin/env -S uv run
"""Small local HTTP API for dashboard feedback writes."""

from __future__ import annotations

import asyncio
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from bs4 import BeautifulSoup
from bs4.element import Tag

from api.client import HNClient
from api.feedback import (
    FEEDBACK_STORE_PATH,
    FeedbackMirrorStatus,
    FeedbackPayload,
    apply_feedback_payload,
    load_feedback,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _feedback_token() -> str:
    _load_env_file(Path.home() / ".config/hn_rerank/secrets.env")
    _load_env_file(Path(".env"))
    return os.environ.get("HN_RERANK_FEEDBACK_TOKEN", "")


def _extract_hn_action_path(html: str, story_id: int, action: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    if action == "up":
        prefix = f"vote?id={story_id}&how=up"
    else:
        prefix = f"hide?id={story_id}"

    for link in soup.find_all("a"):
        if not isinstance(link, Tag):
            continue
        href = link.get("href")
        if isinstance(href, str) and href.startswith(prefix):
            return href
    return None


async def _mirror_hn_action(payload: FeedbackPayload) -> tuple[FeedbackMirrorStatus, str | None]:
    if payload.get("source") != "hn":
        return "none", None
    story_id = int(payload.get("id", 0))
    if story_id <= 0:
        return "none", None

    action = payload.get("action")
    if action == "clear":
        return "none", None
    try:
        async with HNClient() as hn:
            home = await hn.client.get("/")
            if "logout" not in home.text:
                return "failed", "HN cookies are not logged in"
            item = await hn.client.get(f"/item?id={story_id}")
            target = _extract_hn_action_path(item.text, story_id, action)
            if target is None:
                return "failed", f"HN {action} link not found"
            resp = await hn.client.get(target)
            if resp.status_code >= 400:
                return "failed", f"HN returned HTTP {resp.status_code}"
    except Exception as exc:
        return "failed", str(exc)
    return "success", None


class FeedbackHandler(BaseHTTPRequestHandler):
    server_version = "HNRerankFeedback/1.0"

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        if self.path != "/api/feedback":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self._authorized():
            self.send_error(HTTPStatus.UNAUTHORIZED)
            return
        records = load_feedback(FEEDBACK_STORE_PATH)
        self._send_json(
            {"records": {key: record.to_dict() for key, record in records.items()}}
        )

    def do_POST(self) -> None:
        if self.path != "/api/feedback":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self._authorized():
            self.send_error(HTTPStatus.UNAUTHORIZED)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = cast(FeedbackPayload, json.loads(raw_body.decode("utf-8")))
            self._validate_payload(payload)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        mirror_status, mirror_error = asyncio.run(_mirror_hn_action(payload))
        records, record = apply_feedback_payload(
            payload,
            path=FEEDBACK_STORE_PATH,
            mirror_status=mirror_status,
            mirror_error=mirror_error,
        )
        self._send_json(
            {
                "ok": True,
                "record": record.to_dict() if record else None,
                "records": {
                    key: saved_record.to_dict()
                    for key, saved_record in records.items()
                },
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        token = _feedback_token()
        if not token:
            return False
        return self.headers.get("X-HN-RERANK-FEEDBACK-TOKEN") == token

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-HN-RERANK-FEEDBACK-TOKEN",
        )
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _validate_payload(payload: FeedbackPayload) -> None:
        if payload.get("action") not in {"up", "down", "clear"}:
            raise ValueError("action must be up, down, or clear")
        if not isinstance(payload.get("id"), int):
            raise ValueError("id must be an integer")
        if not payload.get("source"):
            raise ValueError("source is required")
        if not payload.get("title"):
            raise ValueError("title is required")


def main() -> None:
    host = os.environ.get("HN_RERANK_FEEDBACK_HOST", DEFAULT_HOST)
    port = int(os.environ.get("HN_RERANK_FEEDBACK_PORT", str(DEFAULT_PORT)))
    server = ThreadingHTTPServer((host, port), FeedbackHandler)
    print(f"Serving HN Rerank feedback API on http://{host}:{port}/api/feedback")
    server.serve_forever()


if __name__ == "__main__":
    main()
