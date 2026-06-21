import os
import asyncio
import json
import logging
import secrets
import sys
import threading
import time

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup
import httpx

from dataclasses import replace
from database import Database, User
from pipeline import Config


def load_env() -> None:
    # Try local .env
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path("/home/dev/hn_rerank/.env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")


async def _fetch_article_body(url: str) -> str | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
    }

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code in (429, 503) and attempt == 0:
                    await asyncio.sleep(1)
                    continue
                if resp.status_code != 200:
                    return None
                html = resp.text
        except Exception:
            return None

        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text and len(text) > 200:
            return text[:10000]

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        for tag in soup.find_all(["article", "main"]):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return text[:10000]
        text = soup.get_text(separator=" ", strip=True)
        return text[:10000] if len(text) > 200 else None

    return None


async def generate_detailed_tldr(
    title: str,
    self_text: str = "",
    top_comments: str = "",
    article_body: str = "",
    points: int = 0,
    comment_count: int = 0,
    age_hours: float = 0.0,
) -> str | None:
    provider = os.environ.get("LLM_PROVIDER", "mistral").lower()
    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        base_url = "https://api.mistral.ai/v1/chat/completions"
        model = "mistral-small-latest"
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        base_url = "https://api.groq.com/openai/v1/chat/completions"
        model = "llama-3.3-70b-versatile"

    if not api_key:
        return "Error: LLM API key not configured in environment."

    content_section = f"Title: {title}"
    if self_text:
        content_section += f"\n\nAuthor's text:\n{self_text[:6000]}"
    if article_body:
        content_section += f"\n\nArticle body:\n{article_body[:15000]}"
    if top_comments:
        content_section += f"\n\nHN comments:\n{top_comments[:10000]}"

    prompt = f"""Summarize the article and the discussion for a knowledgeable reader.
Use ONLY information from the text below.
Write a highly concise, scannable summary (under 180 words) optimized for an 11-inch screen to conserve vertical space.
Use Markdown formatting:
- Headings (###) for main sections.
- Short bullet points (-) with **bold** key terms.
- No nested list levels (conserve horizontal margins).
- Keep each bullet point to a single short sentence.

{content_section}

IMPORTANT:
- Use ONLY information from the article text. Do not expand on the topic with outside knowledge.
- If the article has very few or no comments do not invent discussion.
- If the text below is just a title and no substantive content, say so explicitly.
"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000,
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error from LLM Provider: HTTP {resp.status_code} - {resp.text}"
    except Exception as e:
        return f"Error executing LLM call: {str(e)}"


class Handler(BaseHTTPRequestHandler):
    server_version = "HNRewrite/1.0"
    config: Config
    db: Database
    embedder: object
    regen_event: threading.Event
    _dashboard_cache: dict = {}

    def _get_user(self) -> User | None:
        """Extract user from cookie token."""
        cookie_header = self.headers.get("Cookie", "")
        for part in cookie_header.split(";"):
            kv = part.strip().split("=", 1)
            if len(kv) == 2 and kv[0] == "hn_token":
                return self.db.get_or_create_user(kv[1].strip())
        return None

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        # Token-based user session
        if path.startswith("/u/"):
            token = path[3:].strip("/")
            if token:
                user = self.db.get_or_create_user(token)
                self.send_response(302)
                self.send_header("Location", "../")
                self.send_header("Set-Cookie", f"hn_token={user.token}; Path=/; Max-Age=31536000")
                self.end_headers()
                return
            self.send_error(HTTPStatus.BAD_REQUEST)
            return

        # User info API
        if path == "/api/user":
            user = self._get_user()
            if user:
                self._json_response({"user_id": user.id, "username": user.username, "token": user.token})
            else:
                self._json_response({"error": "No session"}, status=401)
            return

        # Dashboard — dynamic render per-user
        if path in ("/", "/index.html"):
            user = self._get_user()
            if not user:
                token = secrets.token_hex(4)
                self.send_response(302)
                self.send_header("Location", f"u/{token}")
                self.send_header("Set-Cookie", f"hn_token={token}; Path=/; Max-Age=31536000")
                self.end_headers()
                return

            # Render personalized dashboard
            html = self._render_dashboard(user)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(html)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _render_dashboard(self, user: User) -> bytes:
        """Render personalized dashboard for user. Uses short-lived cache."""
        now = time.time()
        cache_key = f"dashboard_{user.id}"

        cached = self._dashboard_cache.get(cache_key)
        if cached and now - cached[1] < 300:
            return cached[0]

        from pipeline import fast_rerank_for_user, generate_dashboard_bytes

        final = fast_rerank_for_user(self.db, self.config, self.embedder, user.id)
        html = generate_dashboard_bytes(final, self.config, user.username, self.db, user.id)

        self._dashboard_cache[cache_key] = (html, now)
        return html

    def do_POST(self) -> None:
        if self.path == "/api/feedback":
            user = self._get_user()
            if not user:
                self._json_response({"error": "No session"}, status=401)
                return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)

                story_id = data["story_id"]
                action = data["action"]

                if action == "clear":
                    self.db.delete_feedback(user.id, story_id)
                else:
                    self.db.upsert_feedback(
                        user.id,
                        story_id,
                        action,
                    )

                # Invalidate user's dashboard cache
                cache_key = f"dashboard_{user.id}"
                self._dashboard_cache.pop(cache_key, None)

                # Also trigger background regen for candidate updates
                self.regen_event.set()
                self._json_response({"ok": True})
            except Exception as e:
                logging.error(f"Error handling feedback: {e}")
                self._json_response({"error": str(e)}, status=400)
        elif self.path == "/api/tldr-detail":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)

                story_id = data["story_id"]
                now = time.time()

                story = self.db.get_story(story_id)
                if not story:
                    self._json_response(
                        {"error": "Story not found in database"}, status=404
                    )
                    return

                # 1. If HN story has comments but top_comments is empty, dynamically fetch them
                if story.source == "hn" and not story.top_comments and (story.comment_count or 0) > 0:
                    try:
                        from pipeline import fetch_story
                        async def do_fetch():
                            async with httpx.AsyncClient(timeout=15.0) as client:
                                return await fetch_story(client, story_id, self.db)
                        updated = asyncio.run(do_fetch())
                        if updated:
                            story = updated
                    except Exception as e:
                        logging.error(f"Failed to dynamically fetch comments for TLDR: {e}")

                age_hours = max(0.0, (now - story.time) / 3600.0)
                article_body = story.article_body or None

                if (
                    article_body is None
                    and story.url
                    and not story.url.startswith("https://news.ycombinator.com")
                    and len(story.self_text) < 500
                ):
                    article_body = asyncio.run(_fetch_article_body(story.url))
                    if article_body:
                        article_body = article_body[:15000]
                        from pipeline import compose_story_text
                        new_text = compose_story_text(
                            story.title,
                            story.self_text,
                            story.top_comments,
                            article_body,
                        )
                        updated_story = replace(
                            story,
                            article_body=article_body,
                            text_content=new_text,
                        )
                        self.db.upsert_story(updated_story)
                tldr = asyncio.run(
                    generate_detailed_tldr(
                        story.title,
                        self_text=story.self_text or "",
                        top_comments=story.top_comments or "",
                        article_body=article_body or "",
                        points=story.score,
                        comment_count=story.comment_count or 0,
                        age_hours=age_hours,
                    )
                )
                if tldr:
                    self._json_response({"ok": True, "tldr": tldr})
                else:
                    self._json_response(
                        {"error": "Failed to generate TLDR"}, status=500
                    )
            except Exception as e:
                logging.error(f"Error handling tldr-detail: {e}")
                self._json_response({"error": str(e)}, status=400)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        logging.info("%s - - %s" % (self.address_string(), format % args))


def regen_loop(config: Config, event: threading.Event, db: Database) -> None:
    logging.info("Starting background regeneration loop...")
    while True:
        # Wait on event or timeout
        triggered = event.wait(timeout=config.regen_interval_seconds)
        if triggered:
            event.clear()
            # Debounce click storms
            time.sleep(2)

        logging.info("Regeneration triggered. Fetching candidates...")
        try:
            from pipeline import fetch_candidates_only
            asyncio.run(fetch_candidates_only(config, db))
            logging.info("Regeneration complete.")
        except Exception as e:
            logging.exception(f"Background regeneration failed: {e}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    load_env()
    config = Config.load()
    db = Database(config.db_path)

    regen_event = threading.Event()
    from pipeline import Embedder
    embedder = Embedder(config.onnx_model_dir)
    Handler.config = config
    Handler.db = db
    Handler.embedder = embedder
    Handler.regen_event = regen_event

    # Start regen thread
    t = threading.Thread(target=regen_loop, args=(config, regen_event, db), daemon=True)
    t.start()

    # Start HTTP server
    server = ThreadingHTTPServer(("0.0.0.0", config.server_port), Handler)
    logging.info("Serving on http://0.0.0.0:%d", config.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        server.server_close()
        db.close()


if __name__ == "__main__":
    main()
