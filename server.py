import os
import asyncio
import json
import logging
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
from database import Database
from pipeline import Config, run_pipeline


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
    regen_event: threading.Event

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/public/", "/public/index.html"):
            target_file = Path(self.config.output)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        if not target_file.exists():
            # Trigger generation and wait a brief moment for it to complete
            logging.info("Dashboard HTML not found. Triggering immediate generation...")
            self.regen_event.set()
            # Loop-wait up to 5 seconds for generation
            for _ in range(50):
                time.sleep(0.1)
                if target_file.exists():
                    break

        if not target_file.exists():
            self.send_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Dashboard is generating, please refresh in a moment.",
            )
            return

        try:
            content = target_file.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_POST(self) -> None:
        if self.path == "/api/feedback":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)

                story_id = data["story_id"]
                action = data["action"]

                if action == "clear":
                    self.db.delete_feedback(story_id)
                else:
                    self.db.upsert_feedback(
                        story_id=story_id,
                        action=action,
                    )

                # Synchronously trigger fast re-ranking and re-rendering of index.html
                from pipeline import fast_rerank_and_render
                fast_rerank_and_render(self.db, self.config, self.embedder)

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

                age_hours = max(0.0, (now - story.time) / 3600.0)
                article_body = story.article_body or None

                if article_body is None and story.url and len(story.text_content) < 500:
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
        # Silence access logs to keep console output concise
        pass


def regen_loop(config: Config, event: threading.Event) -> None:
    logging.info("Starting background regeneration loop...")
    while True:
        # Wait on event or timeout
        triggered = event.wait(timeout=config.regen_interval_seconds)
        if triggered:
            event.clear()
            # Debounce click storms
            time.sleep(2)

        logging.info("Regeneration triggered. Running pipeline...")
        try:
            asyncio.run(run_pipeline(config))
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
    t = threading.Thread(target=regen_loop, args=(config, regen_event), daemon=True)
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
