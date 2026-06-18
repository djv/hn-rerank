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
import httpx

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


async def generate_detailed_tldr(title: str, text_content: str) -> str | None:
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

    prompt = f"""Summarize the article and the discussion for a knowledgeable reader.
Use ONLY information from the article text below (which includes comments).
Write a short 3-4 paragraph summary (under 400 words). Use Markdown formatting:
headings (###), **bold** for key terms, and - for lists where appropriate.

Title: {title}
Article text: {text_content[:30000]}
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

                db = Database(self.config.db_path)
                try:
                    if action == "clear":
                        db.delete_feedback(story_id)
                    else:
                        db.upsert_feedback(
                            story_id=story_id,
                            action=action,
                        )
                finally:
                    db.close()

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

                db = Database(self.config.db_path)
                try:
                    story = db.get_story(story_id)
                finally:
                    db.close()

                if not story:
                    self._json_response(
                        {"error": "Story not found in database"}, status=404
                    )
                    return

                tldr = asyncio.run(
                    generate_detailed_tldr(story.title, story.text_content)
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
    Handler.config = config
    Handler.db = db
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
