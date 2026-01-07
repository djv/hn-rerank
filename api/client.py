import json
import time
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
from api.constants import USER_CACHE_DIR, USER_CACHE_TTL

COOKIES_FILE = Path.home() / ".config" / "hn_rerank" / "cookies.json"
USER_CACHE_DIR_PATH = Path(USER_CACHE_DIR)
USER_CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)

class HNClient:
    BASE_URL = "https://news.ycombinator.com"

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        self._load_cookies()

    def _load_cookies(self) -> None:
        if COOKIES_FILE.exists():
            try:
                cookies = json.loads(COOKIES_FILE.read_text())
                self.client.cookies.update(cookies)
            except Exception:
                pass

    async def login(self, user, pw) -> tuple[bool, str]:
        resp = await self.client.get("/login")
        soup = BeautifulSoup(resp.text, "html.parser")
        fnid = soup.find("input", {"name": "fnid"})["value"]
        resp = await self.client.post("/login", data={"acct": user, "pw": pw, "fnid": fnid})
        if "logout" in resp.text:
            COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
            COOKIES_FILE.write_text(json.dumps(dict(self.client.cookies)))
            return True, "Success"
        return False, "Login failed"

    async def _scrape_ids(self, path, max_pages=3) -> set[int]:
        ids = set()
        for p in range(1, max_pages + 1):
            url = f"{path}&p={p}" if "?" in path else f"{path}?p={p}"
            resp = await self.client.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.find_all("tr", class_="athing")
            for r in rows:
                ids.add(int(r["id"]))
            if not soup.find("a", class_="morelink"):
                break
        return ids

    async def fetch_user_submissions(self, username: str) -> set[int]:
        """Fetch submission IDs for a user using the official Firebase API."""
        resp = await self.client.get(f"https://hacker-news.firebaseio.com/v0/user/{username}.json")
        if resp.status_code == 200:
            data = resp.json()
            return set(data.get("submitted", []))
        return set()

    async def fetch_user_data(self, user) -> dict[str, set[int]]:
        cache_path = USER_CACHE_DIR_PATH / f"{user}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            if time.time() - data["ts"] < USER_CACHE_TTL:
                return {k: set(v) for k, v in data["ids"].items()}

        is_logged_in = "logout" in (await self.client.get("/")).text
        
        pos = await self._scrape_ids(f"/favorites?id={user}")
        if is_logged_in:
            pos.update(await self._scrape_ids(f"/upvoted?id={user}"))
            neg = await self._scrape_ids(f"/hidden?id={user}")
        else:
            neg = set()

        out = {"pos": pos, "neg": neg}
        cache_path.write_text(json.dumps({"ts": time.time(), "ids": {k: list(v) for k, v in out.items()}}))
        return out

    async def close(self): await self.client.aclose()
    async def __aenter__(self): return self
    async def __aexit__(self, *args): await self.close()
