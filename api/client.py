from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import httpx
from bs4 import BeautifulSoup
from api.constants import USER_CACHE_DIR, USER_CACHE_TTL

USER_CACHE_DIR_PATH: Path = Path(USER_CACHE_DIR)
USER_CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
COOKIES_FILE: Path = USER_CACHE_DIR_PATH / "cookies.json"


class HNClient:
    BASE_URL: str = "https://news.ycombinator.com"

    def __init__(self) -> None:
        self.client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.BASE_URL,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        self._load_cookies()

    def _load_cookies(self) -> None:
        if COOKIES_FILE.exists():
            try:
                cookies: dict[str, Any] = json.loads(COOKIES_FILE.read_text())
                self.client.cookies.update(cookies)
            except Exception:
                pass

    async def login(self, user: str, pw: str) -> tuple[bool, str]:
        resp: httpx.Response = await self.client.get("/login")
        soup: BeautifulSoup = BeautifulSoup(resp.text, "html.parser")
        fnid_tag: Any = soup.find("input", {"name": "fnid"})

        data: dict[str, str] = {"acct": user, "pw": pw}
        if fnid_tag:
            data["fnid"] = str(fnid_tag["value"])

        resp = await self.client.post("/login", data=data)
        if "logout" in resp.text:
            COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
            COOKIES_FILE.write_text(json.dumps(dict(self.client.cookies)))
            return True, "Success"

        # Check for specific error message in HN response
        error_msg: str = "Login failed"
        if "Bad login" in resp.text:
            error_msg = "Bad login (check username/password)"

        return False, error_msg

    async def _scrape_ids(self, path: str, max_pages: int = 10) -> set[int]:
        ids: set[int] = set()
        for p in range(1, max_pages + 1):
            url: str = f"{path}&p={p}" if "?" in path else f"{path}?p={p}"
            resp: httpx.Response = await self.client.get(url)
            soup: BeautifulSoup = BeautifulSoup(resp.text, "html.parser")
            rows: Any = soup.find_all("tr", class_="athing")
            for r in rows:
                ids.add(int(r["id"]))
            if not soup.find("a", class_="morelink"):
                break
        return ids

    async def fetch_user_submissions(self, username: str) -> set[int]:
        """Fetch submission IDs for a user using the official Firebase API."""
        resp: httpx.Response = await self.client.get(
            f"https://hacker-news.firebaseio.com/v0/user/{username}.json"
        )
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            return set(data.get("submitted", []))
        return set()

    async def fetch_user_data(self, user: str) -> dict[str, set[int]]:
        cache_path: Path = USER_CACHE_DIR_PATH / f"{user}.json"
        if cache_path.exists():
            data: dict[str, Any] = json.loads(cache_path.read_text())
            if time.time() - float(data["ts"]) < USER_CACHE_TTL:
                return {k: set(v) for k, v in data["ids"].items()}

        is_logged_in: bool = "logout" in (await self.client.get("/")).text

        # Favorites are always positive signals
        favorites: set[int] = await self._scrape_ids(f"/favorites?id={user}")

        upvoted: set[int]
        hidden: set[int]
        if is_logged_in:
            # If logged in, we also know what we've already upvoted and hidden
            upvoted = await self._scrape_ids(f"/upvoted?id={user}")
            hidden = await self._scrape_ids(f"/hidden?id={user}")
        else:
            upvoted = set()
            hidden = set()

        # Output format:
        # pos: used for semantic profile building
        # excluded: IDs to NEVER show as candidates
        out: dict[str, set[int]] = {
            "pos": favorites,
            "upvoted": upvoted,
            "hidden": hidden,
        }
        cache_path.write_text(
            json.dumps({"ts": time.time(), "ids": {k: list(v) for k, v in out.items()}})
        )
        return out

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> HNClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
