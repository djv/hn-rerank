import json
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

COOKIES_FILE = Path.home() / ".config" / "hn_rerank" / "cookies.json"
USER_CACHE_DIR = Path(".cache/user")
USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
USER_CACHE_TTL = 300  # 5 minutes


class HNClient:
    BASE_URL = "https://news.ycombinator.com"

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        self.username: str | None = None
        self._load_cookies()

    def _load_cookies(self) -> None:
        if COOKIES_FILE.exists():
            try:
                with open(COOKIES_FILE) as f:
                    cookies = json.load(f)
                    for k, v in cookies.items():
                        self.client.cookies.set(k, v)
            except Exception:
                pass

    def _save_cookies(self) -> None:
        COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COOKIES_FILE, "w") as f:
            # simple dict conversion
            cookies = {c.name: c.value for c in self.client.cookies.jar}
            json.dump(cookies, f)

    async def login(self, username: str, password: str) -> tuple[bool, str]:
        # 1. Get fnid (if present)
        resp = await self.client.get("/login")
        if resp.status_code != 200:
            return False, "Failed to load login page"

        soup = BeautifulSoup(resp.text, "html.parser")
        fnid_tag = soup.find("input", {"name": "fnid"})
        fnid = fnid_tag["value"] if fnid_tag else None

        # 2. Post credentials
        data = {"acct": username, "pw": password, "goto": "news"}
        if fnid:
            data["fnid"] = fnid

        resp = await self.client.post("/login", data=data)

        if "logout" in resp.text:
            self.username = username
            self._save_cookies()
            return True, "Logged in"

        return False, "Login failed (check credentials)"

    async def _scrape_list(self, url_path: str, max_pages: int = 3) -> set[int]:
        ids = set()
        for page in range(1, max_pages + 1):
            paged_url = (
                f"{url_path}&p={page}" if "?" in url_path else f"{url_path}?p={page}"
            )
            resp = await self.client.get(paged_url)
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.find_all("tr", class_="athing")
            if not rows:
                break

            for r in rows:
                row_id = r.get("id")
                if isinstance(row_id, str) and row_id.isdigit():
                    ids.add(int(row_id))

            if not soup.find("a", class_="morelink"):
                break
        return ids

    def _get_user_cache(self, username: str, cache_type: str) -> set[int] | None:
        """Get cached user data if fresh."""
        cache_path = USER_CACHE_DIR / f"{username}_{cache_type}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                if time.time() - data["ts"] < USER_CACHE_TTL:
                    return set(data["ids"])
            except Exception:
                pass
        return None

    def _set_user_cache(self, username: str, cache_type: str, ids: set[int]) -> None:
        """Save user data to cache."""
        cache_path = USER_CACHE_DIR / f"{username}_{cache_type}.json"
        with open(cache_path, "w") as f:
            json.dump({"ts": time.time(), "ids": list(ids)}, f)

    async def fetch_favorites(self, username: str) -> set[int]:
        if cached := self._get_user_cache(username, "favorites"):
            return cached
        ids = await self._scrape_list(f"/favorites?id={username}")
        self._set_user_cache(username, "favorites", ids)
        return ids

    async def fetch_upvoted(self, username: str) -> set[int]:
        if cached := self._get_user_cache(username, "upvoted"):
            return cached
        ids = await self._scrape_list(f"/upvoted?id={username}")
        self._set_user_cache(username, "upvoted", ids)
        return ids

    async def fetch_hidden(self, username: str) -> set[int]:
        if cached := self._get_user_cache(username, "hidden"):
            return cached
        ids = await self._scrape_list(f"/hidden?id={username}")
        self._set_user_cache(username, "hidden", ids)
        return ids

    async def fetch_submitted(self, username: str) -> set[int]:
        if cached := self._get_user_cache(username, "submitted"):
            return cached
        ids = await self._scrape_list(f"/submitted?id={username}")
        self._set_user_cache(username, "submitted", ids)
        return ids

    async def vote(self, item_id: int, direction: str = "up") -> tuple[bool, str]:
        """
        Vote on a story or comment.
        direction: 'up' or 'down'.
        Note: Stories typically only support 'up'. 'down' is for comments (if high karma).
        """
        # 1. Fetch item page to find auth token
        resp = await self.client.get(f"/item?id={item_id}")
        if resp.status_code != 200:
            return False, "Failed to load item page"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Look for the vote link
        link = soup.find("a", id=f"{direction}_{item_id}")

        if not link:
            # Check if already voted
            unvote_link = soup.find("a", id=f"un_{item_id}")
            if unvote_link:
                return False, "Already voted"

            if "login" in resp.text and "logout" not in resp.text:
                return False, "Not logged in"

            return False, f"Vote link '{direction}' not found"

        href = link.get("href")
        if not href or "auth=" not in href:
            return False, "Auth token not found"

        # 2. Execute Vote
        vote_resp = await self.client.get(f"/{href}")
        if vote_resp.status_code == 200:
            return True, "Voted successfully"

        return False, f"Vote failed: {vote_resp.status_code}"

    async def hide(self, item_id: int) -> tuple[bool, str]:
        """
        Hide a story (negative signal).
        """
        resp = await self.client.get(f"/item?id={item_id}")
        if resp.status_code != 200:
            return False, "Failed to load item page"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find hide link: href starts with hide?id={item_id}
        link = soup.find("a", href=lambda x: x and x.startswith(f"hide?id={item_id}"))

        if not link:
            if "login" in resp.text and "logout" not in resp.text:
                return False, "Not logged in"
            # Check if already hidden? Hard to tell from item page (it might say "unhide"?)
            # Usually if you go to item page of hidden story, you see "unhide" link?
            # unhide link: href starts with unhide?id={item_id}
            unhide = soup.find(
                "a", href=lambda x: x and x.startswith(f"unhide?id={item_id}")
            )
            if unhide:
                return False, "Already hidden"

            return False, "Hide link not found"

        href = link.get("href")

        hide_resp = await self.client.get(f"/{href}")
        if hide_resp.status_code == 200:
            return True, "Hidden successfully"

        return False, f"Hide failed: {hide_resp.status_code}"

    async def check_session(self) -> bool:
        """Verify if cookies are valid."""
        try:
            resp = await self.client.get("/")
            return "logout" in resp.text
        except Exception:
            return False

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "HNClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()
