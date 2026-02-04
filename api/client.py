from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import TypedDict, cast
import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag
from api.constants import USER_CACHE_DIR, USER_CACHE_TTL

logger = logging.getLogger(__name__)

USER_CACHE_DIR_PATH: Path = Path(USER_CACHE_DIR)
USER_CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
COOKIES_FILE: Path = USER_CACHE_DIR_PATH / "cookies.json"


class UserCacheIds(TypedDict):
    pos: list[int]
    upvoted: list[int]
    hidden: list[int]
    hidden_urls: list[str]
    favorites: list[int]


class UserCacheFile(TypedDict):
    ts: float
    ids: UserCacheIds


class UserSignals(TypedDict):
    pos: set[int]
    upvoted: set[int]
    hidden: set[int]
    hidden_urls: set[str]
    favorites: set[int]


class HNClient:
    BASE_URL: str = "https://news.ycombinator.com"

    def __init__(self) -> None:
        self.client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.BASE_URL,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=httpx.Timeout(15.0, connect=10.0),
        )
        self._load_cookies()

    def _load_cookies(self) -> None:
        if COOKIES_FILE.exists():
            try:
                cookies = cast(dict[str, str], json.loads(COOKIES_FILE.read_text()))
                self.client.cookies.update(cookies)
            except Exception:
                pass

    async def login(self, user: str, pw: str) -> tuple[bool, str]:
        resp: httpx.Response = await self.client.get("/login")
        soup: BeautifulSoup = BeautifulSoup(resp.text, "html.parser")
        fnid_tag = soup.find("input", {"name": "fnid"})

        data: dict[str, str] = {"acct": user, "pw": pw}
        if isinstance(fnid_tag, Tag):
            data["fnid"] = str(fnid_tag.get("value", ""))

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
        ids, _ = await self._scrape_items(path, max_pages)
        return ids

    async def _scrape_items(self, path: str, max_pages: int = 10) -> tuple[set[int], set[str]]:
        ids: set[int] = set()
        urls: set[str] = set()
        for p in range(1, max_pages + 1):
            url: str = f"{path}&p={p}" if "?" in path else f"{path}?p={p}"
            resp: httpx.Response = await self.client.get(url)
            soup: BeautifulSoup = BeautifulSoup(resp.text, "html.parser")
            rows: list[Tag] = list(soup.find_all("tr", class_="athing"))
            for r in rows:
                sid_attr = r.get("id")
                if isinstance(sid_attr, list):
                    sid_attr = sid_attr[0] if sid_attr else None
                if not isinstance(sid_attr, str) or not sid_attr.isdigit():
                    continue
                sid = int(sid_attr)
                ids.add(sid)
                
                # Extract URL for duplicate/hidden detection
                title_span = r.find("span", class_="titleline")
                if isinstance(title_span, Tag):
                    a = title_span.find("a")
                    if isinstance(a, Tag):
                        href_val = a.get("href")
                        if isinstance(href_val, str) and not href_val.startswith(
                            "item?id="
                        ):
                            # Normalize: strip query params and trailing slash
                            norm_url = href_val.split("?")[0].rstrip("/")
                            urls.add(norm_url)
            if not soup.find("a", class_="morelink"):
                break
        return ids, urls

    async def fetch_user_submissions(self, username: str) -> set[int]:
        """Fetch submission IDs for a user using the official Firebase API."""
        resp: httpx.Response = await self.client.get(
            f"https://hacker-news.firebaseio.com/v0/user/{username}.json"
        )
        if resp.status_code == 200:
            data = cast(dict[str, object], resp.json())
            submitted = data.get("submitted", [])
            if not isinstance(submitted, list):
                return set()
            out: set[int] = set()
            for item in submitted:
                if isinstance(item, int):
                    out.add(item)
                elif isinstance(item, str) and item.isdigit():
                    out.add(int(item))
            return out
        return set()

    async def fetch_user_data(self, user: str) -> UserSignals:
        cache_path: Path = USER_CACHE_DIR_PATH / f"{user}.json"

        # Check login status first (needed for hidden list)
        resp = await self.client.get("/")
        is_logged_in: bool = "logout" in resp.text
        soup = BeautifulSoup(resp.text, "html.parser")
        me = soup.find("a", id="me")
        logged_in_as = me.text if me else None

        # Always fetch fresh hidden list (changes frequently)
        hidden: set[int] = set()
        hidden_urls: set[str] = set()
        if is_logged_in and logged_in_as == user:
            h_ids, h_urls = await self._scrape_items(f"/hidden?id={user}", max_pages=20)
            hidden = h_ids
            hidden_urls = h_urls

        # Try cache for other signals (favorites, upvotes)
        favorites: set[int] = set()
        upvoted: set[int] = set()
        cache_valid = False

        if cache_path.exists():
            try:
                data = cast(UserCacheFile, json.loads(cache_path.read_text()))
                if time.time() - float(data["ts"]) < USER_CACHE_TTL:
                    cache_valid = True
                    favorites = set(data["ids"].get("favorites", []))
                    upvoted = set(data["ids"].get("upvoted", []))
            except Exception:
                pass

        if not cache_valid:
            # Fetch fresh favorites and upvotes
            fav_ids, _ = await self._scrape_items(f"/favorites?id={user}", max_pages=15)
            favorites = fav_ids

            if is_logged_in and logged_in_as == user:
                upvoted = await self._scrape_ids(f"/upvoted?id={user}", max_pages=15)
            elif is_logged_in:
                logger.warning(f"Logged in as @{logged_in_as}, but requested data for @{user}. Private signals (upvoted/hidden) skipped.")
            else:
                logger.info(f"Not logged in. Only public favorites for @{user} will be fetched.")

        # Combined positive signals: (Favorites | Upvoted) - Hidden
        pos_combined = (favorites | upvoted) - hidden

        # Update cache with all data
        out_ids = {
            "pos": list(pos_combined),
            "upvoted": list(upvoted),
            "hidden": list(hidden),
            "hidden_urls": list(hidden_urls),
            "favorites": list(favorites),
        }

        cache_path.write_text(
            json.dumps({"ts": time.time(), "ids": out_ids})
        )

        return {
            "pos": pos_combined,
            "upvoted": upvoted,
            "hidden": hidden,
            "hidden_urls": hidden_urls,
            "favorites": favorites,
        }

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> HNClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
