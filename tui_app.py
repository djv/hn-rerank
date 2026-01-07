from __future__ import annotations

import argparse
import asyncio
import os
import time
import traceback
import webbrowser
from typing import ClassVar, cast

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    ProgressBar,
    Static,
)

from api import rerank
from api.client import HNClient
from api.config import get_username, save_config
from api.constants import CANDIDATE_FETCH_COUNT
from api.fetching import get_best_stories, get_user_data

# --- CREDENTIALS (Env Vars) ---
HN_USER = os.getenv("HN_USERNAME")
HN_PASS = os.getenv("HN_PASSWORD")

# Time constants
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400
_COMMENT_TRUNCATE_LENGTH = 300


def get_relative_time(timestamp: int | None) -> str:
    if not timestamp:
        return ""
    diff = int(time.time()) - timestamp
    if diff < _SECONDS_PER_MINUTE:
        return "now"
    elif diff < _SECONDS_PER_HOUR:
        return f"{diff // _SECONDS_PER_MINUTE}m"
    elif diff < _SECONDS_PER_DAY:
        return f"{diff // _SECONDS_PER_HOUR}h"
    else:
        return f"{diff // _SECONDS_PER_DAY}d"


class StoryItem(ListItem):
    expanded = reactive(False)
    vote_status: reactive[str | None] = reactive(None)  # 'up', 'down', None

    def __init__(self, story: dict, score: float, reason: str, rel_title: str):
        super().__init__()
        self.story = story
        self.score_val = score
        self.reason = reason
        self.rel_title = rel_title

    def compose(self) -> ComposeResult:
        with Vertical(id="wrapper"):
            with Horizontal(id="row"):
                yield Label(f"{int(self.score_val * 100)}%", id="match-score")
                yield Label(
                    f"[@click=app.open_article]{self.story['title']}[/]",
                    id="story-title",
                )
                yield Label(f"{self.story.get('score', 0)}p", id="hn-score")
                yield Label(
                    get_relative_time(self.story.get("time", 0)), id="time"
                )
                yield Label("", id="vote-icon")

            with Vertical(id="details"):
                if self.reason:
                    yield Label(f"[bold]{self.reason}[/]", classes="reason")
                hn_url = f"https://news.ycombinator.com/item?id={self.story['id']}"
                yield Label(f"[dim]{hn_url}[/dim]", id="hn-link")

                if self.story.get("comments"):
                    for c in self.story["comments"][:8]:
                        # Truncate long comments
                        if len(c) > _COMMENT_TRUNCATE_LENGTH:
                            text = c[:_COMMENT_TRUNCATE_LENGTH] + "..."
                        else:
                            text = c
                        yield Static(text, classes="comment")

                with Horizontal(classes="actions"):
                    yield Button("Article (v)", id="open-art", variant="primary")
                    yield Button("Comments (c)", id="open-hn")
                    yield Button("Upvote (u)", id="upvote", variant="success")
                    yield Button("Hide (d)", id="hide", variant="error")

    @on(Button.Pressed, "#open-art")
    def on_open_art(self):
        url = (
            self.story.get("url")
            or f"https://news.ycombinator.com/item?id={self.story['id']}"
        )
        self.app.notify("Opening article...")
        webbrowser.open_new_tab(url)

    @on(Button.Pressed, "#open-hn")
    def on_open_hn(self):
        url = f"https://news.ycombinator.com/item?id={self.story['id']}"
        self.app.notify("Opening comments...")
        webbrowser.open_new_tab(url)

    @on(Button.Pressed, "#upvote")
    def on_upvote(self):
        cast("HNRerankTUI", self.app).action_upvote()

    @on(Button.Pressed, "#hide")
    def on_hide(self):
        cast("HNRerankTUI", self.app).action_hide()

    def watch_expanded(self, val: bool):
        self.set_class(val, "is-expanded")

    def watch_vote_status(self, val: str):
        try:
            icon = self.query_one("#vote-icon", Static)
            if val == "up":
                icon.update("[bold green]↑[/]")
            elif val == "down":
                icon.update("[bold red]↓[/]")
        except Exception:
            pass


class HNRerankTUI(App):
    CSS = """
    Screen { background: $surface; }
    #loading { display: none; height: 1fr; content-align: center middle; }
    #progress { display: none; dock: bottom; height: 1; margin: 0 0 1 0; }
    App.loading #loading { display: block; }
    App.loading #progress { display: block; }
    App.loading #story-list { display: none; }
    #story-list { height: 1fr; background: transparent; padding: 1; border: none; }
    StoryItem { padding: 0 1; margin: 0; border: none; height: 1; }
    StoryItem:focus { background: $accent-muted; }
    StoryItem.is-expanded {
        height: auto; max-height: 70vh; background: $surface;
        border: tall $primary; margin: 0;
    }
    StoryItem #row { height: 1; }
    StoryItem #match-score { width: 5; text-style: bold; color: $accent; }
    StoryItem #story-title { width: 1fr; text-style: bold; }
    StoryItem #hn-score { width: 7; color: $text-muted; text-align: right; }
    StoryItem #time { width: 6; color: $text-muted; text-align: right; }
    StoryItem #vote-icon { width: 2; }
    StoryItem #details { display: none; padding: 1 2; }
    StoryItem.is-expanded #details {
        display: block; height: auto; max-height: 60vh; overflow-y: auto;
    }
    .meta { color: $text-muted; }
    .reason { background: $primary-muted; padding: 1; border-left: solid $primary; }
    .comment {
        color: $text; background: $surface; padding: 0 1;
        margin-bottom: 1; border-left: solid $accent-muted;
    }
    .section-title { margin-top: 1; color: $accent; }
    .actions { height: 3; margin-top: 1; }
    .actions Button { height: 1; border: none; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "toggle_expand", "Expand"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("v", "open_article", "View"),
        Binding("c", "open_hn", "Comments"),
        Binding("u", "upvote", "Up"),
        Binding("d", "hide", "Hide"),
        Binding("r", "refresh_feed", "Refresh"),
    ]

    def __init__(self, username: str | None):
        super().__init__()
        self.username = username
        self._pending_actions = set()
        self.session_excluded_ids = set()

    def compose(self) -> ComposeResult:
        yield Header()
        yield LoadingIndicator(id="loading")
        yield ProgressBar(id="progress", total=100, show_eta=False)
        yield ListView(id="story-list")
        yield Footer()

    async def on_mount(self) -> None:
        # Auto-login if creds available
        if HN_USER and HN_PASS:
            self.notify(f"Authenticating as {HN_USER}...")
            async with HNClient() as client:
                success, msg = await client.login(HN_USER, HN_PASS)

            if success:
                self.username = HN_USER
                save_config("username", HN_USER)
                self.notify("Authenticated")
            else:
                self.notify(f"Auth failed: {msg}", severity="error")

        if not self.username:
            self.notify(
                "No username provided. Results will be generic.", severity="warning"
            )

        self.refresh_feed()

    @on(ListView.Selected)
    def on_item_selected(self):
        self.action_toggle_expand()

    @on(ListView.Highlighted)
    def on_highlight_changed(self, event: ListView.Highlighted):
        if event.item and isinstance(event.item, StoryItem):
            # Collapse all others to keep view clean during scroll
            try:
                list_view = self.query_one("#story-list", ListView)
            except Exception:
                return
            for child in list_view.children:
                if isinstance(child, StoryItem) and child != event.item:
                    child.expanded = False

            # Auto-expand the highlighted one
            event.item.expanded = True
            # Ensure expanded content is visible
            self.query_one("#story-list", ListView).scroll_to_widget(event.item)

    @work
    async def refresh_feed(self) -> None:  # noqa: PLR0915
        if not self.username:
            self.username = "pg"

        self.add_class("loading")

        progress = self.query_one("#progress", ProgressBar)
        progress.display = True
        progress.update(total=100)
        progress.progress = 0

        self.notify("Loading...")

        try:
            # Load model in background thread
            await asyncio.to_thread(rerank.init_model, "onnx_model")
            progress.progress = 10

            self.notify(f"Fetching {self.username} preferences...")
            pos_data, neg_data, exclude_ids = await get_user_data(self.username)
            progress.progress = 30

            self.notify("Fetching stories...")
            all_exclude = exclude_ids | self.session_excluded_ids

            def fetch_progress(fetched: int, total: int):
                # Map 0-100% of fetching to 30-70% of total progress
                percent = 30 + int((fetched / total) * 40)
                progress.update(progress=percent)

            candidates = await get_best_stories(
                CANDIDATE_FETCH_COUNT, 30, all_exclude, progress_callback=fetch_progress
            )
            progress.progress = 70

            if not candidates:
                self.notify("No new stories", severity="warning")
                self.remove_class("loading")
                progress.display = False
                return

            self.notify(f"Ranking {len(candidates)}...")

            def do_rank() -> list[tuple[int, float, int]]:
                try:
                    p_texts = [s["text_content"] for s in pos_data]
                    n_texts = [s["text_content"] for s in neg_data]
                    c_texts = [s["text_content"] for s in candidates]

                    p_timestamps = [s.get("time", 0) for s in pos_data]
                    p_weights = (
                        rerank.compute_recency_weights(p_timestamps)
                        if p_timestamps
                        else None
                    )

                    # Progress: 70-95% embeddings, 95-100% rank
                    last_pct = 70

                    def update_progress(pct: int) -> None:
                        nonlocal last_pct
                        if pct > last_pct:
                            last_pct = pct
                            self.call_from_thread(lambda p=pct: progress.update(progress=p))

                    total_texts = len(p_texts) + len(n_texts) + len(c_texts)
                    done = 0

                    def emb_progress(current: int, total: int) -> None:
                        # current/total is for the current get_embeddings call
                        pct = 70 + int(((done + current) / max(total_texts, 1)) * 25)
                        update_progress(min(pct, 95))

                    p_emb = rerank.get_embeddings(
                        p_texts, is_query=True, progress_callback=emb_progress
                    )
                    done += len(p_texts)
                    update_progress(70 + int((done / max(total_texts, 1)) * 25))

                    n_emb = rerank.get_embeddings(
                        n_texts, is_query=True, progress_callback=emb_progress
                    )
                    done += len(n_texts)
                    update_progress(70 + int((done / max(total_texts, 1)) * 25))

                    c_emb = rerank.get_embeddings(
                        c_texts, is_query=False, progress_callback=emb_progress
                    )
                    update_progress(95)

                    # Final ranking (fast, in-memory)
                    results = rerank.rank_stories(
                        candidates,
                        cand_embeddings=c_emb,
                        positive_embeddings=p_emb,
                        negative_embeddings=n_emb,
                        positive_weights=p_weights,
                    )
                    update_progress(100)
                    return results
                except Exception as e:
                    # Log the error to console/file if possible, or re-raise
                    # In a thread, printing might not show up well in TUI.
                    # We re-raise so asyncio.to_thread catches it.
                    raise e

            progress.progress = 70
            ranked = await asyncio.to_thread(do_rank)

            list_view = self.query_one("#story-list", ListView)
            list_view.clear()

            for idx, score, fav_idx in ranked[:50]:
                story = candidates[idx]
                reason = ""
                if fav_idx != -1 and fav_idx < len(pos_data):
                    reason = f"Matches your interest in: {pos_data[fav_idx]['title']}"
                await list_view.append(StoryItem(story, score, reason, ""))

            if list_view.children:
                list_view.index = 0
                list_view.focus()

                # Use a small delay to ensure reactivity is ready
                def expand_first():
                    if list_view.children:
                        cast(StoryItem, list_view.children[0]).expanded = True
                self.set_timer(0.2, expand_first)

            self.notify("Feed Ready")
        except Exception as e:
            traceback.print_exc()
            self.notify(f"Sync error: {e}", severity="error")
        finally:
            self.remove_class("loading")
            progress.display = False

    def _get_current(self) -> StoryItem | None:
        lv = self.query_one("#story-list", ListView)
        return cast(StoryItem | None, lv.highlighted_child)

    def action_toggle_expand(self) -> None:
        if item := self._get_current():
            item.expanded = not item.expanded
            if item.expanded:
                self.query_one("#story-list", ListView).scroll_to_widget(item)

    def _open_url(self, url: str) -> None:
        if "SSH_CONNECTION" in os.environ:
            self.app.copy_to_clipboard(url)
            self.notify("Remote session: URL copied to local clipboard")
        else:
            self.notify("Opening in browser...")
            webbrowser.open_new_tab(url)

    def action_open_article(self) -> None:
        if item := self._get_current():
            url = (
                item.story.get("url")
                or f"https://news.ycombinator.com/item?id={item.story['id']}"
            )
            self._open_url(url)

    def action_open_hn(self) -> None:
        if item := self._get_current():
            url = f"https://news.ycombinator.com/item?id={item.story['id']}"
            self._open_url(url)

    @work
    async def action_upvote(self) -> None:
        if item := self._get_current():
            sid = item.story["id"]
            if sid in self._pending_actions:
                return
            self._pending_actions.add(sid)
            try:
                async with HNClient() as client:
                    success, _ = await client.vote(sid, "up")
                if success:
                    item.vote_status = "up"
                    self.session_excluded_ids.add(sid)
                    self.notify("Upvoted")
                    item.remove()
            finally:
                self._pending_actions.remove(sid)

    @work
    async def action_hide(self) -> None:
        if item := self._get_current():
            sid = item.story["id"]
            if sid in self._pending_actions:
                return
            self._pending_actions.add(sid)
            try:
                async with HNClient() as client:
                    success, _ = await client.hide(sid)
                if success:
                    item.vote_status = "down"
                    self.session_excluded_ids.add(sid)
                    self.notify("Story hidden")
                    item.remove()
            finally:
                self._pending_actions.remove(sid)

    def action_refresh_feed(self) -> None:
        self.refresh_feed()


async def run_batch(username: str, count: int, fmt: str) -> None:
    """Run in batch mode without TUI."""
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if not username:
        username = "pg"

    print("Loading model...", file=sys.stderr)
    rerank.init_model("onnx_model")

    print(f"Fetching user data for {username}...", file=sys.stderr)
    pos_data, neg_data, exclude_ids = await get_user_data(username)

    print("Fetching candidates...", file=sys.stderr)
    candidates = await get_best_stories(CANDIDATE_FETCH_COUNT, 30, exclude_ids)

    if not candidates:
        print("No stories found", file=sys.stderr)
        return

    print(f"Ranking {len(candidates)} stories...", file=sys.stderr)

    def do_rank() -> list[tuple[int, float, int]]:
        p_texts = [s["text_content"] for s in pos_data]
        n_texts = [s["text_content"] for s in neg_data]
        c_texts = [s["text_content"] for s in candidates]

        p_timestamps = [s.get("time", 0) for s in pos_data]
        p_weights = (
            rerank.compute_recency_weights(p_timestamps)
            if p_timestamps
            else None
        )

        p_emb = rerank.get_embeddings(p_texts, is_query=True)
        n_emb = rerank.get_embeddings(n_texts, is_query=True)
        c_emb = rerank.get_embeddings(c_texts, is_query=False)

        return rerank.rank_stories(
            candidates,
            cand_embeddings=c_emb,
            positive_embeddings=p_emb,
            negative_embeddings=n_emb,
            positive_weights=p_weights,
        )

    ranked = await asyncio.to_thread(do_rank)

    results = []
    for idx, score, fav_idx in ranked[:count]:
        story = candidates[idx]
        reason = ""
        if fav_idx != -1 and fav_idx < len(pos_data):
            reason = f"Matches: {pos_data[fav_idx]['title']}"

        results.append({
            "id": story["id"],
            "title": story["title"],
            "url": story.get("url") or f"https://news.ycombinator.com/item?id={story['id']}",
            "hn_url": f"https://news.ycombinator.com/item?id={story['id']}",
            "score": round(score * 100),
            "points": story.get("score", 0),
            "comments": story.get("descendants", 0),
            "reason": reason,
        })

    if fmt == "json":
        print(json.dumps(results, indent=2))
    elif fmt == "urls":
        for r in results:
            print(r["url"])
    else:  # text
        for i, r in enumerate(results, 1):
            print(f"{i:2}. [{r['score']:3}%] {r['title']}")
            if r["reason"]:
                print(f"    {r['reason']}")
            print(f"    {r['url']}")
            print(f"    {r['hn_url']}")
            print()


def main():
    """Entry point for hn-rerank CLI."""
    parser = argparse.ArgumentParser(description="HN Reranker")
    parser.add_argument("username", nargs="?", default=get_username())
    parser.add_argument("-b", "--batch", action="store_true", help="Batch mode")
    parser.add_argument("-n", "--count", type=int, default=20, help="Results count")
    parser.add_argument(
        "-f", "--format", choices=["text", "json", "urls"], default="text"
    )
    args = parser.parse_args()

    if args.batch:
        asyncio.run(run_batch(args.username, args.count, args.format))
    else:
        app = HNRerankTUI(args.username)
        app.run()


if __name__ == "__main__":
    main()
