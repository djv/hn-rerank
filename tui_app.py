from __future__ import annotations
import numpy as np
import asyncio
import webbrowser
import argparse
import traceback
import os
import time
from typing import Optional, cast

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, ListView, ListItem, Label, Static, Button, LoadingIndicator
from textual.reactive import reactive
from textual.binding import Binding
from textual import work, on

from api.fetching import get_user_data, get_best_stories
from api import rerank
from api.client import HNClient
from api.config import save_config, get_username

# --- CREDENTIALS (Env Vars) ---
HN_USER = os.getenv("HN_USERNAME")
HN_PASS = os.getenv("HN_PASSWORD")


def get_relative_time(timestamp: Optional[int]) -> str:
    if not timestamp:
        return ""
    diff = int(time.time()) - timestamp
    if diff < 60:
        return "now"
    elif diff < 3600:
        return f"{diff // 60}m"
    elif diff < 86400:
        return f"{diff // 3600}h"
    else:
        return f"{diff // 86400}d"


class StoryItem(ListItem):
    expanded = reactive(False)
    vote_status: reactive[Optional[str]] = reactive(None)  # 'up', 'down', None

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
                host = (
                    self.story.get("url", "hn").split("/")[2]
                    if self.story.get("url") and "//" in self.story["url"]
                    else "hn"
                )
                yield Label(host, id="hostname")
                yield Label(
                    get_relative_time(self.story.get("time", 0)), id="time"
                )
                yield Label("", id="vote-icon")

            with Vertical(id="details"):
                if self.story.get("article_snippet"):
                    yield Label("[bold blue]ARTICLE CONTENT[/]", classes="section-title")
                    yield Static(self.story["article_snippet"], classes="reason")
                
                if self.story.get("comments"):
                    yield Label(
                        "[bold underline]TOP DISCUSSION[/]", classes="section-title"
                    )
                    for c in self.story["comments"][:5]:
                        yield Static(f"💬 {c}\n", classes="comment")

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
    #loading {
        display: none;
        height: 1fr;
        content-align: center middle;
    }
    App.loading #loading {
        display: block;
    }
    App.loading #story-list {
        display: none;
    }
    #story-list { 
        height: 1fr; 
        background: transparent; 
        padding: 1;
        border: none;
    }
    StoryItem {
        padding: 0 1;
        margin: 0;
        border: none;
        height: 1;
    }
    StoryItem:focus {
        background: $accent-muted;
    }
    StoryItem.is-expanded { 
        height: auto; 
        background: $surface; 
        border: tall $primary; 
        margin: 1 0; 
    }
    StoryItem #row { height: 1; }
    StoryItem #match-score { width: 5; text-style: bold; color: $accent; }
    StoryItem #story-title { width: 1fr; text-style: bold; }
    StoryItem #hostname { width: 15; color: $text-muted; text-align: right; }
    StoryItem #time { width: 6; color: $text-muted; text-align: right; }
    StoryItem #vote-icon { width: 2; }
    
    StoryItem #details { display: none; padding: 1 2; }
    StoryItem.is-expanded #details { display: block; }
    .meta { color: $text-muted; }
    .reason { background: $primary-muted; padding: 1; border-left: solid $primary; }
    .comment { 
        color: $text; 
        background: $surface; 
        padding: 0 1; 
        margin-bottom: 1; 
        border-left: solid $accent-muted;
    }
    .section-title { margin-top: 1; color: $accent; }
    .actions { height: 3; margin-top: 1; }
    .actions Button { height: 1; border: none; }
    """

    BINDINGS = [
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

    def __init__(self, username: Optional[str]):
        super().__init__()
        self.username = username
        self._pending_actions = set()

    def compose(self) -> ComposeResult:
        yield Header()
        yield LoadingIndicator(id="loading")
        yield ListView(id="story-list")
        yield Footer()

    async def on_mount(self):
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
            for child in self.query_one("#story-list", ListView).children:
                if isinstance(child, StoryItem) and child != event.item:
                    child.expanded = False
            
            # Auto-expand the highlighted one
            event.item.expanded = True
            # Ensure expanded content is visible
            self.query_one("#story-list", ListView).scroll_to_widget(event.item)

    @work
    async def refresh_feed(self):
        if not self.username:
            self.username = "pg"

        self.add_class("loading")
        self.notify(f"Syncing Intelligence for {self.username}...")

        try:
            rerank.init_model("onnx_model")
            self.notify("Model Loaded. Fetching user data...")
            pos_data, neg_data, exclude_ids = await get_user_data(self.username)
            
            self.notify(f"Got {len(pos_data)} signals. Fetching candidates...")
            candidates = await get_best_stories(200, 30, exclude_ids)
            
            if not candidates:
                self.notify("No new stories found", severity="warning")
                self.remove_class("loading")
                return

            self.notify(f"Ranking {len(candidates)} stories...")
            def do_rank():
                p_texts = [s["text_content"] for s in pos_data]
                n_texts = [s["text_content"] for s in neg_data]
                c_texts = [s["text_content"] for s in candidates]

                p_weights = np.linspace(1.0, 0.5, len(p_texts)).astype(np.float32) if p_texts else None

                p_emb = rerank.get_embeddings(p_texts, is_query=True)
                n_emb = rerank.get_embeddings(n_texts, is_query=True)
                c_emb = rerank.get_embeddings(c_texts, is_query=False)

                return rerank.rank_stories(
                    candidates,
                    cand_embeddings=c_emb,
                    positive_embeddings=p_emb,
                    negative_embeddings=n_emb,
                    positive_weights=p_weights,
                    hn_weight=0.15,
                )

            ranked = await asyncio.to_thread(do_rank)

            list_view = self.query_one("#story-list", ListView)
            list_view.clear()

            for idx, score, fav_idx in ranked[:50]:
                story = candidates[idx]
                await list_view.append(StoryItem(story, score, "", ""))

            if list_view.children:
                list_view.index = 0
                list_view.focus()
                
                # Use a small delay to ensure reactivity is ready
                def expand_first():
                    if list_view.children:
                        list_view.children[0].expanded = True
                self.set_timer(0.2, expand_first)

            self.notify("Feed Ready")
        except Exception as e:
            traceback.print_exc()
            self.notify(f"Sync error: {e}", severity="error")
        finally:
            self.remove_class("loading")

    def _get_current(self) -> Optional[StoryItem]:
        return cast(Optional[StoryItem], self.query_one("#story-list", ListView).highlighted_child)

    def action_toggle_expand(self):
        if item := self._get_current():
            item.expanded = not item.expanded
            if item.expanded:
                self.query_one("#story-list", ListView).scroll_to_widget(item)

    def _open_url(self, url: str):
        if "SSH_CONNECTION" in os.environ:
            self.app.copy_to_clipboard(url)
            self.notify("Remote session: URL copied to local clipboard")
        else:
            self.notify("Opening in browser...")
            webbrowser.open_new_tab(url)

    def action_open_article(self):
        if item := self._get_current():
            url = (
                item.story.get("url")
                or f"https://news.ycombinator.com/item?id={item.story['id']}"
            )
            self._open_url(url)

    def action_open_hn(self):
        if item := self._get_current():
            url = f"https://news.ycombinator.com/item?id={item.story['id']}"
            self._open_url(url)

    @work
    async def action_upvote(self):
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
            finally:
                self._pending_actions.remove(sid)

    @work
    async def action_hide(self):
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
                    self.notify("Story hidden")
            finally:
                self._pending_actions.remove(sid)

    def action_refresh_feed(self):
        self.refresh_feed()


def main():
    """Entry point for hn-rerank CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("username", nargs="?", default=get_username())
    args = parser.parse_args()
    app = HNRerankTUI(args.username)
    app.run()


if __name__ == "__main__":
    main()
