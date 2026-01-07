import argparse
import asyncio
from typing import ClassVar
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, ListView, ListItem, Label, LoadingIndicator, ProgressBar, Static

from api import rerank
from api.client import HNClient
from api.fetching import get_best_stories, fetch_story
from api.constants import CANDIDATE_FETCH_COUNT, MAX_USER_STORIES

class StoryItem(ListItem):
    def __init__(self, story, score, reason):
        super().__init__()
        self.story = story
        self.score = score
        self.reason = reason

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="row"):
                yield Label(f"{int(self.score * 100)}%", id="score")
                yield Label(self.story["title"], id="title")
                yield Label(f"{self.story['score']}p", id="points")
            with Vertical(id="details"):
                if self.reason:
                    yield Label(f"Reason: {self.reason}", id="reason")
                for c in self.story["comments"]:
                    yield Static(c[:200], classes="comment")

class HNRerankTUI(App):
    CSS = """
    #row { height: 1; margin-bottom: 1; }
    #score { width: 5; color: cyan; font-weight: bold; }
    #title { width: 1fr; }
    #points { width: 8; text-align: right; color: yellow; }
    #details { display: none; padding: 1; background: #222; }
    StoryItem.-highlight #details { display: block; }
    .comment { border-left: solid gray; padding-left: 1; margin-bottom: 1; color: #aaa; }
    #reason { color: green; font-style: italic; margin-bottom: 1; }
    """
    
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("u", "upvote", "Upvote"),
        Binding("h", "hide", "Hide"),
        Binding("v", "view", "View"),
    ]

    def __init__(self, user):
        super().__init__()
        self.user = user
        self.session_excluded_ids = set()

    def compose(self) -> ComposeResult:
        yield Header()
        yield LoadingIndicator(id="loading")
        yield ProgressBar(id="progress", show_eta=False)
        yield ListView(id="list")
        yield Footer()

    async def on_mount(self): self.action_refresh()

    @work
    async def action_refresh(self):
        self.add_class("loading")
        try:
            prog = self.query_one(ProgressBar)
            prog.update(total=100, progress=0)
            
            async with HNClient() as hn:
                data = await hn.fetch_user_data(self.user)
                prog.update(progress=20)
                
                pos_stories = [s for s in await asyncio.gather(*[fetch_story(hn.client, i) for i in list(data["pos"])[:MAX_USER_STORIES]]) if s]
                neg_stories = [s for s in await asyncio.gather(*[fetch_story(hn.client, i) for i in list(data["neg"])[:MAX_USER_STORIES]]) if s]
                prog.update(progress=40)
            
            p_emb = rerank.get_embeddings([s["text_content"] for s in pos_stories], is_query=True)
            n_emb = rerank.get_embeddings([s["text_content"] for s in neg_stories], is_query=True)
            p_weights = rerank.compute_recency_weights([s["time"] for s in pos_stories])
            prog.update(progress=60)
            
            exclude = data["pos"] | data["neg"] | self.session_excluded_ids
            cands = await get_best_stories(CANDIDATE_FETCH_COUNT, exclude_ids=exclude)
            prog.update(progress=80)
            
            ranked = rerank.rank_stories(cands, p_emb, n_emb, p_weights)
            
            lst = self.query_one("#list", ListView)
            lst.clear()
            for idx, score, fav_idx in ranked[:50]:
                reason = pos_stories[fav_idx]["title"] if fav_idx != -1 and fav_idx < len(pos_stories) else ""
                await lst.append(StoryItem(cands[idx], score, reason))
            prog.update(progress=100)
        finally:
            self.remove_class("loading")

    async def action_upvote(self):
        if item := self.query_one("#list", ListView).highlighted_child:
            sid = item.story["id"]
            self.session_excluded_ids.add(sid)
            self.notify(f"Upvoted {sid}")
            item.remove()

    async def action_hide(self):
        if item := self.query_one("#list", ListView).highlighted_child:
            sid = item.story["id"]
            self.session_excluded_ids.add(sid)
            self.notify(f"Hidden {sid}")
            item.remove()

    def action_view(self):
        if item := self.query_one("#list", ListView).highlighted_child:
            import webbrowser
            webbrowser.open(item.story.get("url") or f"https://news.ycombinator.com/item?id={item.story['id']}")

async def run_batch(user, count):
    async with HNClient() as hn:
        data = await hn.fetch_user_data(user)
        async with hn.client as ac:
            pos = [s for s in await asyncio.gather(*[fetch_story(ac, i) for i in list(data["pos"])[:MAX_USER_STORIES]]) if s]
            neg = [s for s in await asyncio.gather(*[fetch_story(ac, i) for i in list(data["neg"])[:MAX_USER_STORIES]]) if s]
    
    p_emb = rerank.get_embeddings([s["text_content"] for s in pos], is_query=True)
    n_emb = rerank.get_embeddings([s["text_content"] for s in neg], is_query=True)
    p_weights = rerank.compute_recency_weights([s["time"] for s in pos])
    
    cands = await get_best_stories(CANDIDATE_FETCH_COUNT, exclude_ids=data["pos"] | data["neg"])
    ranked = rerank.rank_stories(cands, p_emb, n_emb, p_weights)
    
    for idx, score, fav_idx in ranked[:count]:
        s = cands[idx]
        print(f"{int(score*100)}% {s['title']} ({s['url']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    
    if args.batch:
        asyncio.run(run_batch(args.user, args.count))
    else:
        HNRerankTUI(args.user).run()
