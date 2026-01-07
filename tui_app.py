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
    #progress-container { display: none; padding: 1; border-top: solid $primary; }
    App.loading #progress-container { display: block; }
    App.loading #loading { display: block; }
    App.loading #list { display: none; }
    
    #task-label { color: $accent; text-style: italic; margin-top: 1; }
    #total-label { color: $primary; text-style: bold; }
    
    #row { height: 1; margin-bottom: 1; }
    #score { width: 5; color: cyan; text-style: bold; }
    #title { width: 1fr; }
    #points { width: 8; text-align: right; color: yellow; }
    #details { display: none; padding: 1; background: #222; }
    StoryItem.-highlight #details { display: block; }
    .comment { border-left: solid gray; padding-left: 1; margin-bottom: 1; color: #aaa; }
    #reason { color: green; text-style: italic; margin-bottom: 1; }
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
        with Vertical(id="progress-container"):
            yield Label("Total Progress", id="total-label")
            yield ProgressBar(id="total-progress", show_eta=False)
            yield Label("Current Task", id="task-label")
            yield ProgressBar(id="task-progress", show_eta=False)
        yield ListView(id="list")
        yield Footer()

    async def on_mount(self): self.action_refresh()

    async def _batch_fetch_stories(self, ac, ids, label):
        results = []
        task_prog = self.query_one("#task-progress", ProgressBar)
        self.query_one("#task-label", Label).update(label)
        task_prog.update(total=len(ids), progress=0)
        
        # Concurrency controlled by fetch_story semaphore
        for i, sid in enumerate(ids):
            res = await fetch_story(ac, sid)
            if res: results.append(res)
            task_prog.update(progress=i+1)
        return results

    @work
    async def action_refresh(self):
        self.add_class("loading")
        try:
            total_prog = self.query_one("#total-progress", ProgressBar)
            task_prog = self.query_one("#task-progress", ProgressBar)
            total_prog.update(total=100, progress=0)
            
            # 1. User Data
            self.query_one("#task-label", Label).update("Fetching User Profile...")
            async with HNClient() as hn:
                data = await hn.fetch_user_data(self.user)
                total_prog.update(progress=10)
                
                # 2. Positive Signal Details
                pos_stories = await self._batch_fetch_stories(
                    hn.client, list(data["pos"])[:MAX_USER_STORIES], "Fetching Positive Signals..."
                )
                total_prog.update(progress=25)
                
                # 3. Negative Signal Details
                neg_stories = await self._batch_fetch_stories(
                    hn.client, list(data["neg"])[:MAX_USER_STORIES], "Fetching Negative Signals..."
                )
                total_prog.update(progress=40)
            
            # 4. Embed Signals
            self.query_one("#task-label", Label).update("Embedding Preferences...")
            task_prog.update(total=100, progress=0)
            def emb_cb(curr, total): task_prog.update(total=total, progress=curr)
            
            p_emb = rerank.get_embeddings([s["text_content"] for s in pos_stories], is_query=True, progress_callback=emb_cb)
            n_emb = rerank.get_embeddings([s["text_content"] for s in neg_stories], is_query=True, progress_callback=emb_cb)
            p_weights = rerank.compute_recency_weights([s["time"] for s in pos_stories])
            total_prog.update(progress=60)
            
            # 5. Fetch Candidates
            self.query_one("#task-label", Label).update("Fetching Candidates...")
            exclude = data["pos"] | data["neg"] | self.session_excluded_ids
            cands = await get_best_stories(
                CANDIDATE_FETCH_COUNT, 
                exclude_ids=exclude,
                progress_callback=lambda curr, tot: task_prog.update(total=tot, progress=curr)
            )
            total_prog.update(progress=80)
            
            # 6. Rank
            self.query_one("#task-label", Label).update("Ranking Stories...")
            # rank_stories will call get_embeddings for candidates
            ranked = rerank.rank_stories(cands, p_emb, n_emb, p_weights, progress_callback=emb_cb)
            
            lst = self.query_one("#list", ListView)
            lst.clear()
            for idx, score, fav_idx in ranked[:50]:
                reason = pos_stories[fav_idx]["title"] if fav_idx != -1 and fav_idx < len(pos_stories) else ""
                await lst.append(StoryItem(cands[idx], score, reason))
            total_prog.update(progress=100)
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
        pos = [s for s in await asyncio.gather(*[fetch_story(hn.client, i) for i in list(data["pos"])[:MAX_USER_STORIES]]) if s]
        neg = [s for s in await asyncio.gather(*[fetch_story(hn.client, i) for i in list(data["neg"])[:MAX_USER_STORIES]]) if s]
    
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
