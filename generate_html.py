from __future__ import annotations
import argparse
import asyncio
import getpass
import html
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from types import ModuleType
from importlib import import_module
from importlib.util import find_spec

import numpy as np
from numpy.typing import NDArray
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.console import Console

from sklearn.metrics.pairwise import cosine_similarity

from api import rerank
from api.client import HNClient, UserSignals
from api.fetching import get_best_stories, fetch_story
from api.models import RankResult, Story, StoryDict, StoryDisplay
from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    CANDIDATE_FETCH_COUNT,
    CLUSTER_SIMILARITY_THRESHOLD,
    DEFAULT_CLUSTER_COUNT,
    KNN_NEIGHBORS,
    MAX_CLUSTERS,
    MAX_USER_STORIES,
)

_tomllib: ModuleType | None = None
if find_spec("tomllib") is not None:  # pragma: no cover - Python 3.11+ only
    _tomllib = import_module("tomllib")

tomllib: ModuleType | None = _tomllib

console: Console = Console()

DEFAULT_CONFIG_PATH = Path("hn_rerank.toml")


def _find_config_path(argv: list[str]) -> Optional[Path]:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return None


def _load_config(argv: list[str]) -> tuple[dict[str, object], Optional[Path]]:
    config_path = _find_config_path(argv)
    explicit = config_path is not None
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None

    if config_path is None:
        return {}, None
    if not config_path.exists():
        if explicit:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return {}, None
    if tomllib is None:
        raise RuntimeError("tomllib not available; cannot load config.")

    raw = tomllib.loads(config_path.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("hn_rerank"), dict):
        raw = raw["hn_rerank"]

    if not isinstance(raw, dict):
        return {}, config_path

    allowed_types: dict[str, type] = {
        "username": str,
        "output": str,
        "count": int,
        "signals": int,
        "candidates": int,
        "clusters": int,
        "days": int,
        "use_hidden_signal": bool,
        "use_classifier": bool,
        "contrastive": bool,
        "knn": int,
        "no_naming": bool,
        "no_rss": bool,
        "no_tldr": bool,
        "debug_scores": bool,
        "debug_scores_path": str,
        "debug_clusters": bool,
        "debug_clusters_path": str,
    }
    config: dict[str, object] = {}
    for key, value in raw.items():
        expected_type = allowed_types.get(key)
        if expected_type is None:
            continue
        if expected_type is int and isinstance(value, bool):
            continue
        if isinstance(value, expected_type):
            config[key] = value
    return config, config_path

HTML_TEMPLATE: str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HN Rerank | {username}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        hn: '#ff6600',
                    }}
                }}
            }}
        }}
    </script>
    <style type="text/tailwindcss">
        @layer base {{
            body {{ @apply bg-stone-50 text-stone-800 antialiased; }}
        }}
        .story-card {{ @apply bg-white border border-stone-200 rounded-lg p-2.5 shadow-sm transition-all hover:border-hn hover:shadow-md h-full flex flex-col; }}
        .story-card.rss-story {{ @apply border-amber-200 bg-amber-50/50; }}
        .rss-badge {{ @apply px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 text-[10px] font-bold; }}
        .cluster-chip {{ @apply px-1.5 py-0.5 bg-stone-50 border border-stone-200 rounded text-[10px] font-medium text-stone-600 hover:border-hn hover:text-hn transition-colors cursor-default whitespace-nowrap; }}
    </style>
</head>
<body class="p-2 md:p-4 bg-stone-100">
    <div class="max-w-7xl mx-auto">
        <header class="mb-4 border-b border-stone-300 pb-3 flex items-end justify-between bg-white p-4 rounded-lg shadow-sm">
            <div>
                <h1 class="text-2xl font-black text-stone-900 tracking-tight">
                    HN <span class="text-hn">Rerank</span>
                </h1>
                <p class="text-stone-500 text-xs">@{username} &bull; <a href="clusters.html" class="text-hn hover:underline">{n_clusters} interest clusters</a></p>
            </div>
            <p class="text-[10px] text-stone-400 font-mono">{timestamp}</p>
        </header>

        <div class="grid gap-3 items-start grid-cols-[repeat(auto-fit,minmax(280px,1fr))]">
            {stories_html}
        </div>

        <footer class="mt-8 py-4 border-t border-stone-200 text-center text-stone-400 text-xs">
            HN Rerank &bull; Local Semantic Analysis
        </footer>
    </div>
</body>
</html>
"""

CLUSTERS_PAGE_TEMPLATE: str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interest Clusters | {username}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        hn: '#ff6600',
                    }}
                }}
            }}
        }}
    </script>
    <style type="text/tailwindcss">
        @layer base {{
            body {{ @apply bg-stone-50 text-stone-800 antialiased; }}
        }}
        .cluster-card {{ @apply bg-white border border-stone-200 rounded-lg shadow-sm; }}
    </style>
</head>
<body class="p-2 md:p-4">
    <div class="max-w-5xl mx-auto">
        <header class="mb-4 border-b border-stone-200 pb-3 flex items-end justify-between">
            <div>
                <h1 class="text-2xl font-black text-stone-900 tracking-tight">
                    Interest <span class="text-hn">Clusters</span>
                </h1>
                <p class="text-stone-500 text-xs">@{username} &bull; {n_signals} signals &rarr; {n_clusters} clusters</p>
            </div>
            <p class="text-[10px] text-stone-400 font-mono">{timestamp}</p>
        </header>

        <div class="grid gap-4 md:grid-cols-2">
            {clusters_html}
        </div>

        <footer class="mt-8 py-4 border-t border-stone-200 text-center text-stone-400 text-xs">
            HN Rerank &bull; Multi-Interest Clustering
        </footer>
    </div>
</body>
</html>
"""

CLUSTER_CARD_TEMPLATE: str = """
<div class="cluster-card">
    <div class="px-3 py-2 border-b border-stone-100 flex items-center justify-between">
        <h2 class="font-bold text-stone-700">{cluster_name}</h2>
        <span class="text-xs text-stone-400">{count} stories</span>
    </div>
    <ul class="divide-y divide-stone-100">
        {stories_html}
    </ul>
</div>
"""

CLUSTER_STORY_TEMPLATE: str = """
<li class="px-3 py-2 hover:bg-stone-50">
    <a href="{hn_url}" target="_blank" class="text-sm text-stone-700 hover:text-hn transition-colors line-clamp-2">
        {title}
    </a>
    <div class="flex items-center gap-2 mt-0.5">
        <span class="text-[10px] text-stone-400">{points} pts</span>
        <span class="text-[10px] text-stone-400">{time_ago}</span>
    </div>
</li>
"""


def get_relative_time(timestamp: int) -> str:
    if not timestamp:
        return ""
    diff: int = int(time.time()) - timestamp
    if diff < 60:
        return "now"
    elif diff < 3600:
        return f"{diff // 60}m"
    elif diff < 86400:
        return f"{diff // 3600}h"
    else:
        return f"{diff // 86400}d"


STORY_CARD_TEMPLATE: str = """
<div class="story-card group{rss_class}">
    <div class="flex items-center gap-2 mb-0.5 flex-wrap">
        <span class="px-1.5 py-0.5 rounded bg-hn/10 text-hn text-[10px] font-bold">
            {score}%
        </span>
        {rss_badge}
        {cluster_chip}
        <span class="text-[10px] text-stone-400 font-mono">{points} pts</span>
        <span class="text-[10px] text-stone-400 font-mono">{time_ago}</span>
    </div>
    <h2 class="text-sm font-semibold text-stone-900 leading-snug mb-1">
        <a href="{url}" target="_blank" class="hover:text-hn transition-colors">{title}</a>
        {comment_link}
    </h2>
    {tldr_html}
</div>
"""


def generate_story_html(story: StoryDisplay) -> str:
    cluster_chip: str = ""
    if story.cluster_name:
        cluster_chip = (
            f'<span class="cluster-chip">{html.escape(story.cluster_name)}</span>'
        )

    is_rss: bool = story.id < 0
    rss_badge: str = '<span class="rss-badge">RSS</span>' if is_rss else ""
    rss_class: str = " rss-story" if is_rss else ""

    tldr_html: str = ""
    if story.tldr:
        tldr_html = f'<div class="text-xs text-stone-600 bg-stone-50 p-2 rounded border border-stone-100 leading-relaxed whitespace-pre-line">{html.escape(story.tldr)}</div>'

    comment_link: str = ""
    if story.hn_url:
        comment_link = (
            f'<a href="{story.hn_url}" target="_blank" class="ml-2 text-xs font-medium text-hn/70 hover:text-hn transition-colors" title="Comments">💬</a>'
        )

    link_url = story.url or story.hn_url or "#"
    return STORY_CARD_TEMPLATE.format(
        score=story.match_percent,
        rss_class=rss_class,
        rss_badge=rss_badge,
        cluster_chip=cluster_chip,
        points=story.points,
        time_ago=story.time_ago,
        url=link_url,
        title=html.escape(story.title, quote=False),
        comment_link=comment_link,
        tldr_html=tldr_html,
    )


def resolve_cluster_name(cluster_names: dict[int, str], cluster_id: int) -> str:
    """Return cluster name with stable fallback for unnamed IDs."""
    if cluster_id == -1:
        return ""
    return cluster_names.get(cluster_id, f"Group {cluster_id + 1}")


async def main() -> None:
    config_defaults, config_path = _load_config(sys.argv[1:])

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate personalized HN dashboard"
    )
    parser.add_argument(
        "--config",
        default=str(config_path) if config_path else None,
        help="Path to hn_rerank.toml config file",
    )
    parser.add_argument(
        "username",
        nargs="?",
        help="Hacker News username",
    )
    parser.add_argument("-o", "--output", default="index.html", help="Output file path")
    parser.add_argument(
        "-c", "--count", type=int, default=30, help="Number of stories to show"
    )
    parser.add_argument(
        "-s",
        "--signals",
        type=int,
        default=MAX_USER_STORIES,
        help=f"Number of user signals to process (default: {MAX_USER_STORIES})",
    )
    parser.add_argument(
        "-k",
        "--candidates",
        type=int,
        default=CANDIDATE_FETCH_COUNT,
        help=f"Number of candidates to fetch from Algolia (default: {CANDIDATE_FETCH_COUNT})",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help=f"Number of interest clusters to discover (2-{MAX_CLUSTERS}, default: {DEFAULT_CLUSTER_COUNT})",
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        default=ALGOLIA_DEFAULT_DAYS,
        help=f"Time window in days for fetching candidates (default: {ALGOLIA_DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--use-hidden-signal",
        action="store_true",
        default=True,
        help="Use hidden stories as negative signals (default: True)",
    )
    parser.add_argument(
        "--no-hidden-signal",
        action="store_false",
        dest="use_hidden_signal",
        help="Don't use hidden stories as negative signals, only exclude them",
    )
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        default=True,
        help="Use Logistic Regression classifier (default: True, disable with --no-classifier)",
    )
    parser.add_argument(
        "--no-classifier",
        action="store_false",
        dest="use_classifier",
        help="Disable classifier, use k-NN heuristics only",
    )
    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="Only penalize when neg_knn > pos_knn (default: always penalize)",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=KNN_NEIGHBORS,
        help=f"Number of neighbors for k-NN scoring (default: {KNN_NEIGHBORS})",
    )
    parser.add_argument(
        "--no-naming",
        action="store_true",
        help="Disable LLM-based cluster naming",
    )
    parser.add_argument(
        "--no-rss",
        action="store_true",
        help="Disable RSS candidate fetching",
    )
    parser.add_argument(
        "--no-tldr",
        action="store_true",
        default=True,
        help="Disable TL;DR generation (default: True, enable with --tldr)",
    )
    parser.add_argument(
        "--tldr",
        action="store_false",
        dest="no_tldr",
        help="Enable TL;DR generation",
    )
    parser.add_argument(
        "--debug-scores",
        action="store_true",
        help="Write score breakdown JSON for selected stories",
    )
    parser.add_argument(
        "--debug-scores-path",
        default=None,
        help="Optional path for score breakdown JSON (defaults next to output)",
    )
    parser.add_argument(
        "--debug-clusters",
        action="store_true",
        help="Write cluster naming prompts/responses to JSON for debugging",
    )
    parser.add_argument(
        "--debug-clusters-path",
        default=None,
        help="Optional path for cluster naming debug JSON (defaults next to output)",
    )
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args: argparse.Namespace = parser.parse_args()

    if args.debug_clusters:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    if not args.username:
        console.print("[red][bold][-] Error:[/bold] username is required.[/red]")
        raise SystemExit(1)

    # Implication: Classifier requires negative signals
    if args.use_classifier:
        args.use_hidden_signal = True

    # Initialize model early
    rerank.init_model()

    if not args.no_naming and not os.environ.get("GROQ_API_KEY"):
        console.print(
            "[red][bold][-] Error:[/bold] GROQ_API_KEY not found in environment.[/red]"
        )
        console.print(
            "[yellow][!] This key is required for cluster naming and story TL;DRs.[/yellow]"
        )
        console.print("    Please run: [cyan]export GROQ_API_KEY='your-key'[/cyan]")
        raise SystemExit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        # 1. Profile Building
        p_task: TaskID = progress.add_task(
            f"[*] Building profile for @{args.username}...", total=100
        )
        async with HNClient() as hn:
            # Check if logged in
            is_logged_in: bool = "logout" in (await hn.client.get("/")).text
            if not is_logged_in:
                progress.stop()
                console.print(
                    "[yellow][!] Not logged in. Upvotes require authentication.[/yellow]"
                )
                pw: str = getpass.getpass(f"Enter password for {args.username}: ")
                success: bool
                msg: str
                success, msg = await hn.login(args.username, pw)
                if not success:
                    console.print(f"[red][-] Login failed: {msg}[/red]")
                    raise SystemExit(1)
                console.print("[green][+] Login successful![/green]")
                progress.start()

            data: UserSignals = await hn.fetch_user_data(args.username)
            progress.update(
                p_task, completed=20, description="[*] Fetching signal details..."
            )

            # Helper for progress-aware batch fetch
            async def fetch_with_progress(ids: list[int], label: str) -> list[Story]:
                results: list[Story] = []
                # Don't create a new task, just update the main one's description
                progress.update(
                    p_task, description=f"[*] Fetching {label} ({len(ids)} items)..."
                )

                if not ids:
                    return []

                # Calculate step size to reach 100% (allocated 20-100 range = 80%)
                # We have two batches (pos + neg), so each gets ~40%
                step = 40.0 / len(ids)

                for res in asyncio.as_completed(
                    [fetch_story(hn.client, sid) for sid in ids]
                ):
                    s: Optional[Story] = await res
                    if s:
                        results.append(s)
                    progress.update(p_task, advance=step)
                return results

            # Positive signals = Favorites + Upvoted (merged in fetch_user_data)
            pos_ids: list[int] = list(data["pos"])[: args.signals]
            
            neg_ids: list[int] = []
            if args.use_hidden_signal:
                neg_ids = list(data["hidden"])[: args.signals]

            pos_stories: list[Story] = await fetch_with_progress(
                pos_ids, "Positive signals"
            )
            neg_stories: list[Story] = []
            if neg_ids:
                neg_stories = await fetch_with_progress(neg_ids, "Negative signals")
            progress.update(
                p_task, completed=100, description="[green][+] Profile built."
            )

        # 2. Embedding
        e_task: TaskID = progress.add_task("[*] Embedding preferences...", total=100)

        def emb_cb(curr: int, total: int) -> None:
            progress.update(e_task, total=total, completed=curr)

        p_emb: Optional[NDArray[np.float32]] = (
            rerank.get_embeddings(
                [s.text_content for s in pos_stories],
                is_query=True,
                progress_callback=emb_cb,
            )
            if pos_stories
            else None
        )
        n_emb: Optional[NDArray[np.float32]] = (
            rerank.get_embeddings(
                [s.text_content for s in neg_stories],
                is_query=True,
                progress_callback=emb_cb,
            )
            if neg_stories
            else None
        )
        progress.update(e_task, description="[green][+] Preferences embedded.")

        cluster_emb: Optional[NDArray[np.float32]] = None
        if pos_stories:
            ce_task: TaskID = progress.add_task(
                "[*] Embedding cluster content...", total=100
            )

            def cluster_emb_cb(curr: int, total: int) -> None:
                progress.update(ce_task, total=total, completed=curr)

            cluster_emb = rerank.get_cluster_embeddings(
                [s.text_content for s in pos_stories],
                progress_callback=cluster_emb_cb,
            )
            progress.update(ce_task, description="[green][+] Cluster content embedded.")

        # 2b. Clustering interests
        cluster_labels: Optional[NDArray[np.int32]] = None
        cluster_centroids: Optional[NDArray[np.float32]] = None
        cluster_names: dict[int, str] = {}
        cluster_source = cluster_emb if cluster_emb is not None else p_emb
        if cluster_source is not None and len(cluster_source) > 0:
            cl_task: TaskID = progress.add_task("[cyan]Clustering interests...", total=1)
            cluster_centroids, cluster_labels = rerank.cluster_interests_with_labels(
                cluster_source, n_clusters=args.clusters
            )
            progress.update(
                cl_task, completed=1, description="[green][+] Interests clustered."
            )

            # Build cluster names (LLM calls)
            # Use story score as weight for LLM to see most popular stories first
            clusters_for_naming: dict[int, list[tuple[StoryDict, float]]] = (
                defaultdict(list)
            )
            for i, label in enumerate(cluster_labels):
                story = pos_stories[i]
                clusters_for_naming[int(label)].append(
                    (story.to_dict(), float(story.score))
                )

            n_clusters = len(clusters_for_naming)
            name_task: TaskID = progress.add_task(
                "[cyan]Naming clusters...", total=n_clusters
            )

            def name_cb(curr: int, total: int) -> None:
                progress.update(name_task, completed=curr)

            if args.no_naming:
                cluster_names = {cid: f"Interest Group {cid + 1}" for cid in clusters_for_naming}
                progress.update(name_task, completed=n_clusters, description="[yellow][!] Using generic cluster names.")
            else:
                try:
                    debug_path = None
                    if args.debug_clusters:
                        debug_path = (
                            Path(args.debug_clusters_path)
                            if args.debug_clusters_path
                            else Path(args.output).with_name("cluster_name_debug.json")
                        )
                    cluster_names = await rerank.generate_batch_cluster_names(
                        clusters_for_naming,
                        progress_callback=name_cb,
                        debug_path=debug_path,
                    )
                    progress.update(name_task, description="[green][+] Clusters named.")
                except RuntimeError as exc:
                    progress.stop()
                    console.print(
                        f"[red][bold][-] Groq naming failed:[/bold] {exc}[/red]"
                    )
                    raise

        # 3. Candidates
        c_task: TaskID = progress.add_task(
            f"[*] Fetching {args.candidates} candidates...", total=args.candidates
        )
        # Exclude everything we've already interacted with
        exclude_ids: set[int] = data["favorites"] | data["upvoted"] | data["hidden"]
        exclude_urls: set[str] = data.get("hidden_urls", set())

        cands: list[Story] = await get_best_stories(
            args.candidates,
            exclude_ids=exclude_ids,
            exclude_urls=exclude_urls,
            progress_callback=lambda curr, tot: progress.update(
                c_task, total=tot, completed=curr
            ),
            days=args.days,
            include_rss=not args.no_rss,
        )
        progress.update(
            c_task, description=f"[green][+] Candidates fetched.   ({len(cands)} valid)"
        )

        # 4. Reranking
        r_task: TaskID = progress.add_task("[*] Reranking stories...", total=100)

        def rank_cb(curr: int, total: int) -> None:
            progress.update(r_task, total=total, completed=curr)

        ranked: list[RankResult] = rerank.rank_stories(
            cands,
            p_emb,
            n_emb,
            use_classifier=args.use_classifier,
            use_contrastive=args.contrastive,
            knn_k=args.knn,
            progress_callback=rank_cb,
        )
        progress.update(
            r_task, completed=100, description="[green][+] Reranking complete."
        )

    # Compute cluster assignments for candidates (only if above similarity threshold)
    cand_cluster_map: dict[int, int] = {}  # cand_idx -> cluster_id (-1 = no cluster)
    if cluster_centroids is not None and len(cands) > 0:
        cand_texts = [c.text_content for c in cands]
        cand_emb = rerank.get_cluster_embeddings(cand_texts)
        if len(cand_emb) > 0:
            sim_to_clusters = cosine_similarity(cand_emb, cluster_centroids)
            for i in range(len(cands)):
                max_sim = float(np.max(sim_to_clusters[i]))
                if max_sim >= CLUSTER_SIMILARITY_THRESHOLD:
                    cand_cluster_map[i] = int(np.argmax(sim_to_clusters[i]))
                else:
                    cand_cluster_map[i] = -1  # Below threshold - no cluster

    stories_data: list[StoryDisplay] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    seen_clusters: set[int] = set()

    def get_cluster_id(result: RankResult) -> int:
        """Get cluster ID for a result (-1 if none)."""
        if (
            result.best_fav_index != -1
            and result.best_fav_index < len(pos_stories)
            and cluster_labels is not None
        ):
            return int(cluster_labels[result.best_fav_index])
        elif result.index in cand_cluster_map:
            return cand_cluster_map[result.index]
        return -1

    def make_story_display(result: RankResult) -> Optional[StoryDisplay]:
        """Create StoryDisplay from RankResult, handling dedup."""
        s: Story = cands[result.index]

        url: Optional[str] = s.url
        title: str = s.title

        norm_url: str = str(url).split("?")[0] if url else f"hn:{s.id}"
        norm_title: str = title.lower().strip() if title else ""

        if norm_url in seen_urls or norm_title in seen_titles:
            return None

        if url:
            seen_urls.add(norm_url)
        if title:
            seen_titles.add(norm_title)

        reason: str = ""
        reason_url: str = ""
        if result.best_fav_index != -1 and result.best_fav_index < len(pos_stories):
            fav_story = pos_stories[result.best_fav_index]
            reason = fav_story.title
            reason_url = f"https://news.ycombinator.com/item?id={fav_story.id}"

        cid = get_cluster_id(result)
        cluster_name: str = resolve_cluster_name(cluster_names, cid)

        hn_url = f"https://news.ycombinator.com/item?id={s.id}" if s.id > 0 else None

        return StoryDisplay(
            id=s.id,
            match_percent=int(result.knn_score * 100),
            cluster_name=cluster_name,
            points=s.score,
            time_ago=get_relative_time(s.time),
            url=s.url,
            title=s.title or "Untitled",
            hn_url=hn_url,
            reason=reason,
            reason_url=reason_url,
            comments=list(s.comments),
        )

    # Phase 1: Ensure top story from each cluster is included
    selected_results: list[RankResult] = []
    used_indices: set[int] = set()
    
    for result in ranked:
        cid = get_cluster_id(result)
        if cid != -1 and cid not in seen_clusters:
            selected_results.append(result)
            seen_clusters.add(cid)
            used_indices.add(result.index)
        if len(seen_clusters) >= len(cluster_names):
            break  # All clusters represented

    # Phase 2: Fill remaining slots with best remaining stories from MMR ranking
    for result in ranked:
        if len(selected_results) >= args.count:
            break
        if result.index in used_indices:
            continue
        selected_results.append(result)
        used_indices.add(result.index)

    # Ensure at least 1/3 of results are RSS stories (best-effort)
    rss_target: int = max(1, (args.count + 2) // 3)

    def is_rss_result(result: RankResult) -> bool:
        return cands[result.index].id < 0

    rss_selected = sum(1 for r in selected_results if is_rss_result(r))
    if rss_selected < rss_target:
        rss_needed = rss_target - rss_selected
        rss_candidates = [
            r for r in ranked if is_rss_result(r) and r.index not in used_indices
        ]
        non_rss_selected = [r for r in selected_results if not is_rss_result(r)]
        non_rss_selected.sort(key=lambda r: r.hybrid_score)

        for new_rss in rss_candidates[:rss_needed]:
            if not non_rss_selected:
                break
            to_remove = non_rss_selected.pop(0)
            selected_results.remove(to_remove)
            used_indices.discard(to_remove.index)
            selected_results.append(new_rss)
            used_indices.add(new_rss.index)

    # FINAL STEP: Re-sort selected stories by hybrid_score to ensure best ranking
    # The cluster logic ensures diversity, but we want the best of those stories at the top.
    selected_results.sort(key=lambda x: x.hybrid_score, reverse=True)
    rss_count = sum(1 for r in selected_results if is_rss_result(r))
    print(f"[+] Selected {rss_count}/{len(selected_results)} RSS stories.")

    if args.debug_scores:
        debug_path = (
            Path(args.debug_scores_path)
            if args.debug_scores_path
            else Path(args.output).with_name("scores_debug.json")
        )
        debug_rows: list[dict[str, object]] = []
        for result in selected_results:
            story = cands[result.index]
            debug_rows.append(
                {
                    "id": story.id,
                    "title": story.title,
                    "url": story.url,
                    "is_rss": story.id < 0,
                    "hybrid_score": result.hybrid_score,
                    "semantic_score": result.semantic_score,
                    "hn_score": result.hn_score,
                    "freshness_boost": result.freshness_boost,
                    "knn_score": result.knn_score,
                }
            )
        debug_path.write_text(json.dumps(debug_rows, indent=2))
        print(f"[+] Score breakdown saved to: {os.path.abspath(debug_path)}")

    stories_data: list[StoryDisplay] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for result in selected_results:
        sd = make_story_display(result)
        if sd:
            stories_data.append(sd)

    if not args.no_tldr:
        # Generate TL;DRs for stories
        print("[*] Generating content via LLM...")
        with progress:
            llm_task: TaskID = progress.add_task(
                "[cyan]Generating TL;DRs...", total=len(stories_data)
            )

            # Batch TL;DR generation (convert to dicts for API)
            stories_for_tldr = [sd.to_dict() for sd in stories_data]
            tldrs = await rerank.generate_batch_tldrs(
                stories_for_tldr,
                progress_callback=lambda curr, tot: progress.update(
                    llm_task, completed=curr
                ),
            )

            for sd in stories_data:
                # Assign batched TL;DR
                sd.tldr = tldrs.get(sd.id, "")

            progress.update(
                llm_task,
                completed=len(stories_data),
                description="[green][+] LLM content generated.",
            )

    print("[*] Generating HTML...")

    # Generate full cluster cards for clusters.html
    clusters_page_html: str = ""
    n_clusters: int = len(cluster_names)
    if cluster_labels is not None and len(pos_stories) > 0:
        # Rebuild clusters dict for the clusters page (reuse cluster_names from earlier)
        clusters: dict[int, list[Story]] = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[int(label)].append(pos_stories[i])

        # Sort each cluster by time (most recent first)
        for cid in clusters:
            clusters[cid].sort(key=lambda x: x.time, reverse=True)

        # Generate cluster cards for clusters.html
        cluster_cards: list[str] = []
        for cid in sorted(clusters.keys(), key=lambda c: -len(clusters[c])):
            items = clusters[cid]
            stories_in_cluster: str = ""
            for story in items[:15]:  # Limit display
                hn_url = (
                    f"https://news.ycombinator.com/item?id={story.id}"
                    if story.id > 0
                    else None
                )
                link_url = story.url or hn_url or ""
                stories_in_cluster += CLUSTER_STORY_TEMPLATE.format(
                    hn_url=link_url,
                    title=html.escape(story.title or "Untitled", quote=False),
                    points=story.score,
                    time_ago=get_relative_time(story.time),
                )
            cluster_cards.append(
                CLUSTER_CARD_TEMPLATE.format(
                    cluster_name=html.escape(resolve_cluster_name(cluster_names, cid)),
                    count=len(items),
                    stories_html=stories_in_cluster,
                )
            )

        clusters_page_html = CLUSTERS_PAGE_TEMPLATE.format(
            username=args.username,
            n_signals=len(pos_stories),
            n_clusters=n_clusters,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            clusters_html="\n".join(cluster_cards),
        )

    stories_html: str = "\n".join([generate_story_html(sd) for sd in stories_data])

    final_html: str = HTML_TEMPLATE.format(
        username=args.username,
        n_clusters=n_clusters,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stories_html=stories_html,
    )

    try:
        Path(args.output).write_text(final_html)
        print(f"[+] Dashboard saved to: {os.path.abspath(args.output)}")

        # Write clusters page
        if clusters_page_html:
            clusters_path = Path(args.output).with_name("clusters.html")
            clusters_path.write_text(clusters_page_html)
            print(f"[+] Clusters saved to: {os.path.abspath(clusters_path)}")
    except OSError as e:
        print(f"[!] Error writing output file: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
