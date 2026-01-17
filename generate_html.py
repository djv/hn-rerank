from __future__ import annotations
import argparse
import asyncio
import getpass
import html
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from sklearn.metrics.pairwise import cosine_similarity

from api import rerank
from api.client import HNClient
from api.fetching import get_best_stories, fetch_story
from api.models import RankResult, Story, StoryDisplay
from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    CANDIDATE_FETCH_COUNT,
    CLUSTER_SIMILARITY_THRESHOLD,
    MAX_USER_STORIES,
)

console: Console = Console()

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
        .story-card {{ @apply bg-white border border-stone-200 rounded-lg p-3 shadow-sm transition-all hover:border-hn hover:shadow-md; }}
        .cluster-chip {{ @apply px-2 py-1 bg-white border border-stone-200 rounded-full text-xs text-stone-600 hover:border-hn hover:text-hn transition-colors cursor-default; }}
    </style>
</head>
<body class="p-2 md:p-4">
    <div class="max-w-3xl mx-auto">
        <header class="mb-4 border-b border-stone-200 pb-3 flex items-end justify-between">
            <div>
                <h1 class="text-2xl font-black text-stone-900 tracking-tight">
                    HN <span class="text-hn">Rerank</span>
                </h1>
                <p class="text-stone-500 text-xs">@{username} &bull; <a href="clusters.html" class="text-hn hover:underline">{n_clusters} interest clusters</a></p>
            </div>
            <p class="text-[10px] text-stone-400 font-mono">{timestamp}</p>
        </header>

        <div class="grid gap-2">
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
<div class="story-card group">
    <div class="flex items-start justify-between gap-2 mb-1">
        <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 mb-0.5 flex-wrap">
                <span class="px-1.5 py-0.5 rounded bg-hn/10 text-hn text-[10px] font-bold">
                    {score}%
                </span>
                {cluster_chip}
                <span class="text-[10px] text-stone-400 font-mono">{points} pts</span>
                <span class="text-[10px] text-stone-400 font-mono">{time_ago}</span>
            </div>
            <h2 class="text-sm font-semibold text-stone-900 group-hover:text-hn transition-colors leading-snug">
                <a href="{url}" target="_blank">{title}</a>
            </h2>
        </div>
        <a href="{hn_url}" target="_blank" class="shrink-0 p-1 rounded bg-stone-100 text-stone-400 hover:bg-hn hover:text-white transition-all" title="HN">
            <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 12h3v8h6v-6h2v6h6v-8h3L12 2z"/></svg>
        </a>
    </div>
    {reason_html}
    {tldr_html}
</div>
"""


def generate_story_html(story: StoryDisplay) -> str:
    reason_html: str = ""
    if story.reason:
        escaped_reason_title: str = html.escape(story.reason, quote=False)

        if story.reason_url:
            reason_html = f'<p class="text-[11px] text-emerald-600 mb-2">↳ <a href="{story.reason_url}" target="_blank" class="hover:underline">"{escaped_reason_title}"</a></p>'
        else:
            reason_html = f'<p class="text-[11px] text-emerald-600 mb-2">↳ "{escaped_reason_title}"</p>'

    cluster_chip: str = ""
    if story.cluster_name:
        cluster_chip = (
            f'<span class="cluster-chip">{html.escape(story.cluster_name)}</span>'
        )

    tldr_html: str = ""
    if story.tldr:
        tldr_html = f'<div class="text-xs text-stone-600 bg-stone-50 p-2 rounded border border-stone-100 leading-relaxed whitespace-pre-line">{html.escape(story.tldr)}</div>'

    return STORY_CARD_TEMPLATE.format(
        score=story.match_percent,
        cluster_chip=cluster_chip,
        points=story.points,
        time_ago=story.time_ago,
        url=story.url or story.hn_url,
        title=html.escape(story.title, quote=False),
        hn_url=story.hn_url,
        reason_html=reason_html,
        tldr_html=tldr_html,
    )


async def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate personalized HN dashboard"
    )
    parser.add_argument("username", help="Hacker News username")
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
        "-d",
        "--days",
        type=int,
        default=ALGOLIA_DEFAULT_DAYS,
        help=f"Time window in days for fetching candidates (default: {ALGOLIA_DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--use-hidden-signal",
        action="store_true",
        help="Use hidden stories as negative signals (default: False, only excludes them)",
    )
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Use a trained Logistic Regression classifier for scoring instead of heuristics (requires hidden signals)",
    )
    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="Only penalize when neg_knn > pos_knn (default: always penalize)",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=3,
        help="Number of neighbors for k-NN scoring (default: 3)",
    )
    args: argparse.Namespace = parser.parse_args()

    # Implication: Classifier requires negative signals
    if args.use_classifier:
        args.use_hidden_signal = True

    # Initialize model early
    rerank.init_model()

    if not os.environ.get("GROQ_API_KEY"):
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
        p_task: Any = progress.add_task(
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

            data: dict[str, set[int]] = await hn.fetch_user_data(args.username)
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

            # Positive signals = Upvoted only (requires login)
            # IMPORTANT: If a story is both upvoted AND hidden, treat it as hidden (negative/neutral).
            # This allows users to "undo" the signal of an upvote by hiding the story.
            upvoted_ids = data["upvoted"] - data["hidden"]

            pos_ids: list[int] = list(upvoted_ids)[: args.signals]
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
        e_task: Any = progress.add_task("[*] Embedding preferences...", total=100)

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

        # 2b. Clustering interests
        cluster_labels: Optional[NDArray[np.int32]] = None
        cluster_centroids: Optional[NDArray[np.float32]] = None
        cluster_names: dict[int, str] = {}
        if p_emb is not None and len(p_emb) > 0:
            cl_task: Any = progress.add_task("[cyan]Clustering interests...", total=1)
            cluster_centroids, cluster_labels = rerank.cluster_interests_with_labels(
                p_emb
            )
            progress.update(
                cl_task, completed=1, description="[green][+] Interests clustered."
            )

            # Build cluster names (LLM calls)
            # Use story score as weight for LLM to see most popular stories first
            clusters_for_naming: dict[int, list[tuple[dict[str, Any], float]]] = (
                defaultdict(list)
            )
            for i, label in enumerate(cluster_labels):
                story = pos_stories[i]
                clusters_for_naming[int(label)].append(
                    (story.to_dict(), float(story.score))
                )

            n_clusters = len(set(cluster_labels))
            name_task: Any = progress.add_task(
                "[cyan]Naming clusters...", total=n_clusters
            )

            def name_cb(curr: int, total: int) -> None:
                progress.update(name_task, completed=curr)

            cluster_names = await rerank.generate_batch_cluster_names(
                clusters_for_naming, progress_callback=name_cb
            )
            progress.update(name_task, description="[green][+] Clusters named.")

        # 3. Candidates
        c_task: Any = progress.add_task(
            f"[*] Fetching {args.candidates} candidates...", total=args.candidates
        )
        # Exclude everything we've already interacted with
        exclude: set[int] = data["pos"] | data["upvoted"] | data["hidden"]
        cands: list[Story] = await get_best_stories(
            args.candidates,
            exclude_ids=exclude,
            progress_callback=lambda curr, tot: progress.update(
                c_task, total=tot, completed=curr
            ),
            days=args.days,
        )
        progress.update(
            c_task, description=f"[green][+] Candidates fetched.   ({len(cands)} valid)"
        )

        # 4. Reranking
        r_task: Any = progress.add_task("[*] Reranking stories...", total=100)

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
        cand_emb = rerank.get_embeddings(cand_texts)
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

    for result in ranked:
        if len(stories_data) >= args.count:
            break

        s: Story = cands[result.index]

        # Deduplication logic
        url: Optional[str] = s.url
        title: str = s.title

        # Normalize for checking
        norm_url: str = str(url).split("?")[0] if url else f"hn:{s.id}"
        norm_title: str = title.lower().strip() if title else ""

        if norm_url in seen_urls or norm_title in seen_titles:
            continue

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

        # Get cluster name for this candidate based on the matching history item
        cluster_name: str = ""
        if (
            result.best_fav_index != -1
            and result.best_fav_index < len(pos_stories)
            and cluster_labels is not None
        ):
            cid = int(cluster_labels[result.best_fav_index])
            cluster_name = cluster_names.get(cid, "")
        elif result.index in cand_cluster_map:
            # Fallback to candidate's own cluster if no specific favorite match
            cid = cand_cluster_map[result.index]
            if cid != -1:  # -1 means below similarity threshold
                cluster_name = cluster_names.get(cid, "")

        stories_data.append(
            StoryDisplay(
                id=s.id,
                # Use k-NN score for display (mean of top-3 neighbors)
                match_percent=int(result.knn_score * 100),
                cluster_name=cluster_name,
                points=s.score,
                time_ago=get_relative_time(s.time),
                url=s.url,
                title=s.title or "Untitled",
                hn_url=f"https://news.ycombinator.com/item?id={s.id}",
                reason=reason,
                reason_url=reason_url,
                comments=list(s.comments),
            )
        )

    # Generate TL;DRs for stories
    print("[*] Generating content via LLM...")
    with progress:
        llm_task = progress.add_task(
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
                stories_in_cluster += CLUSTER_STORY_TEMPLATE.format(
                    hn_url=f"https://news.ycombinator.com/item?id={story.id}",
                    title=html.escape(story.title or "Untitled", quote=False),
                    points=story.score,
                    time_ago=get_relative_time(story.time),
                )
            cluster_cards.append(
                CLUSTER_CARD_TEMPLATE.format(
                    cluster_name=html.escape(
                        cluster_names.get(cid, f"Group {cid + 1}")
                    ),
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
