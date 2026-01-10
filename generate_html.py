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
from api.constants import (
    ALGOLIA_DEFAULT_DAYS,
    CANDIDATE_FETCH_COUNT,
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
        <span class="text-[10px] text-emerald-600 font-medium">{weight:.0%}</span>
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


def generate_story_html(story: dict[str, Any]) -> str:
    reason_html: str = ""
    if story.get("reason"):
        escaped_reason_title: str = html.escape(str(story["reason"]), quote=False)
        reason_url: str = story.get("reason_url", "")
        
        if story.get("smart_reason"):
            escaped_smart = html.escape(story["smart_reason"])
            if reason_url:
                reason_html = f'<p class="text-[11px] text-emerald-600 mb-2">↳ {escaped_smart} <span class="text-stone-400 mx-1">&middot;</span> <a href="{reason_url}" target="_blank" class="text-stone-400 hover:underline hover:text-emerald-600">Because you liked "{escaped_reason_title}"</a></p>'
            else:
                reason_html = f'<p class="text-[11px] text-emerald-600 mb-2">↳ {escaped_smart} <span class="text-stone-400">(from "{escaped_reason_title}")</span></p>'
        else:
            # Fallback
            if reason_url:
                reason_html = f'<p class="text-[11px] text-emerald-600 mb-2">↳ <a href="{reason_url}" target="_blank" class="hover:underline">Similar to: {escaped_reason_title}</a></p>'
            else:
                reason_html = f'<p class="text-[11px] text-emerald-600 mb-2">↳ Similar to: {escaped_reason_title}</p>'

    cluster_chip: str = ""
    if story.get("cluster_name"):
        cluster_chip = f'<span class="cluster-chip">{html.escape(story["cluster_name"])}</span>'

    tldr_html: str = ""
    if story.get("tldr"):
        tldr_html = f'<div class="text-xs text-stone-600 bg-stone-50 p-2 rounded border border-stone-100 leading-relaxed whitespace-pre-line">{html.escape(story["tldr"])}</div>'

    return STORY_CARD_TEMPLATE.format(
        score=story["match_percent"],
        cluster_chip=cluster_chip,
        points=story["points"],
        time_ago=story["time_ago"],
        url=story["url"] or story["hn_url"],
        title=html.escape(str(story["title"]), quote=False),
        hn_url=story["hn_url"],
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
        "--no-recency-bias",
        action="store_true",
        help="Disable recency weighting for user profile (default: False)",
    )
    args: argparse.Namespace = parser.parse_args()

    # Initialize model early
    rerank.init_model()

    if not os.environ.get("GEMINI_API_KEY"):
        console.print(
            "[red][bold][-] Error:[/bold] GEMINI_API_KEY not found in environment.[/red]"
        )
        console.print(
            "[yellow][!] This key is required for cluster naming and story TL;DRs.[/yellow]"
        )
        console.print("    Please run: [cyan]export GEMINI_API_KEY='your-key'[/cyan]")
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
            async def fetch_with_progress(
                ids: list[int], label: str
            ) -> list[dict[str, Any]]:
                results: list[dict[str, Any]] = []
                sub_task: Any = progress.add_task(f"  > {label}", total=len(ids))
                if not ids:
                    progress.remove_task(sub_task)
                    return []

                for res in asyncio.as_completed(
                    [fetch_story(hn.client, sid) for sid in ids]
                ):
                    s: Optional[dict[str, Any]] = await res
                    if s:
                        results.append(s)
                    progress.update(sub_task, advance=1)
                progress.remove_task(sub_task)
                return results

            # Positive signals = Upvoted only (requires login)
            pos_ids: list[int] = list(data["upvoted"])[: args.signals]
            neg_ids: list[int] = list(data["hidden"])[: args.signals]

            pos_stories: list[dict[str, Any]] = await fetch_with_progress(
                pos_ids, "Positive signals"
            )
            neg_stories: list[dict[str, Any]] = await fetch_with_progress(
                neg_ids, "Negative signals"
            )
            progress.update(
                p_task, completed=100, description="[green][+] Profile built."
            )

        # 2. Embedding
        e_task: Any = progress.add_task("[*] Embedding preferences...", total=100)

        def emb_cb(curr: int, total: int) -> None:
            progress.update(e_task, total=total, completed=curr)

        p_emb: Optional[NDArray[np.float32]] = (
            rerank.get_embeddings(
                [str(s["text_content"]) for s in pos_stories],
                is_query=True,
                progress_callback=emb_cb,
            )
            if pos_stories
            else None
        )
        n_emb: Optional[NDArray[np.float32]] = (
            rerank.get_embeddings(
                [str(s["text_content"]) for s in neg_stories],
                is_query=True,
                progress_callback=emb_cb,
            )
            if neg_stories
            else None
        )
        p_weights: Optional[NDArray[np.float32]] = (
            rerank.compute_recency_weights(
                [int(s["time"]) for s in pos_stories],
                decay_rate=0.0 if args.no_recency_bias else None,
            )
            if pos_stories
            else None
        )
        progress.update(e_task, description="[green][+] Preferences embedded.")

        # 2b. Clustering interests
        cluster_labels: Optional[NDArray[np.int32]] = None
        cluster_centroids: Optional[NDArray[np.float32]] = None
        cluster_names: dict[int, str] = {}
        if p_emb is not None and len(p_emb) > 0:
            cl_task: Any = progress.add_task("[cyan]Clustering interests...", total=1)
            cluster_centroids, cluster_labels = rerank.cluster_interests_with_labels(p_emb, p_weights)
            progress.update(cl_task, completed=1, description="[green][+] Interests clustered.")

            # Build cluster names (LLM calls)
            clusters_for_naming: dict[int, list[tuple[dict[str, Any], float]]] = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                weight = float(p_weights[i]) if p_weights is not None else 1.0
                clusters_for_naming[int(label)].append((pos_stories[i], weight))

            n_clusters = len(set(cluster_labels))
            name_task: Any = progress.add_task("[cyan]Naming clusters...", total=n_clusters)
            
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
        cands: list[dict[str, Any]] = await get_best_stories(
            args.candidates,
            exclude_ids=exclude,
            progress_callback=lambda curr, tot: progress.update(
                c_task, total=tot, completed=curr
            ),
            days=args.days,
        )
        progress.update(c_task, description=f"[green][+] Candidates fetched.   ({len(cands)} valid)")

        # 4. Reranking
        r_task: Any = progress.add_task("[*] Reranking stories...", total=100)

        def rank_cb(curr: int, total: int) -> None:
            progress.update(r_task, total=total, completed=curr)

        ranked: list[tuple[int, float, int, float]] = rerank.rank_stories(
            cands,
            p_emb,
            n_emb,
            p_weights,
            progress_callback=rank_cb,
        )
        progress.update(
            r_task, completed=100, description="[green][+] Reranking complete."
        )

    # Compute cluster assignments for candidates
    cand_cluster_map: dict[int, int] = {}  # cand_idx -> cluster_id
    if cluster_centroids is not None and len(cands) > 0:
        cand_texts = [str(c.get("text_content", "")) for c in cands]
        cand_emb = rerank.get_embeddings(cand_texts)
        if len(cand_emb) > 0:
            sim_to_clusters = cosine_similarity(cand_emb, cluster_centroids)
            for i in range(len(cands)):
                cand_cluster_map[i] = int(np.argmax(sim_to_clusters[i]))

    stories_data: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for idx, score, fav_idx, max_sim in ranked:
        if len(stories_data) >= args.count:
            break

        s: dict[str, Any] = cands[idx]

        # Deduplication logic
        url: Optional[str] = s.get("url")
        title: Optional[str] = s.get("title")

        # Normalize for checking
        norm_url: str = str(url).split("?")[0] if url else f"hn:{s['id']}"
        norm_title: str = str(title).lower().strip() if title else ""

        if norm_url in seen_urls or norm_title in seen_titles:
            continue

        if url:
            seen_urls.add(norm_url)
        if title:
            seen_titles.add(norm_title)

        reason: str = ""
        reason_url: str = ""
        if fav_idx != -1 and fav_idx < len(pos_stories):
            reason = str(pos_stories[fav_idx]["title"])
            reason_url = f"https://news.ycombinator.com/item?id={pos_stories[fav_idx]['id']}"

        # Get cluster name for this candidate
        cluster_name: str = ""
        if idx in cand_cluster_map:
            cid = cand_cluster_map[idx]
            cluster_name = cluster_names.get(cid, "")

        stories_data.append(
            {
                "id": int(s["id"]),
                # Use MaxSim for the UI label because it represents the "Best Match"
                # which is what the "Match: X" reason text implies.
                "match_percent": int(max_sim * 100),
                "cluster_name": cluster_name,
                "points": int(s.get("score", 0)),
                "time_ago": get_relative_time(int(s.get("time", 0))),
                "url": s.get("url"),
                "title": str(s.get("title", "Untitled")),
                "hn_url": f"https://news.ycombinator.com/item?id={s['id']}",
                "reason": reason,
                "reason_url": reason_url,
                "comments": list(s.get("comments", [])),
            }
        )

    # Generate TL;DRs for stories and Smart Reasons
    print("[*] Generating content via LLM...")
    with progress:
        llm_task = progress.add_task("[cyan]Generating TL;DRs & Reasons...", total=len(stories_data) * 2)
        
        # Batch TL;DR generation
        tldrs = await rerank.generate_batch_tldrs(
            stories_data, 
            progress_callback=lambda curr, tot: progress.update(llm_task, completed=curr)
        )
        
        # Collect pairs for batch similarity reasons
        pairs_to_gen = []
        for sd in stories_data:
            if sd.get("reason"):
                pairs_to_gen.append((sd["title"], sd["reason"], sd.get("comments", [])))
        
        reasons = await rerank.generate_batch_similarity_reasons(
            pairs_to_gen,
            progress_callback=None # progress handled manually below for simplicity or we can add it
        )
        
        reason_idx = 0
        for sd in stories_data:
            # Assign batched TL;DR
            sd["tldr"] = tldrs.get(sd["id"], "")
            
            # Smart Reason
            if sd.get("reason"):
                sd["smart_reason"] = reasons[reason_idx]
                reason_idx += 1

        progress.update(llm_task, completed=len(stories_data) * 2, description="[green][+] LLM content generated.")

    print("[*] Generating HTML...")

    # Generate full cluster cards for clusters.html
    clusters_page_html: str = ""
    n_clusters: int = len(cluster_names)
    if cluster_labels is not None and len(pos_stories) > 0:
        # Rebuild clusters dict for the clusters page (reuse cluster_names from earlier)
        clusters: dict[int, list[tuple[dict[str, Any], float]]] = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            weight = float(p_weights[i]) if p_weights is not None else 1.0
            clusters[int(label)].append((pos_stories[i], weight))

        # Sort each cluster by weight (recency)
        for cid in clusters:
            clusters[cid].sort(key=lambda x: x[1], reverse=True)

        # Generate cluster cards for clusters.html
        cluster_cards: list[str] = []
        for cid in sorted(clusters.keys(), key=lambda c: -len(clusters[c])):
            items = clusters[cid]
            stories_in_cluster: str = ""
            for story, weight in items[:15]:  # Limit display
                stories_in_cluster += CLUSTER_STORY_TEMPLATE.format(
                    hn_url=f"https://news.ycombinator.com/item?id={story['id']}",
                    title=html.escape(str(story.get("title", "Untitled")), quote=False),
                    points=int(story.get("score", 0)),
                    time_ago=get_relative_time(int(story.get("time", 0))),
                    weight=weight,
                )
            cluster_cards.append(
                CLUSTER_CARD_TEMPLATE.format(
                    cluster_name=html.escape(cluster_names.get(cid, f"Group {cid + 1}")),
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
