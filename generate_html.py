from __future__ import annotations
import argparse
import asyncio
import getpass
import json
import logging
import os
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import hashlib
import numpy as np
from numpy.typing import NDArray
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.console import Console
from rich.logging import RichHandler

from sklearn.metrics.pairwise import cosine_similarity

from api import rerank, llm_utils
from api.client import HNClient, UserSignals
from api.fetching import CandidateProgress, get_best_stories, fetch_story
from api.feedback import (
    FeedbackRecord,
    feedback_action_for_story,
    feedback_key,
    load_feedback,
)
from api.feedback_single_model import (
    build_single_model_feedback_labels,
    train_single_model_from_embeddings,
)
from api.models import RankResult, Story, StoryDict, StoryDisplay
from api.url_utils import normalize_url
from api.config import AppConfig, ExploreConfig
from api.constants import EMBEDDING_MODEL_VERSION

# Regen lock — prevents concurrent dashboard generation from
# feedback regen, timer, and manual runs.
LOCK_PATH = Path(".cache/generate_html.lock")


def _acquire_regen_lock() -> bool:
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError):
            pass
        LOCK_PATH.unlink(missing_ok=True)
        return _acquire_regen_lock()

    import atexit

    atexit.register(lambda: LOCK_PATH.unlink(missing_ok=True))
    return True


console: Console = Console()
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("hn_rerank.toml")


def compute_config_hash(config: AppConfig) -> str:
    """Stable config fingerprint for impression tracking."""
    relevant = {
        "embedding_model": EMBEDDING_MODEL_VERSION,
        "features": config.classifier.features,
        "raw_embedding_features": config.classifier.raw_embedding_features,
        "k_feat": config.classifier.k_feat,
        "model_type": config.single_model.model_type,
        "svm_kernel": config.single_model.svm_kernel,
        "svm_c": config.single_model.svm_c,
        "svm_gamma": str(config.single_model.svm_gamma),
        "explore_enabled": config.explore.enabled,
        "explore_slots": config.explore.slots,
        "explore_min_quality": config.explore.min_quality,
        "explore_top_reserve": config.explore.top_reserve,
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[
        :12
    ]


async def refresh_hn_story_metadata(
    stories: list[Story],
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Refresh volatile HN metadata used directly in dashboard cards."""
    from api.cache_utils import atomic_write_json
    from api.constants import STORY_CACHE_DIR

    hn_stories = [story for story in stories if story.is_hn and story.id > 0]
    if not hn_stories:
        return

    cache_dir = Path(STORY_CACHE_DIR)

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for idx, story in enumerate(hn_stories):
            try:
                resp = await client.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story.id}.json"
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict):
                        descendants = payload.get("descendants")
                        if isinstance(descendants, int):
                            story.comment_count = descendants
                        score = payload.get("score")
                        if isinstance(score, int):
                            story.score = score
                        # Persist updated metadata back to story cache
                        cache_file = cache_dir / f"{story.id}.json"
                        if cache_file.exists():
                            try:
                                existing = json.loads(cache_file.read_text())
                                story_dict = existing.get("story")
                                if isinstance(story_dict, dict):
                                    story_dict["score"] = story.score
                                    story_dict["comment_count"] = story.comment_count
                                    existing["ts"] = time.time()
                                    atomic_write_json(cache_file, existing)
                            except Exception as persist_exc:
                                logging.debug(
                                    "Failed to persist metadata cache for %s: %s",
                                    story.id,
                                    persist_exc,
                                )
            except Exception as exc:
                logging.debug("Failed to refresh HN metadata for %s: %s", story.id, exc)
            if progress_callback:
                progress_callback(idx + 1, len(hn_stories))


# Progress Bar Weights (Total: 1000)
PROGRESS_WEIGHTS = {
    "profile": 150,
    "emb_pref": 50,
    "emb_clust": 50,
    "cluster": 20,
    "naming": 200,
    "candidates": 100,
    "rank": 50,
    "prepare": 50,
    "tldr": 330,
}


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


def format_match_percent(score: float) -> int:
    """Clamp a similarity-like score in [0, 1] to a display percentage."""
    return max(0, min(100, int(round(score * 100))))


def split_feedback_records(
    records: dict[str, FeedbackRecord],
) -> tuple[list[Story], list[Story], set[int], set[str]]:
    positive: list[Story] = []
    negative: list[Story] = []
    hn_ids: set[int] = set()
    urls: set[str] = set()

    for record in records.values():
        story = record.to_story()
        if record.action == "up":
            positive.append(story)
        elif record.action == "down":
            negative.append(story)
        if record.source == "hn" and record.id > 0:
            hn_ids.add(record.id)
        if record.url:
            normalized = normalize_url(record.url)
            if normalized:
                urls.add(normalized)
    return positive, negative, hn_ids, urls


def apply_feedback_signal_overrides(
    data: UserSignals,
    feedback_positive_stories: list[Story],
    feedback_negative_stories: list[Story],
    *,
    signal_limit: int,
    use_hidden_signal: bool,
) -> tuple[list[int], list[int]]:
    feedback_positive_hn_ids = sorted(
        {
            story.id
            for story in feedback_positive_stories
            if story.source == "hn" and story.id > 0
        }
    )
    feedback_negative_hn_ids = sorted(
        {
            story.id
            for story in feedback_negative_stories
            if story.source == "hn" and story.id > 0
        }
    )

    pos_baseline = sorted(
        data["pos"] - set(feedback_negative_hn_ids) - set(feedback_positive_hn_ids)
    )
    pos_ids = (feedback_positive_hn_ids + pos_baseline)[:signal_limit]

    neg_ids: list[int] = []
    if use_hidden_signal:
        neg_baseline = sorted(
            data["hidden"]
            - set(feedback_positive_hn_ids)
            - set(feedback_negative_hn_ids)
        )
        neg_ids = (feedback_negative_hn_ids + neg_baseline)[:signal_limit]
    return pos_ids, neg_ids


def build_candidate_cluster_map(
    cands: list[Story],
    cluster_centroids: NDArray[np.float32] | None,
    threshold: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, int]:
    """Assign candidates to clusters based on centroid similarity."""
    if cluster_centroids is None or not cands:
        return {}

    cand_texts = [c.text_content for c in cands]
    cand_emb = rerank.get_embeddings(
        cand_texts,
        progress_callback=progress_callback,
    )
    if len(cand_emb) == 0:
        return {}

    sim_to_clusters = cosine_similarity(cand_emb, cluster_centroids)
    cluster_map: dict[int, int] = {}
    for i in range(len(cands)):
        max_sim = float(np.max(sim_to_clusters[i]))
        if max_sim >= threshold:
            cluster_map[i] = int(np.argmax(sim_to_clusters[i]))
        else:
            cluster_map[i] = -1
    return cluster_map


def get_cluster_id_for_result(
    result: RankResult,
    cluster_labels: NDArray[np.int32] | None,
    cand_cluster_map: dict[int, int],
    match_threshold: float,
) -> int:
    """Get cluster ID for a result (-1 if none)."""
    # Prefer cluster assignment from candidate embedding when available.
    # This keeps labels aligned with the candidate's own semantic position,
    # while still allowing fallback to best-favorite mapping.
    cand_cid = cand_cluster_map.get(result.index, -1)
    if cand_cid != -1:
        return cand_cid

    if (
        result.best_fav_index != -1
        and result.max_sim_score >= match_threshold
        and cluster_labels is not None
        and result.best_fav_index < len(cluster_labels)
    ):
        return int(cluster_labels[result.best_fav_index])
    return -1


def select_ranked_results(
    ranked: list[RankResult],
    cands: list[Story],
    cluster_labels: NDArray[np.int32] | None,
    cluster_names: dict[int, str],
    cand_cluster_map: dict[int, int],
    count: int,
) -> list[RankResult]:
    """Select a ranked subset with a small fixed external quota and diversity.

    The quota compensates for HN's site-score blend so external stories are not
    crowded out purely by HN points. It also ensures source diversity for external items.
    """
    _ = (cluster_labels, cluster_names, cand_cluster_map)
    if not ranked:
        return []

    def is_external_result(res: RankResult) -> bool:
        return cands[res.index].is_external

    external_candidates = [r for r in ranked if is_external_result(r)]
    hn_candidates = [r for r in ranked if not is_external_result(r)]

    desired_external = round(count * 0.2) + 5
    available_external = len(external_candidates)
    available_hn = len(hn_candidates)
    min_external = max(0, count - available_hn)
    max_external = min(count, available_external)
    target_external = min(max(desired_external, min_external), max_external)
    target_hn = count - target_external

    # Select external with diversity: start with strict per-source quota and relax if needed
    selected_external: list[RankResult] = []

    for max_per_source in [2, 3, count]:
        selected_external = []
        source_counts: Counter[str] = Counter()
        for r in external_candidates:
            if len(selected_external) >= target_external:
                break
            source = cands[r.index].source
            if source_counts[source] < max_per_source:
                selected_external.append(r)
                source_counts[source] += 1
        if len(selected_external) >= target_external:
            break

    # Select HN (unfiltered)
    selected_hn = hn_candidates[:target_hn]

    # Combine and sort by the active final score.
    selected_results = selected_external + selected_hn
    selected_results.sort(key=lambda x: x.model_score, reverse=True)
    return selected_results


def _percentile_rank(values: list[float]) -> list[float]:
    """Rank each value 0..1 across the pool."""
    n = len(values)
    if n <= 1:
        return [1.0] * n
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for pos, i in enumerate(sorted_idx):
        ranks[i] = pos / (n - 1)
    return ranks


def _partition_pool_by_percentile(
    pool: list[RankResult],
) -> dict[str, list[RankResult]]:
    """Partition pool into types by each candidate's strongest percentile signal.

    Every candidate lands in exactly one bucket; tie-break order favours non-uncertainty
    types so disagreement/novel buckets are never empty.
    """
    buckets: dict[str, list[RankResult]] = {
        "uncertainty": [],
        "disagreement": [],
        "novel": [],
    }
    if not pool:
        return buckets

    ent_pct = _percentile_rank([r.entropy for r in pool])
    gap_pct = _percentile_rank([r.max_sim_score - r.model_score for r in pool])
    novel_pct = _percentile_rank([-r.max_sim_score for r in pool])

    # tie-break priority: disagreement > novel > uncertainty
    tie_break = {"disagreement": 0, "novel": 1, "uncertainty": 2}

    for i, r in enumerate(pool):
        candidates = [
            ("uncertainty", ent_pct[i]),
            ("disagreement", gap_pct[i]),
            ("novel", novel_pct[i]),
        ]
        candidates.sort(key=lambda x: (-x[1], tie_break[x[0]]))
        kind = candidates[0][0]
        buckets[kind].append(r)

    return buckets


def select_explore_slots(
    selected: list[RankResult],
    config: ExploreConfig,
) -> list[RankResult]:
    """Reserve explore slots in the final dashboard list.

    Top *top_reserve* stay as pure exploit.
    Remaining *slots* are filled by type: disagreement > novel > uncertainty,
    then interleaved every 5th position.
    """
    if not selected or config.slots <= 0:
        return selected

    n = len(selected)
    reserve = min(config.top_reserve, n)
    slot_count = min(config.slots, n - reserve)

    exploit_part = list(selected[:reserve])
    pool_part = list(selected[reserve:])

    if slot_count <= 0:
        return selected

    # Classify and partition pool candidates by percentile signal
    candidates_by_type = _partition_pool_by_percentile(pool_part)

    # Sort each bucket by its probe-specific criterion
    candidates_by_type["uncertainty"].sort(key=lambda x: -x.entropy)
    candidates_by_type["disagreement"].sort(
        key=lambda x: -(x.max_sim_score - x.model_score)
    )
    candidates_by_type["novel"].sort(key=lambda x: x.max_sim_score)

    # Each non-uncertainty type gets slot_count // 3 slots; uncertainty fills rest.
    per_type = slot_count // 3
    priority: list[str] = ["disagreement", "novel", "uncertainty"]
    explore_items: list[RankResult] = []
    for kind in priority:
        bucket = candidates_by_type[kind]
        target = per_type if kind != "uncertainty" else slot_count - len(explore_items)
        for r in bucket[:target]:
            if len(explore_items) >= slot_count:
                break
            r.acquisition_kind = kind
            explore_items.append(r)

    # Sort explore items by model_score so the best appears first
    explore_items.sort(key=lambda x: -x.model_score)
    explore_ids = {id(r) for r in explore_items}
    remaining = [r for r in pool_part if id(r) not in explore_ids]

    # Interleave: after reserve, every 5th slot is explore
    result: list[RankResult] = list(exploit_part)
    explore_idx = 0
    slot = len(result)
    while slot < n:
        if (slot + 1) % 5 == 0 and explore_idx < len(explore_items):
            result.append(explore_items[explore_idx])
            explore_idx += 1
        elif remaining:
            result.append(remaining.pop(0))
        else:
            result.extend(explore_items[explore_idx:])
            break
        slot += 1

    return result


_JINJA_ENV: Environment = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=True,
)
_STORY_TEMPLATE = _JINJA_ENV.get_template("story_card.html")
_INDEX_TEMPLATE = _JINJA_ENV.get_template("index.html")
_CLUSTER_STORY_TEMPLATE = _JINJA_ENV.get_template("cluster_story.html")
_CLUSTER_CARD_TEMPLATE = _JINJA_ENV.get_template("cluster_card.html")
_CLUSTERS_TEMPLATE = _JINJA_ENV.get_template("clusters.html")


def generate_story_html(story: StoryDisplay) -> str:
    link_url = story.url or story.hn_url or "#"
    if story.is_hn:
        card_url = story.hn_url
        card_aria_label = f"Open comments for {story.title}" if story.hn_url else None
    else:
        card_url = story.hn_url or story.url
        card_aria_label = (
            f"Open comments for {story.title}"
            if story.hn_url
            else (f"Open story for {story.title}" if story.url else None)
        )
    return _STORY_TEMPLATE.render(
        score=story.match_percent,
        is_external=story.is_external,
        source_badge=story.badge_label,
        cluster_name=story.cluster_name,
        points=story.points,
        time_ago=story.time_ago,
        story_time=story.time,
        rank_index=story.rank_index,
        story_id=story.id,
        story_source=story.source,
        story_url=story.url,
        story_score=story.points,
        story_comment_count=story.comment_count,
        feedback_key=feedback_key(story.source, story.id, story.url),
        feedback_action=story.feedback_action,
        model_score=story.model_score,
        knn_score=story.knn_score,
        max_sim_score=story.max_sim_score,
        max_cluster_score=story.max_cluster_score,
        acquisition_kind=story.acquisition_kind,
        card_url=card_url,
        card_aria_label=card_aria_label,
        url=link_url,
        title=story.title,
        hn_url=story.hn_url,
        comment_count=story.comment_count,
        tldr=story.tldr if len(story.tldr) > 20 else "",
    )


def resolve_cluster_name(
    cluster_names: dict[int, str],
    cluster_id: int,
    allow_empty_fallback: bool = False,
) -> str:
    """Return cluster name with stable fallback for unnamed IDs."""
    if cluster_id == -1:
        return ""
    if cluster_id in cluster_names:
        name = cluster_names[cluster_id].strip()
        if name:
            return name
        if allow_empty_fallback:
            return f"Group {cluster_id + 1}"
        return ""
    return f"Group {cluster_id + 1}"


async def main() -> None:
    config_path = None
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate personalized HN dashboard"
    )
    parser.add_argument(
        "--config",
        help="Path to hn_rerank.toml config file",
    )
    parser.add_argument(
        "username",
        nargs="?",
        help="Hacker News username",
    )
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-c", "--count", type=int, help="Number of stories to show")
    parser.add_argument(
        "-s",
        "--signals",
        type=int,
        help="Number of user signals to process",
    )
    parser.add_argument(
        "-k",
        "--candidates",
        type=int,
        help="Number of candidates to fetch from Algolia",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        help="Number of interest clusters to discover",
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        help="Time window in days for fetching candidates",
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
        default=6,
        help="Number of neighbors for k-NN scoring (default: 6)",
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
        "--open-index-archive",
        action="store_true",
        help="Enable open-index archive fetching for older HN candidates",
    )
    parser.add_argument(
        "--bigquery-archive",
        action="store_true",
        help=(
            "Deprecated alias for --open-index-archive; BigQuery archive "
            "fetching has been replaced"
        ),
    )
    parser.add_argument(
        "--no-tldr",
        action="store_true",
        default=False,
        help="Disable TL;DR generation",
    )
    parser.add_argument(
        "--tldr",
        action="store_false",
        dest="no_tldr",
        help="Enable TL;DR generation (default: True)",
    )
    parser.add_argument(
        "--debug-scores",
        action="store_true",
        default=None,
        help="Write score breakdown JSON for selected stories",
    )
    parser.add_argument(
        "--no-debug-scores",
        action="store_false",
        dest="debug_scores",
        help="Disable score breakdown JSON output",
    )
    parser.add_argument(
        "--mistral",
        action="store_true",
        default=True,
        help="Use Mistral AI (default)",
    )
    parser.add_argument(
        "--groq",
        action="store_false",
        dest="mistral",
        help="Use Groq instead of Mistral AI",
    )
    parser.add_argument(
        "--debug-clusters",
        action="store_true",
        help="Write cluster naming prompts/responses to JSON for debugging",
    )
    args: argparse.Namespace = parser.parse_args()
    config_path = args.config

    provider_choice = "mistral" if args.mistral else "groq"
    config = AppConfig.load(
        toml_path=config_path,
        username=args.username,
        output=args.output,
        count=args.count,
        signals=args.signals,
        candidates=args.candidates,
        days=args.days,
        use_classifier=args.use_classifier,
        contrastive=args.contrastive,
        no_rss=args.no_rss,
        no_tldr=args.no_tldr,
        no_naming=args.no_naming,
        debug_scores=args.debug_scores,
        debug_clusters=args.debug_clusters,
    )

    # Apply CLI overrides to nested config objects
    from dataclasses import replace

    if args.clusters:
        config = replace(
            config,
            clustering=replace(
                config.clustering,
                max_clusters=args.clusters,
            ),
        )
    if args.open_index_archive or args.bigquery_archive:
        config = replace(
            config,
            archive=replace(config.archive, open_index_enabled=True),
        )

    config = replace(config, llm=replace(config.llm, provider=provider_choice))

    os.environ["LLM_PROVIDER"] = config.llm.provider
    config_hash: str = compute_config_hash(config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_level=True)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not config.username:
        console.print("[red][bold][-] Error:[/bold] username is required.[/red]")
        console.print("    Provide it as an argument or in hn_rerank.toml")
        raise SystemExit(1)

    if not _acquire_regen_lock():
        console.print("[!] Another generation is already running. Skipping.")
        raise SystemExit(0)

    rerank.init_model()

    needs_llm = (not config.no_naming) or (not config.no_tldr)
    if needs_llm and not os.environ.get("GROQ_API_KEY"):
        console.print(
            "[red][bold][-] Error:[/bold] GROQ_API_KEY not found in environment.[/red]"
        )
        if not args.no_naming and not args.no_tldr:
            reason = "cluster naming and story TL;DRs"
        elif not args.no_naming:
            reason = "cluster naming"
        else:
            reason = "story TL;DRs"
        console.print(f"[yellow][!] This key is required for {reason}.[/yellow]")
        console.print("    Please run: [cyan]export GROQ_API_KEY='your-key'[/cyan]")
        raise SystemExit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[bold]Overall Progress[/bold]", total=1000)

        # 1. Profile Building
        p_task: TaskID = progress.add_task(
            f"[*] Building profile for @{config.username}...", total=100
        )
        async with HNClient() as hn:
            # Check if logged in
            is_logged_in: bool = "logout" in (await hn.client.get("/")).text
            if not is_logged_in:
                progress.stop()
                console.print(
                    "[yellow][!] Not logged in. Upvotes require authentication.[/yellow]"
                )
                pw: str = getpass.getpass(f"Enter password for {config.username}: ")
                success: bool
                msg: str
                success, msg = await hn.login(config.username, pw)
                if not success:
                    console.print(f"[red][-] Login failed: {msg}[/red]")
                    raise SystemExit(1)
                console.print("[green][+] Login successful![/green]")
                progress.start()

            data: UserSignals = await hn.fetch_user_data(config.username)
            feedback_records = load_feedback()
            (
                feedback_positive_stories,
                feedback_negative_stories,
                feedback_hn_ids,
                feedback_urls,
            ) = split_feedback_records(feedback_records)
            progress.update(p_task, description="[*] Fetching signal details...")

            # Helper for progress-aware batch fetch
            async def fetch_with_progress(
                ids: list[int], label: str, weight_share: float
            ) -> list[Story]:
                progress.update(
                    p_task, description=f"[*] Fetching {label} ({len(ids)} items)..."
                )

                if not ids:
                    progress.update(overall_task, advance=weight_share)
                    return []

                # Calculate step sizes for both bars
                step = 100.0 * (weight_share / PROGRESS_WEIGHTS["profile"]) / len(ids)
                overall_step = weight_share / len(ids)

                tasks = [
                    asyncio.create_task(fetch_story(hn.client, sid)) for sid in ids
                ]
                results_map: dict[int, Story] = {}
                for coro in asyncio.as_completed(tasks):
                    s: Story | None = await coro
                    if s:
                        results_map[s.id] = s
                    progress.update(p_task, advance=step)
                    progress.update(overall_task, advance=overall_step)

                return [results_map[sid] for sid in ids if sid in results_map]

            # Positive signals = Favorites + Upvoted (merged in fetch_user_data)
            pos_ids, neg_ids = apply_feedback_signal_overrides(
                data,
                feedback_positive_stories,
                feedback_negative_stories,
                signal_limit=args.signals,
                use_hidden_signal=args.use_hidden_signal,
            )

            # Split profile weight between positive and negative fetches
            pos_weight = PROGRESS_WEIGHTS["profile"] * 0.7
            neg_weight = PROGRESS_WEIGHTS["profile"] * 0.3 if neg_ids else 0.0
            if not neg_ids:
                pos_weight = PROGRESS_WEIGHTS["profile"]

            pos_stories: list[Story] = await fetch_with_progress(
                pos_ids, "Positive signals", pos_weight
            )
            pos_story_ids = {story.id for story in pos_stories if story.source == "hn"}
            pos_stories.extend(
                story
                for story in feedback_positive_stories
                if story.source != "hn" or story.id not in pos_story_ids
            )
            hn_fetched = len(pos_story_ids)
            hn_requested = len(pos_ids)
            feedback_added = len(pos_stories) - hn_fetched
            sorted_id_hash = hashlib.sha256(
                ",".join(
                    str(s.id) for s in sorted(pos_stories, key=lambda x: x.id)
                ).encode()
            ).hexdigest()[:16]
            logger.info(
                "pos_stories: %d/%d HN fetched, %d feedback added, total %d, id_hash=%s",
                hn_fetched,
                hn_requested,
                feedback_added,
                len(pos_stories),
                sorted_id_hash,
            )
            neg_stories: list[Story] = []
            if neg_ids:
                neg_stories = await fetch_with_progress(
                    neg_ids, "Negative signals", neg_weight
                )
            elif neg_weight > 0:
                progress.update(overall_task, advance=neg_weight)
            neg_story_ids = {story.id for story in neg_stories if story.source == "hn"}
            neg_stories.extend(
                story
                for story in feedback_negative_stories
                if story.source != "hn" or story.id not in neg_story_ids
            )

            progress.update(
                p_task, completed=100, description="[green][+] Profile built."
            )

        # 2. Embedding
        e_task: TaskID = progress.add_task("[*] Embedding preferences...", total=100)

        def make_progress_cb(
            task: TaskID, weight_key: str
        ) -> Callable[[int, int], None]:
            last_completed = 0
            weight = PROGRESS_WEIGHTS[weight_key]

            def cb(curr: int, total: int) -> None:
                nonlocal last_completed
                progress.update(task, total=total, completed=curr)
                if total > 0:
                    delta = curr - last_completed
                    progress.update(overall_task, advance=(delta / total) * weight)
                    last_completed = curr

            return cb

        emb_cb = make_progress_cb(e_task, "emb_pref")

        p_emb: NDArray[np.float32] | None = (
            rerank.get_embeddings(
                [s.text_content for s in pos_stories],
                is_query=True,
                progress_callback=emb_cb,
            )
            if pos_stories
            else None
        )
        # If no stories to embed, advance overall bar anyway
        if not pos_stories:
            progress.update(overall_task, advance=PROGRESS_WEIGHTS["emb_pref"])

        n_emb: NDArray[np.float32] | None = (
            rerank.get_embeddings(
                [s.text_content for s in neg_stories],
                is_query=True,
                progress_callback=emb_cb,
            )
            if neg_stories
            else None
        )
        progress.update(e_task, description="[green][+] Preferences embedded.")

        cluster_emb: NDArray[np.float32] | None = None
        if pos_stories:
            ce_task: TaskID = progress.add_task(
                "[*] Embedding cluster content...", total=100
            )
            cluster_emb_cb = make_progress_cb(ce_task, "emb_clust")

            cluster_emb = rerank.get_embeddings(
                [s.text_content for s in pos_stories],
                progress_callback=cluster_emb_cb,
            )
            progress.update(ce_task, description="[green][+] Cluster content embedded.")
        else:
            progress.update(overall_task, advance=PROGRESS_WEIGHTS["emb_clust"])

        # 2b. Clustering interests
        cluster_labels: NDArray[np.int32] | None = None
        cluster_centroids: NDArray[np.float32] | None = None
        cluster_names: dict[int, str] = {}
        cluster_keywords: dict[int, str] = {}
        cluster_source = cluster_emb if cluster_emb is not None else p_emb
        if cluster_source is not None and len(cluster_source) > 0:
            cl_task: TaskID = progress.add_task(
                "[cyan]Clustering interests...", total=1
            )
            cluster_centroids, cluster_labels = rerank.cluster_interests_with_labels(
                cluster_source,
                config=config.clustering,
            )
            progress.update(
                cl_task, completed=1, description="[green][+] Interests clustered."
            )
            progress.update(overall_task, advance=PROGRESS_WEIGHTS["cluster"])

            # Build cluster names (LLM calls)
            clusters_for_naming: dict[int, list[tuple[StoryDict, float]]] = defaultdict(
                list
            )
            for i, label in enumerate(cluster_labels):
                story = pos_stories[i]
                clusters_for_naming[int(label)].append(
                    (story.to_dict(), float(story.score))
                )

            singleton_clusters = {
                cid for cid, items in clusters_for_naming.items() if len(items) == 1
            }
            cluster_names = {cid: "" for cid in singleton_clusters}
            clusters_for_naming = {
                cid: items
                for cid, items in clusters_for_naming.items()
                if cid not in singleton_clusters
            }

            n_clusters = len(clusters_for_naming)
            name_task: TaskID = progress.add_task(
                "[cyan]Naming clusters...", total=n_clusters
            )
            name_cb = make_progress_cb(name_task, "naming")

            if config.no_naming:
                for cid in clusters_for_naming:
                    cluster_names[cid] = f"Interest Group {cid + 1}"
                    cluster_keywords[cid] = ""
                progress.update(
                    name_task,
                    completed=n_clusters,
                    description="[yellow][!] Using generic cluster names.",
                )
                progress.update(overall_task, advance=PROGRESS_WEIGHTS["naming"])
            else:
                try:
                    debug_path = None
                    if config.debug_clusters:
                        debug_path = config.output_path.with_name(
                            "cluster_name_debug.json"
                        )
                        debug_path.parent.mkdir(parents=True, exist_ok=True)
                    cluster_profiles = await llm_utils.generate_batch_cluster_names(
                        clusters_for_naming,
                        progress_callback=name_cb,
                        debug_path=debug_path,
                    )
                    for cid, profile in cluster_profiles.items():
                        cluster_names[cid] = profile["name"]
                        cluster_keywords[cid] = profile["keywords"]

                    progress.update(name_task, description="[green][+] Clusters named.")
                except RuntimeError as exc:
                    progress.stop()
                    provider_name = config.llm.provider.capitalize()
                    console.print(
                        f"[red][bold][-] {provider_name} naming failed:[/bold] {exc}[/red]"
                    )
                    raise
        else:
            progress.update(overall_task, advance=PROGRESS_WEIGHTS["cluster"])
            progress.update(overall_task, advance=PROGRESS_WEIGHTS["naming"])

        # 3. Candidates
        c_task: TaskID = progress.add_task(
            f"[*] Fetching {config.candidates} candidates...", total=100
        )
        candidate_phase_weights = {
            "hn": 55.0,
            "archive_cache": 10.0,
            "archive_open_index": 5.0,
            "rss_feeds": 15.0,
            "rss_content": 15.0,
        }

        def make_phase_cb(
            task: TaskID,
            weight_key: str,
            phase_weights: dict[str, float],
            finalize_label: str,
        ) -> Callable[[str, int, int, str], None]:
            phase_order = list(phase_weights.keys())
            phase_completed = dict.fromkeys(phase_weights, 0.0)
            last_completed = 0.0
            overall_weight = PROGRESS_WEIGHTS[weight_key]

            def cb(phase: str, curr: int, total: int, description: str) -> None:
                nonlocal last_completed
                if phase == "complete":
                    completed = 100.0
                    desc = f"[*] {finalize_label}..."
                else:
                    total = max(total, 1)
                    fraction = min(max(curr / total, 0.0), 1.0)
                    phase_completed[phase] = phase_weights[phase] * fraction
                    idx = phase_order.index(phase)
                    for prior in phase_order[:idx]:
                        phase_completed[prior] = phase_weights[prior]
                    completed = min(sum(phase_completed.values()), 99.0)
                    desc = description
                progress.update(task, completed=completed, description=desc)
                delta = completed - last_completed
                if delta > 0:
                    progress.update(
                        overall_task, advance=(delta / 100.0) * overall_weight
                    )
                    last_completed = completed

            return cb

        _cand_cb_impl = make_phase_cb(
            c_task, "candidates", candidate_phase_weights, "Finalizing candidates"
        )

        def cand_cb(event: CandidateProgress) -> None:
            _cand_cb_impl(
                event["phase"],
                event.get("current", 0),
                event.get("total", 1),
                f"[*] {event.get('label', '')}...",
            )

        # Exclude everything we've already interacted with
        exclude_ids: set[int] = data["favorites"] | data["hidden"]
        exclude_ids |= feedback_hn_ids
        exclude_urls: set[str] = set()
        exclude_urls |= data.get("hidden_urls", set())
        exclude_urls |= data.get("favorites_urls", set())
        exclude_urls |= feedback_urls

        cands: list[Story] = await get_best_stories(
            config.candidates,
            exclude_ids=exclude_ids,
            exclude_urls=exclude_urls,
            progress_callback=cand_cb,
            config=config,
        )
        progress.update(
            c_task, description=f"[green][+] Candidates fetched.   ({len(cands)} valid)"
        )

        # 4. Reranking
        r_task: TaskID = progress.add_task("[*] Reranking stories...", total=100)
        rank_phase_weights = {
            "embeddings": 55.0,
            "scoring": 35.0,
            "finalize": 10.0,
        }
        _rank_cb_impl = make_phase_cb(
            r_task, "rank", rank_phase_weights, "Finalizing rerank"
        )

        def rank_cb(event: rerank.RankProgress) -> None:
            _rank_cb_impl(
                event["phase"],
                event.get("current", 0),
                event.get("total", 1),
                f"[*] {event.get('label', '')}...",
            )

        feedback_labels = build_single_model_feedback_labels(feedback_records).labels
        feedback_story_embeddings = rerank.get_embeddings(
            [story.story.text_content for story in feedback_labels]
        )
        single_model, _ = train_single_model_from_embeddings(
            feedback_labels,
            feedback_story_embeddings,
            p_emb,
            n_emb,
            config,
            config.single_model,
        )

        # Populate per-vote story age cache for the story_age feature (Now done automatically inside rank_stories/train)
        # Populate per-domain recency cache for the domain_recency feature (Now done automatically inside rank_stories/train)

        ranked: list[RankResult] = rerank.rank_stories(
            cands,
            single_model,
            p_emb,
            n_emb,
            config=config,
            progress_callback=rank_cb,
            positive_stories=pos_stories,
            negative_stories=neg_stories,
            cluster_names=cluster_names,
            cluster_keywords=cluster_keywords,
        )
        progress.update(
            r_task, completed=100, description="[green][+] Reranking complete."
        )

        # 5. Final result preparation
        stories_data: list[StoryDisplay] = []
        # Temporary stories_data list for make_story_display logic
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()

        prep_task: TaskID = progress.add_task(
            "[*] Preparing final story cards...", total=100
        )
        prep_phase_weights = {
            "cluster_map": 45.0,
            "select": 5.0,
            "dupes": 30.0,
            "metadata": 10.0,
            "cards": 10.0,
        }
        update_prep = make_phase_cb(
            prep_task, "prepare", prep_phase_weights, "Finalizing prep"
        )

        # Pre-build StoryDisplay items (without TL;DRs yet)
        cand_cluster_map = build_candidate_cluster_map(
            cands,
            cluster_centroids,
            config.clustering.similarity_threshold,
            progress_callback=lambda curr, total: update_prep(
                "cluster_map",
                curr,
                total,
                "[*] Assigning story clusters...",
            ),
        )
        update_prep("cluster_map", 1, 1, "[*] Assigning story clusters...")

        selected_results = select_ranked_results(
            ranked,
            cands,
            cluster_labels,
            cluster_names,
            cand_cluster_map,
            config.count,
        )
        update_prep("select", 1, 1, "[*] Selecting final stories...")

        update_prep("dupes", 1, 1, "[*] Checking duplicate HN submissions...")

        if config.explore.enabled and config.explore.slots > 0:
            selected_results = select_explore_slots(selected_results, config.explore)

        selected_stories = [cands[result.index] for result in selected_results]
        await refresh_hn_story_metadata(
            selected_stories,
            progress_callback=lambda curr, total: update_prep(
                "metadata",
                curr,
                total,
                "[*] Refreshing HN comment counts...",
            ),
        )
        update_prep("metadata", 1, 1, "[*] Refreshing HN comment counts...")

        def make_story_display_local(result: RankResult) -> StoryDisplay | None:
            s: Story = cands[result.index]
            url: str | None = s.url
            title: str = s.title
            norm_url: str = normalize_url(url) if url else f"{s.source}:{s.id}"
            norm_title: str = title.lower().strip() if title else ""
            if norm_url in seen_urls or norm_title in seen_titles:
                return None
            if url:
                seen_urls.add(norm_url)
            if title:
                seen_titles.add(norm_title)
            reason, reason_url = "", ""
            if result.best_fav_index != -1 and result.best_fav_index < len(pos_stories):
                fav_story = pos_stories[result.best_fav_index]
                reason = fav_story.title
                reason_url = f"https://news.ycombinator.com/item?id={fav_story.id}"
            cid = get_cluster_id_for_result(
                result,
                cluster_labels,
                cand_cluster_map,
                config.semantic.match_threshold,
            )
            cluster_name = resolve_cluster_name(
                cluster_names, cid, allow_empty_fallback=s.is_external
            )
            discussion_url = s.discussion_url
            if discussion_url is None and s.is_hn and s.id > 0:
                discussion_url = f"https://news.ycombinator.com/item?id={s.id}"
            return StoryDisplay(
                id=s.id,
                match_percent=format_match_percent(result.model_score),
                cluster_name=cluster_name,
                points=s.score,
                time_ago=get_relative_time(s.time),
                time=s.time,
                url=s.url,
                title=s.title or "Untitled",
                hn_url=discussion_url,
                reason=reason,
                reason_url=reason_url,
                comments=list(s.comments),
                source=s.source,
                text_content=s.text_content,
                model_score=result.model_score,
                knn_score=result.knn_score,
                max_sim_score=result.max_sim_score,
                max_cluster_score=result.max_cluster_score,
                comment_count=s.comment_count,
                feedback_action=feedback_action_for_story(
                    feedback_records,
                    source=s.source,
                    story_id=s.id,
                    url=s.url,
                ),
                acquisition_kind=result.acquisition_kind,
            )

        for rank_index, result in enumerate(selected_results):
            sd = make_story_display_local(result)
            if sd:
                sd.rank_index = rank_index
                stories_data.append(sd)
            update_prep(
                "cards",
                rank_index + 1,
                len(selected_results),
                "[*] Building story cards...",
            )
        update_prep("cards", 1, 1, "[green][+] Final story cards prepared.")

        # 6. TL;DR Generation
        # Move TL;DR generation inside the progress context
        if not config.no_tldr and stories_data:
            llm_task: TaskID = progress.add_task(
                "[cyan]Generating TL;DRs...", total=len(stories_data)
            )
            tldr_cb = make_progress_cb(llm_task, "tldr")

            stories_for_tldr = [sd.to_dict() for sd in stories_data]
            tldrs = await llm_utils.generate_batch_tldrs(
                stories_for_tldr,
                progress_callback=tldr_cb,
            )
            for sd in stories_data:
                sd.tldr = tldrs.get(sd.id, "")
            progress.update(
                llm_task,
                completed=len(stories_data),
                description="[green][+] LLM content generated.",
            )
        else:
            for sd in stories_data:
                sd.tldr = ""
            progress.update(overall_task, advance=PROGRESS_WEIGHTS["tldr"])

    source_counts = Counter(cands[r.index].source for r in selected_results)
    counts_summary = ", ".join(
        f"{source}={count}" for source, count in sorted(source_counts.items())
    )
    print(f"[+] Selected sources: {counts_summary}")

    if config.debug_scores:
        debug_path = config.output_path.with_name("scores_debug.json")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_rows: list[dict[str, object]] = []
        for result in selected_results:
            story = cands[result.index]
            debug_rows.append(
                {
                    "id": story.id,
                    "source": story.source,
                    "title": story.title,
                    "url": story.url,
                    "is_external": story.is_external,
                    "model_score": result.model_score,
                    "knn_score": result.knn_score,
                    "max_cluster_score": result.max_cluster_score,
                    "max_sim_score": result.max_sim_score,
                }
            )
        debug_path.write_text(json.dumps(debug_rows, indent=2))
        print(f"[+] Score breakdown saved to: {os.path.abspath(debug_path)}")

        # Save all scored candidates (not just top 40) for diagnostics
        full_debug_path = config.output_path.with_name("scores_full.json")
        full_debug_rows: list[dict[str, object]] = []
        for result in ranked:
            story = cands[result.index]
            full_debug_rows.append(
                {
                    "id": story.id,
                    "source": story.source,
                    "title": story.title,
                    "model_score": result.model_score,
                    "knn_score": result.knn_score,
                    "max_cluster_score": result.max_cluster_score,
                    "max_sim_score": result.max_sim_score,
                }
            )
        full_debug_path.write_text(json.dumps(full_debug_rows, indent=2))
        print(f"[+] Full score breakdown saved to: {os.path.abspath(full_debug_path)}")

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
                hn_url = story.discussion_url
                if hn_url is None and story.is_hn and story.id > 0:
                    hn_url = f"https://news.ycombinator.com/item?id={story.id}"
                link_url = story.url or hn_url or ""
                stories_in_cluster += _CLUSTER_STORY_TEMPLATE.render(
                    hn_url=link_url,
                    title=story.title or "Untitled",
                    points=story.score,
                    time_ago=get_relative_time(story.time),
                )
            cluster_cards.append(
                _CLUSTER_CARD_TEMPLATE.render(
                    cluster_name=resolve_cluster_name(cluster_names, cid),
                    keywords=cluster_keywords.get(cid, ""),
                    count=len(items),
                    stories_html=Markup(stories_in_cluster),
                )
            )

        clusters_page_html = _CLUSTERS_TEMPLATE.render(
            username=config.username,
            n_signals=len(pos_stories),
            n_clusters=n_clusters,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            clusters_html=Markup("\n".join(cluster_cards)),
        )

    stories_html: str = "\n".join([generate_story_html(sd) for sd in stories_data])

    final_html: str = _INDEX_TEMPLATE.render(
        username=config.username,
        n_clusters=n_clusters,
        config_hash=config_hash,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stories_html=Markup(stories_html),
    )

    try:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(final_html)
        print(f"[+] Dashboard saved to: {os.path.abspath(config.output_path)}")

        if clusters_page_html:
            clusters_path = config.output_path.with_name("clusters.html")
            clusters_path.write_text(clusters_page_html)
            print(f"[+] Clusters saved to: {os.path.abspath(clusters_path)}")

        # Write shared CSS for clusters page (index.html inlines its own copy)
        css_path = config.output_path.with_name("index.css")
        # Extract the inline <style> block from the rendered index HTML
        import re as _re

        css_match = _re.search(r"<style>(.*?)</style>", final_html, _re.DOTALL)
        if css_match:
            css_path.write_text(css_match.group(1).strip() + "\n")
            print(f"[+] CSS saved to: {os.path.abspath(css_path)}")
    except OSError as e:
        print(f"[!] Error writing output file: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
