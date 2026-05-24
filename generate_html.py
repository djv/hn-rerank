from __future__ import annotations
import argparse
import asyncio
import getpass
import json
import logging
import os
import re
import time
from typing import Callable, cast
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from bs4 import BeautifulSoup
from numpy.typing import NDArray
from jinja2 import Environment
from markupsafe import Markup
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.console import Console

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
from api.config import AppConfig

console: Console = Console()

DEFAULT_CONFIG_PATH = Path("hn_rerank.toml")
HN_DUPE_CACHE_DIR = Path(".cache/hn_dupes")
HN_DUPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HN_DUPE_TRUE_CACHE_TTL = 15 * 24 * 60 * 60
HN_DUPE_FALSE_CACHE_TTL = 24 * 60 * 60
HN_FIREBASE_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{sid}.json"


def _extract_hn_dupe_target(page_html: str, sid: int) -> tuple[bool, int | None]:
    """Return (is_dupe, target_story_id) from an HN item page."""
    soup = BeautifulSoup(page_html, "html.parser")
    titleline = soup.select_one("span.titleline")
    if titleline is None or "[dupe]" not in titleline.get_text(" ", strip=True):
        return False, None

    for link in soup.select(".comment a[href]"):
        href_value = link.get("href")
        href = href_value if isinstance(href_value, str) else ""
        match = re.search(r"(?:^|[?&])id=(\d+)", href)
        if match is None:
            continue
        target_id = int(match.group(1))
        if target_id != sid:
            return True, target_id
    return True, None


def _load_cached_hn_dupe_target(sid: int) -> tuple[bool, int | None] | None:
    path = HN_DUPE_CACHE_DIR / f"{sid}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logging.debug("Failed to load HN dupe cache %s: %s", path, exc)
        return None
    is_dupe = data.get("is_dupe")
    target_id = data.get("target_id")
    if not isinstance(is_dupe, bool):
        logging.debug("Ignoring malformed HN dupe cache payload in %s", path)
        return None
    if target_id is not None and not isinstance(target_id, int):
        logging.debug("Ignoring malformed HN dupe cache payload in %s", path)
        return None
    ttl = HN_DUPE_TRUE_CACHE_TTL if is_dupe else HN_DUPE_FALSE_CACHE_TTL
    if time.time() - path.stat().st_mtime >= ttl:
        return None
    return is_dupe, target_id


def _save_cached_hn_dupe_target(sid: int, result: tuple[bool, int | None]) -> None:
    path = HN_DUPE_CACHE_DIR / f"{sid}.json"
    path.write_text(json.dumps({"is_dupe": result[0], "target_id": result[1]}))


async def _fetch_hn_dupe_target(
    client: httpx.AsyncClient, sid: int
) -> tuple[bool, int | None]:
    cached = _load_cached_hn_dupe_target(sid)
    if cached is not None:
        return cached

    try:
        for attempt in range(2):
            resp = await client.get(
                "https://news.ycombinator.com/item",
                params={"id": sid},
            )
            if resp.status_code == 429 and attempt == 0:
                await asyncio.sleep(1.0)
                continue
            if resp.status_code != 200 or not resp.text:
                return False, None
            result = _extract_hn_dupe_target(resp.text, sid)
            _save_cached_hn_dupe_target(sid, result)
            return result
    except Exception as exc:
        logging.debug("Failed to fetch HN dupe marker for %s: %s", sid, exc)
        return False, None
    return False, None


async def filter_top_ranked_hn_dupes(
    ranked: list[RankResult],
    cands: list[Story],
    exclude_ids: set[int],
    count: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[RankResult]:
    """Drop [dupe]-marked HN submissions from the final page results."""
    if not ranked:
        return ranked

    _ = exclude_ids
    check_limit = min(len(ranked), count)
    checked_positions = [
        pos
        for pos, result in enumerate(ranked[:check_limit])
        if cands[result.index].is_hn and cands[result.index].id > 0
    ]
    if not checked_positions:
        return ranked

    checked_by_index: dict[int, tuple[bool, int | None]] = {}
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for idx, pos in enumerate(checked_positions):
            if idx:
                await asyncio.sleep(0.2)
            checked_by_index[ranked[pos].index] = await _fetch_hn_dupe_target(
                client, cands[ranked[pos].index].id
            )
            if progress_callback:
                progress_callback(idx + 1, len(checked_positions))

    filtered: list[RankResult] = []
    skipped = 0
    for result in ranked:
        info = checked_by_index.get(result.index)
        if info is not None:
            is_dupe, _target_id = info
            if is_dupe:
                skipped += 1
                continue
        filtered.append(result)

    if skipped:
        print(f"[+] Filtered {skipped} duplicate HN submissions from final results")
    return filtered


async def refresh_hn_story_metadata(
    stories: list[Story],
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Refresh volatile HN metadata used directly in dashboard cards."""
    hn_stories = [story for story in stories if story.is_hn and story.id > 0]
    if not hn_stories:
        return

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for idx, story in enumerate(hn_stories):
            try:
                resp = await client.get(HN_FIREBASE_ITEM_URL.format(sid=story.id))
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict):
                        descendants = payload.get("descendants")
                        if isinstance(descendants, int):
                            story.comment_count = descendants
                        score = payload.get("score")
                        if isinstance(score, int):
                            story.score = score
            except Exception as exc:
                logging.debug("Failed to refresh HN metadata for %s: %s", story.id, exc)
            if progress_callback:
                progress_callback(idx + 1, len(hn_stories))


HTML_TEMPLATE: str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HN Rerank | {{ username }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        {% raw %}
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        hn: '#ff6600',
                    }
                }
            }
        }
        {% endraw %}
    </script>
    <style type="text/tailwindcss">
        @layer base {
            body { @apply bg-stone-50 text-stone-800 antialiased; }
        }
        .story-card { @apply bg-white border border-stone-200 rounded-lg p-2.5 shadow-sm transition-all hover:border-hn hover:shadow-md h-full flex flex-col; }
        .story-card.rss-story { @apply border-amber-200 bg-amber-50/50; }
        .story-card.feedback-removing { opacity: 0; transform: translateY(6px) scale(0.98); pointer-events: none; }
        .rss-badge { @apply px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 text-[10px] font-bold; }
        .cluster-chip { @apply px-1.5 py-0.5 bg-stone-50 border border-stone-200 rounded text-[10px] font-medium text-stone-600 hover:border-hn hover:text-hn transition-colors cursor-default whitespace-nowrap; }
    </style>
</head>
<body class="p-2 md:p-4 bg-stone-100">
    <div class="max-w-7xl mx-auto">
        <header class="mb-4 border-b border-stone-300 pb-3 flex items-end justify-between bg-white p-4 rounded-lg shadow-sm">
            <div>
                <h1 class="text-2xl font-black text-stone-900 tracking-tight">
                    HN <span class="text-hn">Rerank</span>
                </h1>
                <p class="text-stone-500 text-xs">@{{ username }} &bull; <a href="clusters.html" class="text-hn hover:underline">{{ n_clusters }} interest clusters</a></p>
            </div>
            <div class="flex items-end gap-3">
                <label for="sort-mode" class="flex flex-col gap-1 text-[10px] text-stone-400 font-mono">
                    <span>SORT</span>
                    <select id="sort-mode" class="rounded border border-stone-200 bg-white px-2 py-1 text-xs text-stone-700">
                        <option value="current" selected>Similarity</option>
                        <option value="date">Date</option>
                    </select>
                </label>
                <p class="text-[10px] text-stone-400 font-mono">{{ timestamp }}</p>
            </div>
        </header>

        <div id="stories-grid" class="grid gap-3 items-start grid-cols-[repeat(auto-fit,minmax(280px,1fr))]">
            {{ stories_html | safe }}
        </div>

        <footer class="mt-8 py-4 border-t border-stone-200 text-center text-stone-400 text-xs">
            HN Rerank &bull; Local Semantic Analysis
        </footer>
    </div>
    <script>
        (() => {
            const grid = document.getElementById('stories-grid');
            const sortMode = document.getElementById('sort-mode');
            if (!grid || !sortMode) return;

            const cards = Array.from(grid.querySelectorAll('[data-rank-index][data-story-time]'));
            const compareCurrent = (a, b) =>
                Number(a.dataset.rankIndex) - Number(b.dataset.rankIndex);
            const compareDate = (a, b) => {
                const timeDiff = Number(b.dataset.storyTime) - Number(a.dataset.storyTime);
                if (timeDiff !== 0) return timeDiff;
                return compareCurrent(a, b);
            };

            const renderSort = (mode) => {
                const sorted = [...cards].sort(mode === 'date' ? compareDate : compareCurrent);
                for (const card of sorted) {
                    grid.appendChild(card);
                }
            };

            sortMode.addEventListener('change', () => renderSort(sortMode.value));
            renderSort(sortMode.value);
        })();

        (() => {
            const TOKEN_KEY = 'hnRerankFeedbackToken';
            const ACTED_KEYS_KEY = 'hnRerankActedFeedbackKeys';
            const FEEDBACK_URL = window.HN_RERANK_FEEDBACK_URL || '/api/feedback';
            const cards = Array.from(document.querySelectorAll('[data-feedback-key]'));
            if (!cards.length) return;
            const cardsByKey = new Map(cards.map((card) => [card.dataset.feedbackKey, card]));

            const loadActedKeys = () => {
                try {
                    const raw = localStorage.getItem(ACTED_KEYS_KEY);
                    const parsed = raw ? JSON.parse(raw) : [];
                    return new Set(Array.isArray(parsed) ? parsed.filter(Boolean) : []);
                } catch (error) {
                    return new Set();
                }
            };

            const saveActedKeys = (keys) => {
                localStorage.setItem(ACTED_KEYS_KEY, JSON.stringify([...keys]));
            };

            const actedKeys = loadActedKeys();

            const rememberActedKey = (key) => {
                if (!key) return;
                actedKeys.add(key);
                saveActedKeys(actedKeys);
            };

            const forgetActedKey = (key) => {
                if (!key) return;
                actedKeys.delete(key);
                saveActedKeys(actedKeys);
            };

            const hideCard = (card) => {
                card.remove();
            };

            const setCardAction = (card, action) => {
                card.dataset.feedbackAction = action || '';
                const buttons = card.querySelectorAll('[data-feedback-button]');
                for (const button of buttons) {
                    const buttonAction = button.dataset.feedbackButton;
                    const active = buttonAction === action;
                    button.classList.toggle('bg-hn', active && action === 'up');
                    button.classList.toggle('bg-emerald-600', active && action === 'neutral');
                    button.classList.toggle('bg-stone-800', active && action === 'down');
                    button.classList.toggle('text-white', active);
                    button.classList.toggle('bg-white', !active);
                    button.classList.toggle('text-stone-500', !active);
                    button.classList.toggle('text-emerald-700', !active && buttonAction === 'neutral');
                }
            };

            const setStatus = (card, message, failed = false) => {
                const status = card.querySelector('[data-feedback-status]');
                if (!status) return;
                status.textContent = message || '';
                status.classList.toggle('text-red-600', failed);
                status.classList.toggle('text-stone-400', !failed);
            };

            const removeCard = (card) => {
                card.classList.add('feedback-removing');
                window.setTimeout(() => card.remove(), 220);
            };

            const getToken = () => {
                let token = localStorage.getItem(TOKEN_KEY);
                if (!token) {
                    token = window.prompt('Dashboard feedback token');
                    if (token) localStorage.setItem(TOKEN_KEY, token);
                }
                return token;
            };

            const hidePreviouslyActedCards = () => {
                for (const card of cards) {
                    const action = card.dataset.feedbackAction || '';
                    if (action === 'up' || action === 'neutral' || action === 'down' || actedKeys.has(card.dataset.feedbackKey)) {
                        hideCard(card);
                    }
                }
            };

            const syncServerFeedback = async () => {
                const token = localStorage.getItem(TOKEN_KEY);
                if (!token) return;
                try {
                    const response = await fetch(FEEDBACK_URL, {
                        method: 'GET',
                        headers: {
                            'X-HN-RERANK-FEEDBACK-TOKEN': token,
                        },
                    });
                    const payload = await response.json();
                    if (!response.ok || !payload.records) return;
                    for (const [key, record] of Object.entries(payload.records)) {
                        if (!record || !['up', 'neutral', 'down'].includes(record.action)) continue;
                        rememberActedKey(key);
                        const card = cardsByKey.get(key);
                        if (card) hideCard(card);
                    }
                } catch (error) {
                    return;
                }
            };

            hidePreviouslyActedCards();
            syncServerFeedback();

            for (const card of cards) {
                setCardAction(card, card.dataset.feedbackAction || '');
                for (const button of card.querySelectorAll('[data-feedback-button]')) {
                    button.addEventListener('click', async (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        const previousAction = card.dataset.feedbackAction || '';
                        const buttonAction = button.dataset.feedbackButton;
                        const nextAction = previousAction === buttonAction ? 'clear' : buttonAction;
                        const token = getToken();
                        if (!token) return;

                        setCardAction(card, nextAction === 'clear' ? '' : nextAction);
                        setStatus(card, 'Saving...');

                        try {
                            const response = await fetch(FEEDBACK_URL, {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'X-HN-RERANK-FEEDBACK-TOKEN': token,
                                },
                                body: JSON.stringify({
                                    id: Number(card.dataset.storyId),
                                    source: card.dataset.storySource,
                                    title: card.dataset.storyTitle,
                                    url: card.dataset.storyUrl || null,
                                    discussion_url: card.dataset.storyDiscussionUrl || null,
                                    text_content: card.dataset.storyTextContent || card.dataset.storyTitle,
                                    time: Number(card.dataset.storyTime),
                                    score: Number(card.dataset.storyScore),
                                    comment_count: card.dataset.storyCommentCount === '' ? null : Number(card.dataset.storyCommentCount),
                                    action: nextAction,
                                }),
                            });
                            const payload = await response.json();
                            if (!response.ok || !payload.ok) {
                                throw new Error(payload.error || `HTTP ${response.status}`);
                            }
                            const mirrorFailed = payload.record && payload.record.hn_mirror_status === 'failed';
                            setStatus(
                                card,
                                mirrorFailed ? 'Saved locally; HN sync failed' : 'Saved',
                                Boolean(mirrorFailed),
                            );
                            if (nextAction !== 'clear') {
                                rememberActedKey(card.dataset.feedbackKey);
                                removeCard(card);
                            } else {
                                forgetActedKey(card.dataset.feedbackKey);
                            }
                        } catch (error) {
                            setCardAction(card, previousAction);
                            setStatus(card, 'Save failed', true);
                        }
                    });
                }
            }
        })();
    </script>
</body>
</html>
"""

CLUSTERS_PAGE_TEMPLATE: str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interest Clusters | {{ username }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        {% raw %}
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        hn: '#ff6600',
                    }
                }
            }
        }
        {% endraw %}
    </script>
    <style type="text/tailwindcss">
        @layer base {
            body { @apply bg-stone-50 text-stone-800 antialiased; }
        }
        .cluster-card { @apply bg-white border border-stone-200 rounded-lg shadow-sm; }
    </style>
</head>
<body class="p-2 md:p-4">
    <div class="max-w-5xl mx-auto">
        <header class="mb-4 border-b border-stone-200 pb-3 flex items-end justify-between">
            <div>
                <h1 class="text-2xl font-black text-stone-900 tracking-tight">
                    Interest <span class="text-hn">Clusters</span>
                </h1>
                <p class="text-stone-500 text-xs">@{{ username }} &bull; {{ n_signals }} signals &rarr; {{ n_clusters }} clusters</p>
            </div>
            <p class="text-[10px] text-stone-400 font-mono">{{ timestamp }}</p>
        </header>

        <div class="grid gap-4 md:grid-cols-2">
            {{ clusters_html | safe }}
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
        <h2 class="font-bold text-stone-700">{{ cluster_name }}</h2>
        <span class="text-xs text-stone-400">{{ count }} stories</span>
    </div>
    {% if keywords %}
    <div class="px-3 py-1 bg-stone-50 border-b border-stone-100 text-xs text-stone-500 font-mono italic line-clamp-2" title="{{ keywords }}">
        {{ keywords }}
    </div>
    {% endif %}
    <ul class="divide-y divide-stone-100">
        {{ stories_html | safe }}
    </ul>
</div>
"""

CLUSTER_STORY_TEMPLATE: str = """
<li class="px-3 py-2 hover:bg-stone-50">
    <a href="{{ hn_url }}" target="_blank" class="text-sm text-stone-700 hover:text-hn transition-colors line-clamp-2">
        {{ title }}
    </a>
    <div class="flex items-center gap-2 mt-0.5">
        <span class="text-[10px] text-stone-400">{{ points }} pts</span>
        <span class="text-[10px] text-stone-400">{{ time_ago }}</span>
    </div>
</li>
"""


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
    feedback_positive_hn_ids = {
        story.id
        for story in feedback_positive_stories
        if story.source == "hn" and story.id > 0
    }
    feedback_negative_hn_ids = {
        story.id
        for story in feedback_negative_stories
        if story.source == "hn" and story.id > 0
    }

    pos_baseline = data["pos"] - feedback_negative_hn_ids - feedback_positive_hn_ids
    pos_ids = list(feedback_positive_hn_ids) + list(pos_baseline)
    pos_ids = pos_ids[:signal_limit]

    neg_ids: list[int] = []
    if use_hidden_signal:
        neg_baseline = (
            data["hidden"] - feedback_positive_hn_ids - feedback_negative_hn_ids
        )
        neg_ids = list(feedback_negative_hn_ids) + list(neg_baseline)
        neg_ids = neg_ids[:signal_limit]
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
    cand_emb = rerank.get_cluster_embeddings(
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

    desired_external = round(count * 0.2) + 5
    available_external = sum(1 for r in ranked if is_external_result(r))
    available_hn = len(ranked) - available_external
    min_external = max(0, count - available_hn)
    max_external = min(count, available_external)
    target_external = min(max(desired_external, min_external), max_external)
    target_hn = count - target_external

    # Select external with diversity: start with strict per-source quota and relax if needed
    external_candidates = [r for r in ranked if is_external_result(r)]
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

    # Select HN
    hn_candidates = [r for r in ranked if not is_external_result(r)]
    selected_hn = hn_candidates[:target_hn]

    # Combine and sort by the active final score.
    selected_results = selected_external + selected_hn
    selected_results.sort(key=lambda x: x.hybrid_score, reverse=True)
    return selected_results


STORY_CARD_TEMPLATE: str = """
<div class="story-card group relative{% if is_external %} rss-story{% endif %}" data-rank-index="{{ rank_index }}" data-story-time="{{ story_time }}" data-story-id="{{ story_id }}" data-story-source="{{ story_source }}" data-story-title="{{ title }}" data-story-url="{{ story_url or '' }}" data-story-discussion-url="{{ hn_url or '' }}" data-story-text-content="{{ text_content }}" data-story-score="{{ story_score }}" data-story-comment-count="{{ story_comment_count if story_comment_count is not none else '' }}" data-feedback-key="{{ feedback_key }}" data-feedback-action="{{ feedback_action or '' }}">
    {% if card_url %}
    <a href="{{ card_url }}" target="_blank" class="absolute inset-0 z-10 rounded-lg" aria-label="{{ card_aria_label }}"></a>
    {% endif %}
    <div class="flex items-center gap-2 mb-0.5 flex-wrap{% if card_url %} relative z-20 pointer-events-none{% endif %}">
        <span class="px-1.5 py-0.5 rounded bg-hn/10 text-hn text-[10px] font-bold" title="Hybrid Match Score">
            {{ score }}%
        </span>
        {% if source_badge %}
        <span class="rss-badge">{{ source_badge }}</span>
        {% endif %}
        {% if cluster_name %}
        <span class="cluster-chip">{{ cluster_name }}</span>
        {% endif %}
        <span class="text-[10px] text-stone-400 font-mono">{{ points }} pts</span>
        <span class="text-[10px] text-stone-400 font-mono">{{ time_ago }}</span>
        <span class="ml-auto flex items-center gap-1 pointer-events-auto" aria-label="Dashboard feedback">
            <button type="button" class="h-6 w-6 rounded border border-stone-200 bg-white text-xs text-stone-500 hover:border-hn hover:text-hn" title="Upvote for future dashboards" data-feedback-button="up">▲</button>
            <button type="button" class="h-6 w-6 rounded border border-stone-200 bg-white text-xs text-emerald-700 hover:border-emerald-500 hover:text-emerald-700" title="Mark neutral for future dashboards" data-feedback-button="neutral">✓</button>
            <button type="button" class="h-6 w-6 rounded border border-stone-200 bg-white text-xs text-stone-500 hover:border-stone-700 hover:text-stone-900" title="Downvote for future dashboards" data-feedback-button="down">▼</button>
        </span>
        <span class="text-[10px] text-stone-400 font-mono pointer-events-none" data-feedback-status></span>
    </div>
    <h2 class="text-sm font-semibold text-stone-900 leading-snug mb-1{% if card_url %} relative z-20 pointer-events-none{% endif %}">
        <a href="{{ url }}" target="_blank" class="hover:text-hn transition-colors pointer-events-auto">{{ title }}</a>
        {% if hn_url %}
        <a href="{{ hn_url }}" target="_blank" class="ml-2 text-xs font-medium text-stone-900 hover:text-hn transition-colors pointer-events-auto" title="Comments">💬{% if comment_count is not none %} {{ comment_count }}{% endif %}</a>
        {% elif comment_count is not none %}
        <span class="ml-2 text-xs font-medium text-stone-500" title="Comments">💬 {{ comment_count }}</span>
        {% endif %}
    </h2>
    {% if tldr %}
    <div class="text-sm text-stone-600 bg-stone-50 p-2 rounded border border-stone-100 leading-relaxed whitespace-pre-line{% if card_url %} relative z-20 pointer-events-none{% endif %}">{{ tldr }}</div>
    {% endif %}
</div>
"""

_JINJA_ENV: Environment = Environment(autoescape=True)
_STORY_TEMPLATE = _JINJA_ENV.from_string(STORY_CARD_TEMPLATE)
_INDEX_TEMPLATE = _JINJA_ENV.from_string(HTML_TEMPLATE)
_CLUSTER_STORY_TEMPLATE = _JINJA_ENV.from_string(CLUSTER_STORY_TEMPLATE)
_CLUSTER_CARD_TEMPLATE = _JINJA_ENV.from_string(CLUSTER_CARD_TEMPLATE)
_CLUSTERS_TEMPLATE = _JINJA_ENV.from_string(CLUSTERS_PAGE_TEMPLATE)


def generate_story_html(story: StoryDisplay) -> str:
    link_url = story.url or story.hn_url or "#"
    if story.is_hn:
        card_url = story.hn_url
        card_aria_label = (
            f"Open comments for {story.title}" if story.hn_url else None
        )
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
        text_content=story.text_content[:2000],
        story_score=story.points,
        story_comment_count=story.comment_count,
        feedback_key=feedback_key(story.source, story.id, story.url),
        feedback_action=story.feedback_action,
        hybrid_score=story.hybrid_score,
        semantic_score=story.semantic_score,
        knn_score=story.knn_score,
        max_sim_score=story.max_sim_score,
        max_cluster_score=story.max_cluster_score,
        card_url=card_url,
        card_aria_label=card_aria_label,
        url=link_url,
        title=story.title,
        hn_url=story.hn_url,
        comment_count=story.comment_count,
        tldr=story.tldr,
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
    parser.add_argument(
        "-c", "--count", type=int, help="Number of stories to show"
    )
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

    # Create unified config from TOML and CLI overrides
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
        config = replace(config, clustering=replace(config.clustering, default_count=args.clusters, max_clusters=args.clusters))
    if args.open_index_archive or args.bigquery_archive:
        config = replace(
            config,
            archive=replace(config.archive, open_index_enabled=True),
        )

    config = replace(config, llm=replace(config.llm, provider=provider_choice))

    os.environ["LLM_PROVIDER"] = config.llm.provider

    if config.debug_clusters:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    if not config.username:
        console.print("[red][bold][-] Error:[/bold] username is required.[/red]")
        console.print("    Provide it as an argument or in hn_rerank.toml")
        raise SystemExit(1)

    # Initialize model early
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
                results: list[Story] = []
                progress.update(
                    p_task, description=f"[*] Fetching {label} ({len(ids)} items)..."
                )

                if not ids:
                    progress.update(overall_task, advance=weight_share)
                    return []

                # Calculate step sizes for both bars
                step = 100.0 * (weight_share / PROGRESS_WEIGHTS["profile"]) / len(ids)
                overall_step = weight_share / len(ids)

                for res in asyncio.as_completed(
                    [fetch_story(hn.client, sid) for sid in ids]
                ):
                    s: Story | None = await res
                    if s:
                        results.append(s)
                    progress.update(p_task, advance=step)
                    progress.update(overall_task, advance=overall_step)
                return results

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

        # Track overall advancement for embeddings to avoid double-counting
        last_e_completed = 0

        def emb_cb(curr: int, total: int) -> None:
            nonlocal last_e_completed
            progress.update(e_task, total=total, completed=curr)
            # Advance overall bar proportionally to the weight of this phase
            if total > 0:
                delta = curr - last_e_completed
                overall_advance = (delta / total) * PROGRESS_WEIGHTS["emb_pref"]
                progress.update(overall_task, advance=overall_advance)
                last_e_completed = curr

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

        last_e_completed = 0
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
            last_ce_completed = 0

            def cluster_emb_cb(curr: int, total: int) -> None:
                nonlocal last_ce_completed
                progress.update(ce_task, total=total, completed=curr)
                if total > 0:
                    delta = curr - last_ce_completed
                    overall_advance = (delta / total) * PROGRESS_WEIGHTS["emb_clust"]
                    progress.update(overall_task, advance=overall_advance)
                    last_ce_completed = curr

            cluster_emb = rerank.get_cluster_embeddings(
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
            last_n_completed = 0

            def name_cb(curr: int, total: int) -> None:
                nonlocal last_n_completed
                progress.update(name_task, completed=curr)
                if total > 0:
                    delta = curr - last_n_completed
                    overall_advance = (delta / total) * PROGRESS_WEIGHTS["naming"]
                    progress.update(overall_task, advance=overall_advance)
                    last_n_completed = curr

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
                        debug_path = config.output_path.with_name("cluster_name_debug.json")
                        debug_path.parent.mkdir(parents=True, exist_ok=True)
                    cluster_profiles = await llm_utils.generate_batch_cluster_names(
                        clusters_for_naming,
                        progress_callback=name_cb,
                        debug_path=debug_path,
                    )
                    for cid, profile in cluster_profiles.items():
                        profile_dict = cast(dict[str, str], profile)
                        cluster_names[cid] = profile_dict["name"]
                        cluster_keywords[cid] = profile_dict["keywords"]
                    
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
        candidate_phase_order = list(candidate_phase_weights)
        candidate_phase_completed = dict.fromkeys(candidate_phase_weights, 0.0)
        last_c_completed = 0.0

        def cand_cb(event: CandidateProgress) -> None:
            nonlocal last_c_completed
            phase = event["phase"]
            if phase == "complete":
                completed = 100.0
                description = "[*] Finalizing candidates..."
            else:
                total = max(event["total"], 1)
                phase_fraction = min(max(event["current"] / total, 0.0), 1.0)
                candidate_phase_completed[phase] = (
                    candidate_phase_weights[phase] * phase_fraction
                )
                phase_index = candidate_phase_order.index(phase)
                for prior_phase in candidate_phase_order[:phase_index]:
                    candidate_phase_completed[prior_phase] = candidate_phase_weights[
                        prior_phase
                    ]
                completed = min(sum(candidate_phase_completed.values()), 99.0)
                description = f"[*] {event['label']}..."
            progress.update(c_task, completed=completed, description=description)
            delta = completed - last_c_completed
            if delta > 0:
                progress.update(
                    overall_task,
                    advance=(delta / 100.0) * PROGRESS_WEIGHTS["candidates"],
                )
                last_c_completed = completed

        # Exclude everything we've already interacted with
        exclude_ids: set[int] = data["favorites"] | data["upvoted"] | data["hidden"]
        exclude_ids |= feedback_hn_ids
        exclude_urls: set[str] = set()
        exclude_urls |= data.get("hidden_urls", set())
        exclude_urls |= data.get("favorites_urls", set())
        exclude_urls |= data.get("upvoted_urls", set())
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
        rank_phase_order = list(rank_phase_weights)
        rank_phase_completed = dict.fromkeys(rank_phase_weights, 0.0)
        last_r_completed = 0.0

        def rank_cb(event: rerank.RankProgress) -> None:
            nonlocal last_r_completed
            phase = event["phase"]
            if phase == "complete":
                completed = 100.0
                description = "[*] Finalizing rerank..."
            else:
                total = max(event["total"], 1)
                phase_fraction = min(max(event["current"] / total, 0.0), 1.0)
                rank_phase_completed[phase] = (
                    rank_phase_weights[phase] * phase_fraction
                )
                phase_index = rank_phase_order.index(phase)
                for prior_phase in rank_phase_order[:phase_index]:
                    rank_phase_completed[prior_phase] = rank_phase_weights[
                        prior_phase
                    ]
                completed = min(sum(rank_phase_completed.values()), 99.0)
                description = f"[*] {event['label']}..."
            progress.update(r_task, completed=completed, description=description)
            delta = completed - last_r_completed
            if delta > 0:
                progress.update(
                    overall_task,
                    advance=(delta / 100.0) * PROGRESS_WEIGHTS["rank"],
                )
                last_r_completed = completed

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
        prep_phase_completed = dict.fromkeys(prep_phase_weights, 0.0)
        last_prep_completed = 0.0

        def update_prep(
            phase: str,
            current: int,
            total: int,
            description: str,
        ) -> None:
            nonlocal last_prep_completed
            phase_fraction = min(max(current / max(total, 1), 0.0), 1.0)
            prep_phase_completed[phase] = prep_phase_weights[phase] * phase_fraction
            completed = min(sum(prep_phase_completed.values()), 100.0)
            progress.update(prep_task, completed=completed, description=description)
            delta = completed - last_prep_completed
            if delta > 0:
                progress.update(
                    overall_task,
                    advance=(delta / 100.0) * PROGRESS_WEIGHTS["prepare"],
                )
                last_prep_completed = completed

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
            ranked, cands, cluster_labels, cluster_names, cand_cluster_map, config.count
        )
        update_prep("select", 1, 1, "[*] Selecting final stories...")
        selected_results = await filter_top_ranked_hn_dupes(
            selected_results,
            cands,
            exclude_ids=exclude_ids,
            count=config.count,
            progress_callback=lambda curr, total: update_prep(
                "dupes",
                curr,
                total,
                "[*] Checking duplicate HN submissions...",
            ),
        )
        update_prep("dupes", 1, 1, "[*] Checking duplicate HN submissions...")

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
                result, cluster_labels, cand_cluster_map, config.semantic.match_threshold
            )
            cluster_name = resolve_cluster_name(
                cluster_names, cid, allow_empty_fallback=s.is_external
            )
            discussion_url = s.discussion_url
            if discussion_url is None and s.is_hn and s.id > 0:
                discussion_url = f"https://news.ycombinator.com/item?id={s.id}"
            return StoryDisplay(
                id=s.id,
                match_percent=format_match_percent(result.max_cluster_score),
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
                hybrid_score=result.hybrid_score,
                semantic_score=result.semantic_score,
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
            last_t_completed = 0

            def tldr_cb(curr: int, total: int) -> None:
                nonlocal last_t_completed
                progress.update(llm_task, completed=curr)
                if total > 0:
                    delta = curr - last_t_completed
                    overall_advance = (delta / total) * PROGRESS_WEIGHTS["tldr"]
                    progress.update(overall_task, advance=overall_advance)
                    last_t_completed = curr

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

    # --- HTML Generation starts here ---
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
                    "hybrid_score": result.hybrid_score,
                    "semantic_score": result.semantic_score,
                    "knn_score": result.knn_score,
                    "max_cluster_score": result.max_cluster_score,
                    "max_sim_score": result.max_sim_score,
                }
            )
        debug_path.write_text(json.dumps(debug_rows, indent=2))
        print(f"[+] Score breakdown saved to: {os.path.abspath(debug_path)}")

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
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stories_html=Markup(stories_html),
    )

    try:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(final_html)
        print(f"[+] Dashboard saved to: {os.path.abspath(config.output_path)}")

        # Write clusters page
        if clusters_page_html:
            clusters_path = config.output_path.with_name("clusters.html")
            clusters_path.write_text(clusters_page_html)
            print(f"[+] Clusters saved to: {os.path.abspath(clusters_path)}")
    except OSError as e:
        print(f"[!] Error writing output file: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
