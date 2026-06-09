# Telemetry Features for Ordinal Ranking

Activate **Path 2** from `plans/implicit_ctr_signal.md` (Aggregated CTR Features).
The old deferral ("~6 months soak") is obsolete — we have 3,732 events / 210 clicks /
146 unique URLs over 8 days. Enough variance to be useful.

## Two feature groups, one SQL query

Both groups are computed from `telemetry_events` in a single `GROUP BY` query:

### Per-story features (specific — penalize/boost individual stories)

| Feature | SQL | Transform |
|---------|-----|-----------|
| `impression_count` | `COUNT(*) WHERE event='impression'` | `log1p` |
| `click_count` | `COUNT(*) WHERE event='click'` | `log1p` |
| `click_ratio` | `clicks / MAX(impressions, 1)` | clamp `[0, 1]` |
| `days_since_last_impression` | `MAX(server_ts) → now - ts` | cap at 30, invert `[0, 1]` |

These solve the concrete "story appeared 81× and I never clicked it" problem.

### Domain-level features (generalize — handles new stories from known domains)

| Feature | SQL | Transform |
|---------|-----|-----------|
| `domain_ctr` | `SUM(clicks) / SUM(impressions)` per domain | clamp `[0, 1]` |
| `domain_impression_count` | `COUNT(*)` per domain | `log1p` |

These capture "I always click arxiv papers, never click digg" without an
explicit vote on each story. New stories from known domains get a score
from the first regen.

## File changes

| File | Change | Lines |
|------|--------|-------|
| `api/telemetry_features.py` (new) | One SQL query, return `(per_story_stats, domain_stats)` | ~45 |
| `api/rerank.py` | Import, call once before feature loop, register 6 entries in `METADATA_FEATURES` | ~25 |
| `api/config.py` | Add 6 names to `ClassifierConfig.features` default | ~2 |
| **Total** | | **~72** |

### `api/telemetry_features.py` design

```python
"""Aggregate impression/click stats from telemetry SQLite for use as ranking features."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DB_PATH = Path(".cache/user_feedback/feedback_events.sqlite")


@dataclass(frozen=True)
class StoryImpressionStats:
    impression_count: int = 0
    click_count: int = 0
    click_ratio: float = 0.0
    days_since_last_impression: float = 30.0


@dataclass(frozen=True)
class DomainImpressionStats:
    domain_ctr: float = 0.0
    domain_impression_count: int = 0


def fetch_impression_stats(
    story_ids: set[int],
) -> tuple[dict[int, StoryImpressionStats], dict[str, DomainImpressionStats]]:
    """Query telemetry_events and return per-story + per-domain stats.

    Returns ({story_id: stats}, {domain: stats}).  Missing keys = all-zeros.
    """
    if not DB_PATH.exists():
        return {}, {}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # Per-story aggregation
        story_rows = conn.execute(
            """
            SELECT story_id,
                   SUM(CASE WHEN event='impression' THEN 1 ELSE 0 END) as impressions,
                   SUM(CASE WHEN event='click' THEN 1 ELSE 0 END) as clicks,
                   MAX(server_ts) as last_seen
            FROM telemetry_events
            GROUP BY story_id
            """
        ).fetchall()

        now = time.time()
        per_story: dict[int, StoryImpressionStats] = {}
        for row in story_rows:
            imps = row["impressions"] or 0
            clks = row["clicks"] or 0
            last = row["last_seen"] or 0.0
            days_since = min((now - last) / 86400.0, 30.0) if last > 0 else 30.0
            per_story[row["story_id"]] = StoryImpressionStats(
                impression_count=imps,
                click_count=clks,
                click_ratio=clks / max(imps, 1),
                days_since_last_impression=days_since,
            )

        # Domain-level aggregation (extract domain from url + external links)
        domain_rows = conn.execute(
            """
            SELECT url,
                   SUM(CASE WHEN event='click' THEN 1 ELSE 0 END) as clicks,
                   SUM(CASE WHEN event='impression' THEN 1 ELSE 0 END) as impressions
            FROM telemetry_events
            WHERE url IS NOT NULL AND url != ''
            GROUP BY url
            """
        ).fetchall()

        per_domain: dict[str, DomainImpressionStats] = {}
        for row in domain_rows:
            imps = row["impressions"] or 0
            clks = row["clicks"] or 0
            domain = _extract_domain(row["url"])
            if domain:
                existing = per_domain.get(domain)
                if existing:
                    imps += existing.domain_impression_count
                    clks = int(existing.domain_ctr * existing.domain_impression_count + clks)
                per_domain[domain] = DomainImpressionStats(
                    domain_ctr=clks / max(imps, 1),
                    domain_impression_count=imps,
                )

        return per_story, per_domain
    finally:
        conn.close()


def _extract_domain(url: str) -> str | None:
    """Pull domain from a URL string."""
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None
```

### `api/rerank.py` changes

```python
from api.telemetry_features import (
    fetch_impression_stats,
    StoryImpressionStats,
    DomainImpressionStats,
)
```

Before the metadata feature loop in `rank_stories` (around line 1160):

```python
_impression_stats, _domain_stats = fetch_impression_stats({s.id for s in stories})
```

Two new entries in `METADATA_FEATURES` (around line 305):

```python
METADATA_FEATURES["impression_count"] = _meta_impression_count
METADATA_FEATURES["click_count"] = _meta_click_count
METADATA_FEATURES["click_ratio"] = _meta_click_ratio
METADATA_FEATURES["days_since_last_impression"] = _meta_days_since_last_impression
METADATA_FEATURES["domain_ctr"] = _meta_domain_ctr
METADATA_FEATURES["domain_impression_count"] = _meta_domain_impression_count
```

Each function follows the existing `MetadataFeatureFn` signature:
`(list[Story], float | None) -> NDArray[np.float32]`, indexing into the
pre-fetched dicts.

### `api/config.py` changes

Extend default `ClassifierConfig.features`:

```python
features: tuple[str, ...] = (
    "centroid", "pos_knn", "neg_knn", "closest_margin",
    "title_len", "text_len", "has_url", "is_github", "is_pdf",
    "comments_count", "is_hn", "source_trust",
    "impression_count", "click_count", "click_ratio",
    "days_since_last_impression", "domain_ctr", "domain_impression_count",
)
```

## Edge cases

| Case | Behavior |
|------|----------|
| No SQLite file | `fetch_impression_stats` returns `({}, {})` → all features = 0 |
| Telemetry table empty | Per-story stats all zero; per-domain stats all zero |
| New story, never seen | Zero for all story features; domain features from known domains still work |
| Story with high impressions AND clicks | `click_ratio` ~ 1.0 — model learns this is positive (not "high impressions = bad") |
| Domain never seen before | `domain_ctr = 0`, `domain_impression_count = 0` — no penalty |
| Very long domain with many URLs | `domain_impression_count` is `log1p`-transformed, prevents outlier domination |

## How features reach the ordinal model

```
fetch_impression_stats()
        │
        ▼
  METADATA_FEATURES (rerank.py)
  6 feature functions → 6 columns
        │
        ▼
  _classifier_metadata_features()
  hstack with other metadata columns
        │
        ▼
  rank_stories: feature matrix hstack
  (raw embeddings + derived similarity + metadata + telemetry)
        │
        ▼
  train_model_from_matrix / predict
```

The ordinal model learns weights for each telemetry feature just like it
learns weights for `log_points` or `source_trust`. No hard-coded rules,
no sample weights, no pipeline changes.

## Relationship to existing plan

`plans/implicit_ctr_signal.md` defines three paths:

| Path | What | Status |
|------|------|--------|
| Path 1 | Implicit labels with sample weights | Deferred — training pipeline changes, complex |
| **Path 2** | **Aggregated CTR as features** | **Activated by this plan** |
| Path 3 | Dwell time instrumentation | Future |

This plan activates Path 2. It does **not** replace Path 1 — the two are
complementary (features guide the model; labels train it). When we have
sufficient data to justify the training pipeline changes, Path 1 can still
be implemented on top.

## Verification

1. `uv run pytest -q -n auto` — all tests pass
2. `uv run ruff check .` — no lint errors
3. `uv run ty check api/ tests/` — no type errors
4. After regen: a story with 10+ impressions and 0 clicks ranks lower than
   a semantically similar story with no impression history
5. After regen: a story from a domain with 30% CTR ranks higher than one
   from a domain with 0% CTR, all else equal
