# Caching

This file documents the main caches used by the repo.

## Overview

The project caches several different things:

- story payloads
- candidate id lists
- user HN signals
- embedding and cluster-embedding work
- CE scores
- HN dupe checks
- dashboard feedback

These caches have different lifetimes and are used by different paths.

## Story Cache

Code:

- `api/fetching.py`

Directory:

- `STORY_CACHE_DIR` from `api.constants`

Purpose:

- cache fetched HN story payloads and hydrated content

Used by:

- candidate fetch
- evaluator story loading
- dashboard generation

## Candidate Cache

Code:

- `api/fetching.py`

Directory:

- `CANDIDATE_CACHE_DIR` from `api.constants`

Purpose:

- cache candidate id lists for ranking fetches

Used by:

- `get_best_stories`

## User Signal Cache

Code:

- `api/client.py`

Directory:

- `USER_CACHE_DIR` from `api.constants`

Example payload:

- `<user>.json`

Purpose:

- cache scraped HN user signals:
  - upvoted
  - favorites
  - hidden
  - associated URLs

## Embedding Cache

Code:

- `api/rerank.py`

Directories:

- `EMBEDDING_CACHE_DIR`
- `CLUSTER_EMBEDDING_CACHE_DIR`

Purpose:

- avoid recomputing expensive embedding work
- speed up repeated ranking passes

## Cross-Encoder Score Cache

Code:

- `api/rerank.py`

Directory:

- `.cache/cross_encoder_scores`

Purpose:

- cache per-candidate CE scores keyed by:
  - candidate text
  - query text
  - model fingerprint

Important note:

- this cache is about CE score reuse, not final ranking reuse

## HN Dupe Cache

Code:

- `generate_html.py`

Directory:

- `.cache/hn_dupes`

Payload:

- one JSON file per HN story id
- stores:
  - `is_dupe`
  - `target_id`

Current TTLs:

- dupe result: `15 days`
- non-dupe result: `1 day`

This cache exists because live item-page dupe checks are slow and rate-limited.

## Feedback Store

Code:

- `api/feedback.py`

Path:

- `.cache/user_feedback/dashboard_feedback.json`

Purpose:

- persistent dashboard feedback store
- stores:
  - label action
  - story metadata
  - rank diagnostics such as:
    - `hybrid_score`
    - `semantic_score`
    - `knn_score`
    - `max_sim_score`
    - `max_cluster_score`
    - `cross_encoder_score`

This store is the source for learned-ranker training labels.

## `--cache-only`

In `evaluate_quality.py`, `--cache-only` means:

- use cached user data
- use cached stories/candidates where available
- disable RSS fetch
- ignore TTL freshness constraints for those cached artifacts

It is useful for repeatable local comparisons, but it is not identical to live
production fetch behavior.

## Practical Debug Rule

If a result looks wrong, ask which cache layer might be serving stale data:

1. user-signal cache
2. candidate cache
3. story cache
4. CE score cache
5. dupe cache
6. feedback store

Do not assume a single "the cache" in this repo. There are several distinct
cache layers with different invalidation rules.
