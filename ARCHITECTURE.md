# HN Rerank Architecture

## Overview

HN Rerank is a local-first recommendation pipeline for Hacker News. It combines cached HN signals, locally generated embeddings, candidate discovery from Algolia and RSS, and static HTML rendering.

The system is local for ranking and artifact generation, but not fully offline:
- HN and Algolia provide source data.
- RSS feeds and article fetches expand the candidate pool.
- Groq is optional and only used for cluster naming and TL;DR generation.
- The generated HTML currently pulls Tailwind from a CDN at render time.

## Main Entry Point

### `generate_html.py`

Responsibilities:
- parse CLI arguments and optional `hn_rerank.toml`
- load user signals through `HNClient`
- fetch signal details and candidate stories
- embed, cluster, rank, and select final stories
- optionally generate cluster names and TL;DRs
- render `index.html` and `clusters.html`

Notable runtime behavior:
- top-level config is loaded from `[hn_rerank]`
- `HN_RERANK_FORCE_NO_TLDR=1` hard-disables TL;DR generation for automated runs
- the displayed match badge uses `knn_score`
- final selection targets a best-effort `2:1` HN:RSS mix

## User Signals

### `api/client.py`

`HNClient` handles:
- cookie-backed HN login
- scraping favorites, upvotes, and hidden items
- URL normalization for duplicate and hidden-item suppression
- short-lived caching of signal IDs in `.cache/user`

Signal semantics:
- positive set: `(favorites | upvoted) - hidden`
- hidden items are always excluded from candidate selection
- without login, the pipeline falls back to public favorites only

## Candidate Discovery

### `api/fetching.py`

Candidate discovery uses Algolia plus optional RSS expansion.

Algolia path:
- queries are split into a live window plus archive windows to work around the 1000-hit cap
- live windows are cached with short TTLs, older windows with longer TTLs
- story detail fetches use Algolia item endpoints, not HN HTML scraping
- non-story items and stories with too few usable comments are cached negatively as `None`

Story content path:
- comment text is cleaned, filtered, and depth-penalized
- title plus top-ranked comments become `text_content` for embedding and ranking
- UI comments are a shorter subset of the ranking comment pool

RSS path:
- parses RSS/Atom entries into synthetic negative IDs
- optionally fetches full article text for embedding enrichment
- excludes URLs already seen in favorites, upvotes, or hidden items

## Embeddings and Clustering

### `api/rerank.py`

Embedding model:
- local ONNX model under `onnx_model/`
- tokenizer access is serialized for thread safety
- embedding results are cached in `.cache/embeddings` and `.cache/embeddings_cluster`

Clustering path:
- default algorithm is `spectral`
- alternatives exist for `agglomerative` and `kmeans`
- cluster count is capped by sample count, `MAX_CLUSTERS`, and `MIN_SAMPLES_PER_CLUSTER`
- post-processing refines assignments, merges small clusters, splits oversized clusters, and can split low-similarity outliers into singleton clusters

The clustering configuration is controlled by constants and nested TOML sections such as `[hn_rerank.clustering]`.

## Ranking

### Ranking modes

The ranking engine supports two modes:
- classifier mode when both positive and negative signal sets have at least 5 embeddings
- k-NN fallback otherwise

Classifier mode:
- trains a `LogisticRegressionCV` model per run
- augments embedding features with centroid similarity, positive k-NN, and negative k-NN features
- applies cluster-balanced positive sample weights and configurable negative sample weights
- still computes k-NN display scores and explicit negative penalties after classification

Fallback mode:
- computes median top-k similarity against positive history
- blends cluster-max similarity and k-NN behavior for semantic scoring
- uses the best positive match for display reasoning

Final ranking behavior:
- hybrid score blends semantic relevance, adaptive HN weighting, and freshness
- MMR-style diversification suppresses redundant candidates
- the UI match badge uses `knn_score`, while ordering uses `hybrid_score`

## LLM Enrichment

Optional Groq-backed stages:
- Cluster naming via `generate_batch_cluster_names()`:
    - Uses "Rich Context": passes all cluster titles, plus the first 250 characters of `text_content` or top comments for the top 3 stories.
    - Ensures "Global Uniqueness": maintains a list of `already_used_names` to prevent duplicate labels within a single report.
    - Applies a strict coverage rule: "The cluster name MUST cover all stories in the cluster."
    - Automatically standardizes technical acronyms (e.g., "AI", "LLM") through post-processing.
- Story TL;DR generation via `generate_batch_tldrs()`:
    - Batches 5 stories per request to minimize API calls and latency.
    - Focuses on technical insights and trade-offs rather than summary-style descriptions.

Operational rules:
- `GROQ_API_KEY` is required only when naming or TL;DRs are enabled.
- Singleton clusters are left unnamed unless generic fallback naming is used.
- Cluster naming and TL;DR results are cached under `.cache/` with versioning (e.g., `v9` for naming) to force refreshes when logic changes.

## Rendered Artifacts

### `index.html`
Contains:
- ranked story cards
- match badge from `knn_score`
- RSS badge for RSS items
- optional cluster chip
- HN points and relative age
- external story link and optional HN discussion link
- optional TL;DR block

### `clusters.html`
Contains:
- all positive signals grouped by cluster label
- cluster sizes
- recent stories per cluster

Both pages are generated from inline Jinja templates in `generate_html.py`.

## Configuration Model

Current configuration is split across two layers:
- `generate_html.py` parses top-level runtime options from `[hn_rerank]`
- `api/constants.py` loads nested tuned values from `[hn_rerank.<section>]`

This split works, but it is an architectural debt item because execution does not yet flow through a single typed config object.

## Testing Surface

The strongest automated coverage is around:
- ranking invariants
- clustering behavior
- classifier fallbacks and weighting
- HTML selection and rendering helpers
- promotion and evaluation utilities

The weakest coverage is around:
- true end-to-end runtime integration
- config precedence and parsing behavior
- doc and test contract drift
