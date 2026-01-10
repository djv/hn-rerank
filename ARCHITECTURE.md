# HN Rerank Architecture

## Overview
HN Rerank is a local-first application that personalizes Hacker News content using semantic embedding models. It learns from your upvoted/favorited stories to rerank the front page (or recent stories) according to your specific interests.

## Core Components

### 1. CLI Entrypoint (`generate_html.py`)
- **Orchestrator**: Manages the flow of fetching user data, embedding, clustering, fetching candidates, reranking, and generating the dashboard.
- **UI Generation**: Produces two self-contained HTML files:
    - `index.html` - Ranked recommendations with per-story cluster chips and LLM-generated TL;DRs.
    - `clusters.html` - Full interest cluster visualization with stories per cluster.
- **Concurrency**: Uses `asyncio` for parallel fetching of stories.

### 2. Client API (`api/client.py`)
- **HN Client**: Handles authentication and user profile scraping.
- **Session Management**: Stores login cookies in `.cache/user/cookies.json`.
- **Signal Fetching**: Retrieves IDs of favorited, upvoted, and hidden stories.
- **Security Note**: Cookies are stored in plain text. This is intended for local single-user use only.

### 3. Content Fetching (`api/fetching.py`)
- **Hybrid Approach**:
    - **Discovery**: Uses Algolia API to find candidate stories (search by date/points).
    - **Detail**: Uses HTML Scraping (`BeautifulSoup`) against `news.ycombinator.com` to fetch story details and *ranked comments*.
- **Smart Scraping**:
    - **Breadth-First Selection**: Prioritizes root comments to capture diverse viewpoints.
    - **Weighted Sort**: Balances "Page Rank" with "Indent Depth" to avoid deep rabbit holes.
    - **Cleaning**: Filters out low-quality comments (ASCII art, short replies).
- **Caching**:
    - **Positive Cache**: Stores valid story data for 24h.
    - **Negative Cache**: Stores failures/invalid items (e.g., jobs, comments) to prevent infinite re-fetching loops.

### 4. Reranking Engine (`api/rerank.py`)
- **Model**: Uses a local ONNX embedding model (`bge-base-en-v1.5`).
- **Multi-Interest Clustering**:
    - Uses Agglomerative Clustering (Ward linkage) to discover interest groups.
    - Silhouette score threshold (≥0.1) ensures coherent clusters.
    - Searches from high k to low to maximize granularity.
    - `cluster_interests_with_labels(embeddings, weights)` returns `(centroids, labels)`.
- **Cluster Naming** (`generate_single_cluster_name()` via local LLM):
    - Uses ollama with `llama3.2:3b` to generate contextual 1-3 word labels.
    - Strips HN prefixes (Show HN:, Ask HN:, Tell HN:) before sending to LLM.
    - Falls back to "Misc" if ollama unavailable.
    - Progress bar shows per-cluster naming progress.
- **Scoring Algorithm**:
    - **Cluster MaxSim**: Best match to any interest cluster centroid (70% weight).
    - **Cluster MeanSim**: Broad appeal across all clusters (30% weight).
    - **Display Score**: Raw MaxSim to individual stories for interpretable "reason" links.
    - **Weighting**: Semantic (95%) + HN Popularity (5%).
- **Diversity**: Applies Maximal Marginal Relevance (MMR, λ=0.35) to prevent redundant results.
- **Story TL;DR** (`generate_story_tldr()` via local LLM):
    - Generates concise summaries (up to 800 chars) using `llama3.2:3b`.
    - Format: Story summary, followed by a newline and key discussion points/debates.
    - Replaces the raw comments section in the story card for a cleaner UI.
    - Cached by story ID in `.cache/tldrs.json`.

### 5. Constants (`api/constants.py`)
- Centralized configuration for cache TTLs, scoring weights, clustering parameters, and limits.

## Data Flow

```
1. Login/Profile     → Fetch user's upvoted and hidden IDs
2. Signal Fetching   → Scrape content for these IDs, cache locally
3. Embedding         → Generate vectors for user's history (BGE-base)
4. Clustering        → Group signals into clusters (Agglomerative + Ward)
5. Cluster Naming    → Generate names via local LLM (ollama)
6. Candidate Fetch   → Get top N stories from Algolia (last 30 days)
7. Candidate Embed   → Generate vectors for candidates
8. Cluster Assign    → Map each candidate to best-matching cluster centroid
9. Ranking           → Compute similarity to centroids, apply MMR diversity
10. TL;DR Generation → Generate 1-sentence summaries via local LLM
11. Render           → Generate index.html + clusters.html
```

## HTML Output Structure

### index.html (Ranked Recommendations)
```
┌─────────────────────────────────────┐
│ HN Rerank | @username               │
│ N interest clusters (link)          │
├─────────────────────────────────────┤
│ Story Card                          │
│ ┌─────────────────────────────────┐ │
│ │ 85% [ML, AI] 142pts 2h          │ │
│ │ Story Title                     │ │
│ │ ↳ Similar to: Your Past Story  │ │
│ │ "Top comment snippet..."        │ │
│ └─────────────────────────────────┘ │
│ ...more stories...                  │
└─────────────────────────────────────┘
```

Each story card shows:
- Match percentage (semantic similarity)
- Cluster chip (which interest cluster it matches)
- HN points and age
- "Similar to" link showing which upvoted story it matches
- Top comment snippets

### clusters.html (Interest Clusters)
```
┌─────────────────────────────────────┐
│ Interest Clusters | @username       │
│ N signals → M clusters              │
├─────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐             │
│ │ ML, AI  │ │ Rust    │             │
│ │ 12 items│ │ 8 items │             │
│ │ Story 1 │ │ Story 1 │             │
│ │ Story 2 │ │ Story 2 │             │
│ └─────────┘ └─────────┘             │
└─────────────────────────────────────┘
```

## Key Invariants
- **Local-First**: No data is sent to third-party AI APIs. All inference is local (ONNX).
- **Privacy**: Cookies and data live in `.cache/` inside the project.
- **Robustness**: Negative caching prevents API hammering on invalid IDs.
- **Determinism**: Same input → same clustering output (fixed random state).

## Testing
- **Unit Tests**: `pytest` for core logic
- **Property Tests**: `hypothesis` for invariants (clustering, ranking, boundaries)
- **Coverage**: ~76% of `api/` module
