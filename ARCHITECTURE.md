# HN Rerank Architecture

## Overview
HN Rerank is a local-first application that personalizes Hacker News content using semantic embedding models. It learns from your upvoted/favorited stories to rerank the front page (or recent stories) according to your specific interests.

## Core Components

### 1. CLI Entrypoint (`generate_html.py`)
- **Orchestrator**: Manages the flow of fetching user data, embedding, clustering, fetching candidates, reranking, and generating the dashboard.
- **UI Generation**: Produces two self-contained HTML files:
    - `index.html` - Ranked recommendations with per-story cluster chips and LLM-generated TL;DRs.
    - `clusters.html` - Full interest cluster visualization with stories per cluster.
- **Concurrency**: Uses `asyncio` for parallel fetching of stories and non-blocking rate-limited LLM calls.

### 2. Client API (`api/client.py`)
- **HN Client**: Handles authentication and user profile scraping.
- **Session Management**: Stores login cookies in `.cache/user/cookies.json`.
- **Signal Fetching**: Retrieves IDs of favorited, upvoted, and hidden stories.
- **Security Note**: Cookies are stored in plain text. This is intended for local single-user use only.

### 3. Content Fetching (`api/fetching.py`)
- **Hybrid Approach**:
    - **Discovery**: Uses Algolia API to find candidate stories (search by date/points).
    - **Time Windows**: Fetches candidates in **7-day windows** to stay under Algolia's 1000-hit limit while minimizing API calls.
    - **Detail**: Uses HTML Scraping (`BeautifulSoup`) against `news.ycombinator.com` to fetch story details and *ranked comments*.
- **Caching**:
    - **Positive Cache**: Stores valid story data for 24h.
    - **Negative Cache**: Stores failures/invalid items (e.g., jobs, comments) to prevent infinite re-fetching loops.
    - **Candidate Cache**: Stores Algolia search results per time-window. Recent window (30m TTL), older windows (1w TTL).
- **Smart Scraping**:

### 4. Reranking Engine (`api/rerank.py`)
- **Model**: Uses a local ONNX embedding model (`bge-base-en-v1.5`).
- **Multi-Interest Clustering**:
    - Uses **Agglomerative Clustering** with **Average Linkage** and **Cosine Metric** (better for irregular semantic shapes than Ward/Euclidean).
    - Silhouette score threshold (≥0.14) ensures coherent clusters while allowing semantic merges.
    - Searches from high k to low to maximize granularity.
    - `cluster_interests_with_labels(embeddings, weights)` returns `(centroids, labels)`.
- **Cluster Naming** (`generate_batch_cluster_names()` via Groq API):
    - Uses Google Groq API (`llama-3.3-70b-versatile`) to generate contextual 1-3 word labels.
    - Batches naming requests (10 per call) to optimize quota.
    - Strips HN prefixes (Show HN:, Ask HN:, Tell HN:) before sending to LLM.
    - Falls back to "Misc" if API unavailable.
    - Cached by cluster content hash in `.cache/cluster_names.json`.
    - Progress bar shows per-cluster naming progress.
- **Scoring Algorithm**:
    - **k-NN Scoring**: Calculates the mean similarity of the top 3 nearest neighbors from your history. This handles irregular interest shapes better than centroids.
    - **k-NN Negative Signals**: Hidden stories also use k-NN (top-3 avg). Penalty applied when negative k-NN > positive k-NN (contrastive). Weight: 0.5.
    - **Soft Sigmoid Activation**: Applies a sigmoid (k=15, threshold=0.35) to semantic scores to suppress noise while preserving strong signals.
    - **Display Score**: k-NN score (mean of top-3 neighbors) for consistent ranking/display alignment.
    - **Weighting**: Semantic (95%) + HN Popularity (5%).
- **Diversity**: Applies Maximal Marginal Relevance (MMR, λ=0.35) to prevent redundant results.
- **Story TL;DR** (`generate_batch_tldrs()` via Groq API):
    - Generates concise summaries using `llama-3.1-8b-instant`.
    - Batches requests (5 per call) to minimize API quota consumption.
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
4. Clustering        → Group signals into clusters (Agglomerative + Average Linkage)
5. Cluster Naming    → Generate names via Groq API
6. Candidate Fetch   → Get top N stories from Algolia (last 30 days, 7-day windows)
7. Candidate Embed   → Generate vectors for candidates
8. Reranking         → Compute similarity to centroids, apply MMR diversity
9. UI Clustering     → Candidate chips show the User Interest Cluster that triggered the match
10. TL;DR Generation → Generate 1-sentence summaries via Groq API
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
- **Hybrid Privacy**: Embeddings and reranking are local (ONNX). LLM features (naming, summaries) use Groq API.
- **Privacy**: Cookies and data live in `.cache/` inside the project.
- **Robustness**: Negative caching prevents API hammering on invalid IDs.
- **Determinism**: Same input → same clustering output (fixed random state).

## Testing
- **Unit Tests**: `pytest` for core logic
- **Property Tests**: `hypothesis` for invariants (clustering, ranking, boundaries)
- **Coverage**: ~76% of `api/` module
