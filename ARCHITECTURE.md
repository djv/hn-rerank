# HN Rerank Architecture

## Overview
HN Rerank is a local-first application that personalizes Hacker News content using semantic embedding models. It learns from your upvoted/favorited stories to rerank the front page (or recent stories) according to your specific interests.

## Core Components

### 1. CLI Entrypoint (`main.py` / `generate_html.py`)
- **Orchestrator**: Manages the flow of fetching user data, fetching candidates, embedding content, reranking, and generating the dashboard.
- **UI Generation**: Produces a self-contained HTML dashboard with a modern dark-mode UI.
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
- **Scoring Algorithm**:
    - **MaxSim**: Finds the single best match between a candidate and your history.
    - **Density**: Rewards candidates that match *clusters* of your interest (mean of top-3 matches).
    - **Hybrid Score (Ranking)**: `(MaxSim + Density) / 2`. This is used to sort the stories.
    - **Display Score (UI)**: Uses the raw `MaxSim` percentage. This provides more intuitive feedback when a story is linked to a specific "Reason" story in your history.
    - **Weighting**: Heavily favors Semantic Score (95%) over HN Popularity (5%).
- **Diversity**: Applies Maximal Marginal Relevance (MMR) to prevent showing 30 identical stories about the same topic.

### 5. Constants (`api/constants.py`)
- Centralized configuration for cache TTLs, scoring weights, and limits.

## Data Flow
1.  **Login/Profile**: Fetch user's `upvoted` and `hidden` IDs.
2.  **Signal Fetching**: Scrape content for these IDs. Cache locally.
3.  **Embedding (Signals)**: Generate vectors for user's history.
4.  **Candidate Discovery**: Fetch top N stories from Algolia (last 30 days).
5.  **Candidate Fetching**: Scrape content for candidates.
6.  **Embedding (Candidates)**: Generate vectors for candidates.
7.  **Ranking**: Compute similarity matrix. Apply penalties. Sort.
8.  **Render**: Generate `index.html`.

## Key Invariants
- **Local-First**: No data is sent to third-party AI APIs. All inference is local (ONNX).
- **Privacy**: Cookies and data live in `.cache/` inside the project.
- **Robustness**: Negative caching prevents API hammering on invalid IDs.
