# Architecture & Design: hn-rewrite

This document outlines the architecture, core design decisions, database schema, ranking system, and maintenance instructions for the `hn-rewrite` minimalist local-first Hacker News reranking dashboard.

---

## 1. System Overview

`hn-rewrite` is a unified, resource-efficient rewrite of the original reranking system. It functions as a local-first web application that fetches stories from Hacker News and multiple RSS feeds, semantic-ranks them using a locally run sentence-embedding model and SVM, and presents them in a clean web dashboard.

```mermaid
graph TD
    subgraph Core Pipeline
        A[fetch_candidates] --> B[get_or_compute_embeddings]
        B --> C[rank_stories]
        C --> D[mmr_filter]
        D --> E[generate_dashboard]
    end

    subgraph Service Layer
        F[server.py] -->|1. GET /| E
        F -->|2. POST /api/feedback| G[(Database)]
        F -->|3. POST /api/tldr-detail| H[LLM Provider]
        F -->|4. Thread| CorePipeline[run_pipeline]
    end
```

---

## 2. Component Layout

The codebase consists of five primary modules:

1. **[database.py](file:///home/dev/hn-rewrite/database.py)**: Encapsulates all SQLite interactions. Manages schemas (`stories`, `embeddings`, `feedback`, `article_cache`), cascade-deletes, pruned retention rules, and automatic schema migrations. The `article_cache` table stores fetched article bodies for LLM enrichment, keyed by `story_id` with a 7-day TTL.
2. **[pipeline.py](file:///home/dev/hn-rewrite/pipeline.py)**: Orchestrates the background update sequence. Integrates RSS parsed feeds, computes text embeddings using ONNX, fits the SVM, and generates the final dashboard.
3. **[server.py](file:///home/dev/hn-rewrite/server.py)**: A multi-threaded web server serving the static dashboard, handling feedback writes, proxying detailed TLDR summaries to LLM APIs, and housing the background regeneration event thread.
4. **[templates/index.html](file:///home/dev/hn-rewrite/templates/index.html)**: Jinja2 dashboard template styled with a compact dark-theme Pico CSS layout. Includes client-side sorting, autohide transitions, and asynchronous detailed analysis rendering.
5. **[migrate_feedback.py](file:///home/dev/hn-rewrite/migrate_feedback.py)**: Imports legacy feedback data from `hn_rerank` JSON files, backfilling candidate story contents and caching embeddings.

---

## 3. Key Design Decisions

### 3.1 Normalized Schema & Data Integrity
To eliminate data redundancy, the feedback schema is strictly normalized. Metadata (`title`, `url`, `text_content`, `source`) is not duplicated in the `feedback` table. Instead, a foreign key references `stories(id)`. 
To prevent constraint violations or data loss during cleanup:
* `prune_stories` leaves feedback-associated stories intact (`id NOT IN (SELECT story_id FROM feedback)`).
* `get_all_feedback` and `get_feedback_for_training` perform a `LEFT JOIN` against `stories` to resolve attributes dynamically.

### 3.2 392-Dimensional Feature Space
Rather than mixing semantic matches with engagement counts using arbitrary manual weights, we feed them directly into the Support Vector Machine (SVM). The model trains on a **392-dimensional feature vector**:
* **`[0-383]` (384-d)**: MiniLM sentence embedding of `text_content`.
* **`[384]` (1-d)**: Normalized log points: `min(log1p(score), 8.0) / 8.0`.
* **`[385]` (1-d)**: Normalized log comment count: `min(log1p(comments), 7.0) / 7.0`.
* **`[386]` (1-d)**: Normalized log text length: `min(log1p(len), 12.0) / 12.0`.
* **`[387]` (1-d)**: Normalized engagement quality (points per hour since submission): `min(log1p(quality), 8.0) / 8.0`.
  * **Quality formula**: `score / (hours_since_submission + 1)`. Raw standalone age is not directly appended, but is utilized here.
* **`[388-391]` (4-d)**: Normalized similarity metrics to historical feedback:
  * Mean cosine similarity to upvoted story embeddings.
  * Mean cosine similarity to downvoted story embeddings.
  * Maximum cosine similarity to any upvoted story embedding.
  * Maximum cosine similarity to any downvoted story embedding.

### 3.3 Engagement-Aware MMR Filtering
Standard MMR (Maximal Marginal Relevance) strictly penalizes topic duplication based on similarity. If two stories are similar, the lower-ranked one is discarded. We modified `mmr_filter` to identify similarity groups. If an alternative candidate has significantly higher engagement than the group leader:
```python
other_engagement > leader_engagement * 2.0 + 30
```
the higher-engagement candidate is promoted as the representative for the cluster. The final set is sorted back to match original SVM relative rank order.

After MMR, three surfacing passes append stories beyond the `limit` following the same pattern (boolean flag on `RankedStory`, threshold during `rank_stories`, surfacing loop in `run_pipeline`): **Novel** (top 15% least similar to feedback, SVM score > 0.5, `✨ Novel`, 5 slots, sorted by SVM score), **Discussion-rich** (top 10% by `comment_count`, badge `💬 Talk-worthy`, 5 slots, sorted by comment_count descending), and **High-engagement** (top 10% by `story.score`, badge `🔥 Trending`, 5 slots, sorted by score descending). Each deduplicates against previously surfaced IDs before appending.

### 3.4 Client-side Autohide
When a user upvotes/downvotes a card, the UI writes the current card height inline, triggers a CSS collapse transition (`max-height: 0 !important; opacity: 0;`), and removes the card from the DOM after 400ms. The background thread updates the actual static page asynchronously.

### 3.5 Algolia Candidate Fetch Window
The live-window fetch (`pipeline.py:336`) queries the Algolia HN search API in 7 daily chunks. Each day's fetch collects up to **350 hits** (5 pages of 100, minus stories with `points <= 5`). This cap was raised from 150 to capture the majority of high-score stories on busy days; previously, stories on high-volume days could be dropped before the reranker evaluated them.

---

## 4. LLM Detailed Analysis

### 4.1 Article Body Enrichment

The `/api/tldr-detail` endpoint enriches the LLM prompt with the full article body when the story's HN-provided text is thin (<500 chars) and a URL is available.

Fetch flow (server.py `_fetch_article_body`):
1. **Cache lookup**: checks `article_cache` table (keyed by `story_id`, invalidated if URL changes).
2. **Fetch** (if cache miss): HTTP GET with Chrome 131 browser-grade headers. Single retry on 429/503 after 1s sleep.
3. **Extraction chain**: `trafilatura.extract()` first (robust against 100+ site templates); falls back to `BeautifulSoup` (strips non-content tags, prefers `<article>`/`<main>` containers).
4. **Cache write**: successful extractions are stored in `article_cache` with a 7-day TTL. Empty/failed fetches are never cached; retried on next request.

### 4.2 Prompt Construction

The detailed summary endpoint `/api/tldr-detail` proxies requests to Mistral or Groq. It compiles the story title, engagement context (points, comments, age), article body (when available), and up to **30,000 characters** of HN discussion text (~45K effective prompt size with surrounding system/user scaffolding). The prompt includes hedging rules: if engagement is low (<20 points or <5 comments), the LLM uses cautious language like "the article describes...". If the article body was unavailable, it notes that explicitly.

### 4.3 Client-side Rendering

The raw Markdown response is formatted on the fly using a robust, line-by-line parser (`parseSimpleMarkdown`) to render headers, bold text, and lists safely.

---

## 5. Maintenance Guide

### 5.1 Service Control
The server runs as a systemd user service.
```bash
# Manage the service
systemctl --user {status|start|stop|restart} hn_rewrite.service

# View active logs
journalctl --user -u hn_rewrite.service -f -n 100
```

### 5.2 Verification Suite
Ruff and Pytest are configured for standard validation.
```bash
# Run all unit tests
uv run pytest tests/

# Check styling and types
uv run ruff check .
```
