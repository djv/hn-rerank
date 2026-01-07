# HN Rerank

> A personalized, privacy-first Hacker News dashboard that learns from your upvotes.

HN Rerank fetches stories from the last 30 days and re-ranks them based on semantic similarity to your Hacker News history (upvoted/favorited stories). It runs entirely locally using ONNX models—no data ever leaves your machine.

## Features

- **Personalized Ranking**: Uses local embeddings to find stories matching your specific interests.
- **Hybrid Scoring**: Rewards both niche hits (MaxSim) and established interest clusters (Density).
- **Privacy-First**: Your credentials and history stay in `.cache/`. No third-party AI APIs.
- **Smart Summaries**: Weighted breadth-first comment selection to capture diverse viewpoints without deep-thread noise.
- **Modern UI**: Generates a responsive, dark-mode `index.html` dashboard.

## Quick Start

1. **Install Dependencies** (requires `uv`):
   ```bash
   uv sync
   ```

2. **Setup Model**:
   ```bash
   uv run setup_model.py
   ```

3. **Run**:
   ```bash
   uv run main.py
   ```

## Documentation

- [**ARCHITECTURE.md**](ARCHITECTURE.md): Deep dive into the scoring algorithms, scraping logic, and data flow.
- [**TECHNICAL_DEBT.md**](TECHNICAL_DEBT.md): Security audit, performance notes, and future roadmap.
