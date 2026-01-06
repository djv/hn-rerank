# HN Rerank

A semantic search engine for Hacker News that reranks the front page based on your personal interests.

## Architecture
*   **Engine**: `BAAI/bge-small-en-v1.5` (State-of-the-art sentence embeddings).
*   **Logic**: Calculates the "Centroid" of your interests and ranks HN stories by cosine similarity.
*   **UI**: Textual (TUI).

## Usage

### TUI (Recommended)
Interactive terminal UI.

```bash
uv run tui_app.py
```

*   **Left Pane**: Add your interests (e.g., "Rust", "SpaceX", "Databases").
*   **Right Pane**: Automatically updates with the most relevant HN stories.
*   **Click**: Opens the story in your default browser.
