# Plan: Write-Time Composition and Read-Path Optimization

This plan refactors the `stories.text_content` cache to move composition to the write-path, optimizing database read performance and simplifying state synchronization.

## 1. Database Schema & Dataclass Changes (`database.py`)

- **Dataclass**: Remove `db_text_content` field from `Story`.
- **`_row_to_story`**: Remove the runtime `compose_story_text` call entirely. Populate `text_content` directly from the database column `row[5]`.
- **`get_all_feedback_stories`**: Remove runtime `compose_story_text` call, loading `text_content` directly from the column.

## 2. Invalidation & Write-Path Updates

When updating story parts (e.g., `article_body`), we must update the `text_content` column as well and overwrite/delete the cached embedding:
- **`pipeline.py` (Proactive Scraper)**: Ensure the in-memory `updated` object uses composed text for both `text_content` and database storage.
- **`server.py` (TLDR Handler)**: Recompose `text_content` when saving newly scraped article bodies to the database.
- **Remove `_enrich_article_body`**: Since all updates immediately refresh the composed `text_content` and database rows, the background loop `_enrich_article_body` and the `db_text_content` attribute become completely obsolete.

## 3. Execution Verification

1. **Schema Check**: Verify that `db_text_content` removal does not break standard database queries and that `text_content` is correctly populated on all read paths.
2. **Pytest Run**: Run the suite to verify that constructors and query assertions pass.
