# OpenCode Instructions for `hn-rewrite`

## Project shape

- This repository is a minimalist, local-first Hacker News reranking dashboard rewrite.
- Use `uv run python <script>.py` or standard `uv` commands to execute scripts and run tests.
- Treat `public/` as generated output unless a task explicitly targets it.

## Working rules

- Make minimal, behavior-preserving changes unless the user asks for a broader refactor.
- Keep the runtime path local-first; do not add new external dependencies unless needed.
- **Be very skeptical of unusually high metrics** (e.g. NDCG > 0.40). We are unlikely to beat the Hacker News baseline by a large margin; high metrics often indicate feature leakage, train-test contamination, or metric saturation artifacts.
- **Do NOT standard-scale raw embeddings**: StandardScaler must only be applied to metadata columns (indices `emb_dim:` onward), leaving the 384-dimensional unit-normed raw embeddings untouched. Scaling raw embedding dimensions independently distorts their semantic cosine similarity structure and collapses ranking performance.
- **Never delete or destructively modify the local database** (`hn_rewrite.db`, `hn.db`, or any `*.db` file in the working tree). The DB holds the user's accumulated feedback and is the single source of truth for personalization. No `rm`, no `DELETE FROM` without a `WHERE` clause that excludes all rows, no schema migrations that drop tables or columns with data. The pipeline's own `prune_stories` and `prune_*` operations are fine — they have explicit retention rules and `id NOT IN (SELECT story_id FROM feedback)` guards. When in doubt, ask before running any command that touches the DB file.
- Keep test execution times optimized (target under 10 seconds total).
- **Always update relevant documentation** (e.g., [ARCHITECTURE.md](file:///home/dev/hn-rewrite/ARCHITECTURE.md)) after making code or behavior changes.

## Common commands

- Install or refresh the environment: `uv sync`
- Run tests: `uv run pytest tests/`
- Run linting: `uv run ruff check .`
- Run persistent server: `systemctl --user {status|start|stop|restart} hn_rewrite.service` (or directly: `uv run python server.py`)
- Run one-shot generation: `uv run python generate.py`
- Migrate feedback from legacy JSON: `uv run python migrate_feedback.py`

## Backup

The HN database is backed up daily to Google Drive via a systemd user timer.

- Script: `scripts/backup_hn_db.sh`
- Service: `~/.config/systemd/user/hn-rewrite-backup.service`
- Timer: `~/.config/systemd/user/hn-rewrite-backup.timer` (active)
- Target: `drive:hn-rewrite/backups/<YYYYMMDDTHHMMSSZ>/hn_rewrite.db`
- Retention: 30 most recent snapshots (env: `HN_KEEP_N=30`)
- Logs: `journalctl --user -u hn-rewrite-backup.service`

### Manual backup

```bash
./scripts/backup_hn_db.sh                          # default config
HN_DB_PATH=/path/to/other.db ./scripts/backup_hn_db.sh
HN_KEEP_N=7 ./scripts/backup_hn_db.sh             # keep 7
```

### Restore

```bash
LATEST=$(rclone lsf --dirs-only drive:hn-rewrite/backups/ | sort -r | head -1)
rclone copy drive:hn-rewrite/backups/$LATEST/hn_rewrite.db ./hn_rewrite.db
sqlite3 hn_rewrite.db "PRAGMA integrity_check;"

## Comment Backfill

- `fetch_story` short-circuits on cached stories with stale/missing `top_comments`. Fixed: `comments_stale` check falls through to Algolia items API. Comment re-fetch capped at 100 per pipeline run (`fetch_stories_by_id` sorts stale IDs descending, takes top 100).
- Error paths in `fetch_story` (non-200, invalid item, exception) preserve existing cached rows — `if story is None` guard prevents `_empty_story(sid)` from overwriting real data on transient failures.
- Corrupted stories (`title=""` but `text_content` preserved from COALESCE) are detected and given priority in the re-fetch queue, ahead of the 100-slot stale-comment cap.
- `_empty_story` is destructive: zeros `title`, `time`, `score`, `self_text`, `top_comments`, `text_content`. Only `article_body` survives (COALESCE in `upsert_story`). A single transient 403 on the Algolia items API can permanently damage cached stories if error paths call `_empty_story`.
- `_row_to_story` recomposes `text_content` live from raw parts on every read. This hides corruption — zeroed `title`/`time` with preserved `article_body` produces non-empty `text_content`, so the story passes filtering but shows blank title and epoch timestamp ("20624d ago").
- `upsert_story` COALESCE only covers `article_body`. All other columns (title, score, time, self_text, top_comments, text_content) are overwritten unconditionally. This is an architectural vulnerability.
- 1,940 stories still need comment backfill. They'll be gradually re-fetched as they appear in future Algolia search windows (100 per pipeline run).
```
