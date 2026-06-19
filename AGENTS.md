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
- Run persistent server: `uv run python server.py`
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
```
