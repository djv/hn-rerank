# OpenCode Instructions for `hn_rerank`

## Project shape

- This repository builds a local-first Hacker News reranking dashboard.
- Prefer `uv run ...` for Python entrypoints and tests.
- Treat `public/` as generated output unless a task explicitly targets it.

## Working rules

- Make minimal, behavior-preserving changes unless the user asks for a broader refactor.
- Keep the runtime path local-first; do not add new external dependencies unless needed.
- If you change ranking, candidate selection, or dashboard output, update the relevant docs and tests together.
- Respect existing uncommitted work in the tree.

## Common commands

- Install or refresh the environment: `uv sync`
- Run tests (parallel): `uv run pytest -q -n auto`
- Run tests (fast, skip slow): `uv run pytest -q -m "not slow"`
- Run linting: `uv run ruff check .`
- Run type checker: `uv run ty check api/ tests/`
- Generate the dashboard: `uv run python generate_html.py <hn-username>`

## Repo notes

- `hn_rerank.toml` controls most runtime behavior.
- `README.md` is the canonical user-facing setup reference.
- `public/index.html` and `public/clusters.html` are generated artifacts.
- Tests marked `@pytest.mark.slow` (6 tests: integration, pagination, SVM, clustering stability) are skipped with `-m "not slow"`.

## Long-running services

All persistent services run as **systemd user services**. Never start long-running Python processes with `&` or `nohup` — use systemd.

| Service | File | Manage |
|---------|------|--------|
| Feedback API | `~/.config/systemd/user/hn_rerank_feedback.service` | `systemctl --user {start\|stop\|restart\|status} hn_rerank_feedback` |
| Dashboard regen timer | `~/.config/systemd/user/hn_rerank.timer` + `hn_rerank.service` | `systemctl --user {start\|stop} hn_rerank.timer` |
| Caddy reverse proxy | `/etc/systemd/system/hn-dashboard.service` | `sudo systemctl {start\|stop\|restart\|status} hn-dashboard` |

The feedback server auto-triggers dashboard regeneration on upvote/neutral/down via `api/regen_scheduler.py`. Regen logs to `.cache/regen.log`.
