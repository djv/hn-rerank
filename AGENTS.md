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
- Run tests: `uv run pytest`
- Run linting: `uv run ruff check .`
- Generate the dashboard: `uv run python generate_html.py <hn-username>`

## Repo notes

- `hn_rerank.toml` controls most runtime behavior.
- `README.md` is the canonical user-facing setup reference.
- `public/index.html` and `public/clusters.html` are generated artifacts.
