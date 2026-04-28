# HN Rerank

> A local-first Hacker News dashboard that ranks stories against your own history.

HN Rerank builds a static dashboard from your HN signals, recent Algolia stories, and optional RSS feeds. Embeddings and reranking run locally with ONNX. Optional enrichments such as cluster names and TL;DRs use Groq when enabled.

Requires Python 3.12+.

## What It Does

- Builds a preference profile from favorites, upvotes, and hidden stories.
- Fetches recent candidate stories from Algolia in cache-friendly time windows.
- Optionally expands the pool with RSS/Atom feeds and full-article extraction.
- Clusters your positive signals into interest groups.
- Reranks candidates with local embeddings plus optional classifier mode.
- Renders `index.html` and `clusters.html` as static artifacts.

## Runtime Characteristics

- Local:
  embeddings, clustering, reranking, caching, and HTML generation.
- External services:
  Hacker News, Algolia, RSS feeds, and optionally Groq for cluster names and TL;DRs.
- Built-in extra feeds:
  `jack-clark.net`, `lobste.rs`, and `tildes.net` are appended to the OPML feed set.
- Browser dependency:
  the generated HTML currently loads Tailwind from `https://cdn.tailwindcss.com` at render time.

## Quick Start

1. Install dependencies.
   ```bash
   uv sync
   ```
2. Download/export the ONNX model if `onnx_model/model.onnx` is not already present.
   ```bash
   uv sync --extra model-export
   uv run setup_model.py
   ```
3. Configure `GROQ_API_KEY` only if you want cluster naming and/or TL;DR generation.
   ```bash
   export GROQ_API_KEY="your-api-key"
   ```
4. Generate the dashboard.
   ```bash
   uv run generate_html.py <your-hn-username> --days 7 --clusters 30
   ```

Outputs:
- `index.html`: ranked recommendations
- `clusters.html`: your clustered positive signals

## Optional Extras

- Default `uv sync` keeps the runtime env lean and avoids installing the PyTorch/CUDA stack.
- `uv sync --extra model-export` installs the ONNX export toolchain for `setup_model.py` and `export_tuned.py`.
- `uv sync --extra train` installs the training/tuning toolchain for `tune_embeddings.py`, `optimize_hyperparameters.py`, and `finetune_runpod.py`.
- On Linux, the `train` extra is large because it pulls PyTorch and CUDA wheels transitively.
- After a one-off export or training session, run `uv sync` again to return to the lean runtime env.

## Login Behavior

- Public mode works without logging in and uses public favorites only.
- Private signals such as upvotes and hidden stories require logging in as the same HN user.
- Successful login stores session cookies in `.cache/user/cookies.json`.

## How It Works

1. Load user signals from HN and local cache.
2. Fetch story details from Algolia item endpoints.
3. Embed positive and negative signals locally.
4. Cluster positive signals and optionally name clusters with Groq.
5. Fetch candidate stories from Algolia and optional RSS feeds.
6. Rank candidates with classifier mode when enough positive and negative signals exist, otherwise fall back to k-NN heuristics.
7. Select a diversified final list with a best-effort `2:1` HN:RSS mix.
8. Render static HTML and optional debug JSON artifacts.

## Configuration

There are two config surfaces today:
- CLI/runtime options in `generate_html.py`
- nested tuned parameters under `[hn_rerank.*]` in `hn_rerank.toml`

Top-level CLI-style keys currently read from `[hn_rerank]` include:
- `username`
- `output`
- `count`
- `signals`
- `candidates`
- `clusters`
- `days`
- `use_hidden_signal`
- `use_classifier`
- `contrastive`
- `knn`
- `no_naming`
- `no_rss`
- `no_tldr`
- `debug_scores`
- `debug_scores_path`
- `debug_clusters`
- `debug_clusters_path`

Tuned parameters are read from nested sections such as `[hn_rerank.ranking]`, `[hn_rerank.semantic]`, `[hn_rerank.classifier]`, and `[hn_rerank.clustering]`.

Example:

```toml
[hn_rerank]
username = "your-hn-username"
days = 7
clusters = 30
count = 30
no_rss = false
no_tldr = true
debug_scores = false

[hn_rerank.ranking]
negative_weight = 0.5529
diversity_lambda = 0.2397
diversity_lambda_classifier = 0.30
max_results = 500

[hn_rerank.semantic]
knn_neighbors = 1

[hn_rerank.classifier]
k_feat = 5
neg_sample_weight = 1.70

[hn_rerank.clustering]
similarity_threshold = 0.91
outlier_similarity_threshold = 0.87
min_samples_per_cluster = 2
max_cluster_fraction = 0.25
max_cluster_size = 40
refine_iters = 2
```

## Output and Debug Artifacts

Optional artifacts written next to the output HTML include:
- `scores_debug.json` via `--debug-scores`
- `cluster_name_debug.json` via `--debug-clusters`
- `cluster_stats.json` via `--cluster-stats`

The match badge shown in the UI is derived from `max_cluster_score`, not the blended `hybrid_score` used for ranking. `knn_score` remains available in debug output as a diagnostic neighborhood score.

## Systemd Timer

The repo includes `hn_rerank.service` and `hn_rerank.timer` for periodic runs.

Typical setup:

```bash
cp /home/dev/hn_rerank/hn_rerank.service ~/.config/systemd/user/hn_rerank.service
cp /home/dev/hn_rerank/hn_rerank.timer ~/.config/systemd/user/hn_rerank.timer
systemctl --user daemon-reload
systemctl --user enable --now hn_rerank.timer
```

Manual check:

```bash
systemctl --user start hn_rerank.service
journalctl --user -u hn_rerank.service -n 50 --no-pager
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): runtime data flow, ranking, clustering, and caching
- [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md): current backlog and cleanup targets
- [optuna.md](optuna.md): notes on tuning and promotion workflow

## Development

```bash
uv run pytest
uv run ruff check .
uv run ty check .
```

## Multi-Seed Promotion

Use the promotion runner to evaluate candidate parameter sets across multiple seeds and only emit promoted params when stability gates pass.

```bash
uv run promote_stable_params.py <your-hn-username> \
  --seeds 42,43,44 \
  --space core \
  --trials 120 \
  --cv-folds 8 \
  --top-k-per-seed 5 \
  --candidates 500 \
  --cache-only
```

Artifacts are written under `runs/promotion/<timestamp>_*` and include `promotion_report.json`. Promotion output files are only emitted when the candidate clears the configured stability gates.

## UI Change Verification

For browser-visible changes:

```bash
uv run generate_html.py <your-hn-username> --no-tldr
./scripts/check_ui_invariants.sh
./scripts/verify_showboat_demos.sh
```

If you add or materially change UI behavior, update or add a Showboat demo markdown file and verify it.
