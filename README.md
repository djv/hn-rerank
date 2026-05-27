# HN Rerank

> A local-first Hacker News dashboard that ranks stories against your own history.

HN Rerank builds a static dashboard from your HN signals, Algolia stories,
open-index archive candidates, and optional RSS feeds. Embeddings, clustering,
single-model reranking, caching, and HTML generation run locally. Optional
cluster names and TL;DRs use the configured LLM provider.

Requires Python 3.12+.

## What It Does

- Builds a preference profile from favorites, upvotes, and hidden stories.
- Fetches candidate stories from Algolia in cache-friendly live windows.
- Adds older HN candidates from cached archive stories and, when enabled,
  open-index archive data.
- Expands the pool with curated RSS/Atom feeds and full-article extraction.
- Clusters your positive signals into interest groups.
- Reranks candidates with local embeddings, a feedback-trained single model,
  and diversity.
- Renders `public/index.html` and `public/clusters.html` as static artifacts by
  default.

## Runtime Characteristics

- Local:
  embeddings, clustering, reranking, caching, and HTML generation.
- External services:
  Hacker News, Algolia, RSS feeds, Hugging Face open-index data, and optionally
  Mistral or Groq for cluster names and TL;DRs.
- Built-in extra feeds:
  `jack-clark.net`, `lobste.rs`, `tildes.net`, `lesswrong.com`, `Slashdot`,
  GitHub Trending, `r/MachineLearning`, `r/programming`, `r/compsci`, and
  Digg AI are appended to the OPML feed set.
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
3. Configure LLM keys if cluster naming or TL;DRs are enabled. The CLI
   preflight currently expects `GROQ_API_KEY`; Mistral runs also need
   `MISTRAL_API_KEY`.
   ```bash
   export GROQ_API_KEY="your-groq-key"
   export MISTRAL_API_KEY="your-mistral-key"
   ```
   To run without LLM calls, pass `--no-naming --no-tldr`.
4. Generate the dashboard.
   ```bash
   uv run generate_html.py <your-hn-username> --days 7 --clusters 30
   ```

Outputs:
- `public/index.html`: ranked recommendations
- `public/clusters.html`: your clustered positive signals

## Optional Extras

- Default `uv sync` keeps the runtime env lean and avoids installing the PyTorch/CUDA stack.
- `uv sync --extra archive-vector` installs DuckDB and SentenceTransformers for one-off searches against the public HN VectorSearch archive.
- `uv sync --extra model-export` installs the ONNX export toolchain for `setup_model.py`.
- `uv sync --extra train` installs the training toolchain for `tune_embeddings.py` and `finetune_runpod.py`.
- On Linux, the `archive-vector` and `train` extras are large because they pull PyTorch and CUDA wheels transitively.
- After a one-off export or training session, run `uv sync` again to return to the lean runtime env.

## Login Behavior

- Public mode works without logging in and uses public favorites only.
- Private signals such as upvotes and hidden stories require logging in as the same HN user.
- Successful login stores session cookies in `.cache/user/cookies.json`.

## How It Works

1. Load user signals from HN and local cache.
2. Fetch story details from Algolia item endpoints.
3. Embed positive and negative signals locally.
4. Cluster positive signals and optionally name clusters with the configured
   LLM provider.
5. Fetch candidate stories from Algolia live windows, local archive cache,
   optional open-index archive data, and RSS feeds.
6. Rank candidates with the feedback-trained single model when enough positive
   and negative signals exist, otherwise fall back to centroid-max semantic
   scoring.
7. Select a diversified final list and then apply HN source deduplication.
8. Render static HTML and optional debug JSON artifacts.

## HN Vector Archive Experiment

The public ClickHouse/Hugging Face VectorSearch dataset contains MiniLM
embeddings for historical HN items. To search it with your cached HN upvotes:

```bash
uv sync --extra archive-vector
uv run python -m scripts.query_hn_vector_archive <your-hn-username> --top-k 10 --cache-only --min-score 10 --min-comments 1
```

The script embeds your cached upvoted story texts with
`sentence-transformers/all-MiniLM-L6-v2`, averages them into one profile vector,
then scans the archive for the nearest non-deleted HN stories. The full archive
scan is large, so use `--limit-shards 1` for a quick smoke test. `--min-score`
and `--min-comments` filter out low-quality archive matches before final
selection.

## Configuration

Most scheduled behavior is controlled by `hn_rerank.toml`. CLI arguments can
override several top-level options for one-off runs.

Top-level keys currently loaded from `[hn_rerank]`:
- `username`
- `output`
- `days`
- `count`
- `candidates`
- `signals`
- `contrastive`
- `no_rss`
- `no_tldr`
- `no_naming`
- `debug_scores`
- `debug_clusters`

Nested config sections currently loaded:
- `[hn_rerank.ranking]`
- `[hn_rerank.semantic]`
- `[hn_rerank.classifier]`
- `[hn_rerank.clustering]`
- `[hn_rerank.llm]`
- `[hn_rerank.archive]`
- `[hn_rerank.single_model]`

Older local TOML files can still contain removed legacy sections or keys. The
loader ignores unknown fields instead of failing, but the sections above are
the supported runtime surface.

`--clusters` and `--open-index-archive` are CLI-only overrides. The deprecated
`--bigquery-archive` flag is an alias for `--open-index-archive`.

Example:

```toml
[hn_rerank]
username = "your-hn-username"
output = "public/index.html"
days = 7
count = 50
candidates = 1000
signals = 2000
no_rss = false
no_tldr = false
no_naming = false
debug_scores = false

[hn_rerank.llm]
provider = "mistral"

[hn_rerank.archive]
open_index_enabled = true
use_cached_stories = true
open_index_candidate_limit = 50

[hn_rerank.ranking]
negative_weight = 0.26932
diversity_lambda = 0.23966
max_results = 500

[hn_rerank.semantic]
knn_neighbors = 6
match_threshold = 0.85

[hn_rerank.classifier]
scoring_mode = "pairwise_logistic"
pairwise_negatives = 15
k_feat = 7
use_neg_knn_feature = true
use_log_points_feature = true
use_log_comments_feature = true
use_closest_pos_feature = true
use_closest_neg_feature = true
use_title_len_feature = true
use_text_len_feature = true
use_has_url_feature = true
use_github_feature = true
use_pdf_feature = true
use_comments_count_feature = true

[hn_rerank.clustering]
similarity_threshold = 0.55
min_clusters = 2
max_clusters = 40

[hn_rerank.single_model]
min_positive_labels = 10
min_negative_labels = 10
balance_training_labels = true
model_type = "svm"
svm_kernel = "rbf"
svm_c = 3.0
svm_gamma = "scale"
```

## Output and Debug Artifacts

Optional artifacts written next to the output HTML include:
- `scores_debug.json` via `--debug-scores`
- `cluster_name_debug.json` via `--debug-clusters`

The match badge shown in the UI is derived from `max_cluster_score`, not the
active ordering score. `model_score` is the main
score field in debug output, alongside `knn_score`, `max_sim_score`, and
`max_cluster_score`. The old cross-encoder and learned-ranker debug fields are
no longer emitted by the live runtime.

## Systemd Timer

The repo includes `hn_rerank.service` and `hn_rerank.timer` for periodic runs.
The service runs `update_dashboard.sh`, and that script runs
`uv run generate_html.py` from `/home/dev/hn_rerank`. Runtime generation
settings live in `hn_rerank.toml`; the execution cadence lives in
`hn_rerank.timer`. The repo timer runs every 3 hours after a 15-minute boot
delay, with up to 5 minutes of randomized delay.

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

`update_dashboard.sh` loads secrets from the systemd environment, then
`~/.config/hn_rerank/secrets.env`, then the repo-local `.env` fallback. It logs
to `update.log`.

## Dashboard Feedback

The generated dashboard can save upvote/downvote feedback through a tiny local
writer service. The dashboard remains static, and button clicks are designed to
POST to `/api/feedback`; expose that route to the local writer service only in
the deployment layer you already use.

Setup:

```bash
printf 'HN_RERANK_FEEDBACK_TOKEN=%s\n' '<choose-a-long-random-token>' >> ~/.config/hn_rerank/secrets.env
cp /home/dev/hn_rerank/hn_rerank_feedback.service ~/.config/systemd/user/hn_rerank_feedback.service
systemctl --user daemon-reload
systemctl --user enable --now hn_rerank_feedback.service
```

The service listens on `127.0.0.1:8765`. Expose it through the existing dashboard
Caddy server by adding a narrow `/api/feedback*` reverse proxy to that local
service before the static file handler:

```caddyfile
handle /api/feedback* {
    reverse_proxy 127.0.0.1:8765
}
```

Feedback is stored locally at `.cache/user_feedback/dashboard_feedback.json`.
Existing HN favorites, upvotes, and hidden stories remain the main historical
signals. Dashboard feedback is layered on top for stories explicitly voted in
the dashboard: upvotes become extra positive signals, downvotes become extra
negative signals, and voted stories are excluded from future candidate results.
For HN stories, dashboard upvote attempts an HN upvote and dashboard downvote
attempts HN hide; local feedback is kept even if that HN sync fails.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): runtime data flow, ranking, clustering, and caching
- [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md): current backlog and cleanup targets
- [archive/MODEL_PROVENANCE.md](archive/MODEL_PROVENANCE.md): active embedding artifact and rollback notes
- [demo.md](demo.md): current UI verification workflow

## Development

```bash
uv run pytest
uv run ruff check .
uv run ty check .
```

## UI Change Verification

For browser-visible changes:

```bash
uv run generate_html.py <your-hn-username> --no-tldr
./scripts/check_ui_invariants.sh
./scripts/verify_showboat_demos.sh
```

If you add or materially change UI behavior, update [demo.md](demo.md) and verify it.
