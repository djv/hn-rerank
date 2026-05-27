# Ranking Pipeline

This file describes the current runtime ranking flow in the repo as of the
current `main` worktree.

## Overview

The dashboard build in `generate_html.py` uses a staged pipeline:

1. Candidate retrieval
2. Single-model scoring in `api/rerank.py`
3. Final slate selection in `generate_html.py`
4. Post-selection HN dupe filtering
5. HTML/debug output

The main public score field is `model_score`, the active
single-model score used for ordering.

## 1. Candidate Retrieval

Entry points:

- `generate_html.py`
- `api/fetching.py::get_best_stories`

Current config defaults from `hn_rerank.toml`:

- `days = 30`
- `count = 40`
- `archive.open_index_enabled = true`
- `archive.use_cached_stories = true`

Candidate sources:

- HN live window
- HN archive cache
- Open Index archive ids, then Algolia hydration
- RSS / external sources unless `no_rss=true`

## 2. First-Stage Model

Entry point:

- `api/rerank.py::rank_stories`

Current runtime mode:

- `single_model.model_type = "svm"`
- `single_model.svm_kernel = "rbf"`
- `single_model.svm_c = 3.0`
- `single_model.svm_gamma = "scale"`

Activation rule:

- the trained model is used only when there are at least the configured
  minimum positive and negative feedback labels
- otherwise the code falls back to centroid-max cosine similarity

What happens:

1. candidate story text is embedded
2. positive and negative history embeddings are prepared
3. positive embeddings are clustered into interest centroids
4. derived similarity features are built
5. metadata features are appended
6. a feedback-trained single model is scored
7. candidate scores are produced as normalized ranking scores

In the current code, `model_score` is set to this single-model
score.

## 3. Final Slate Selection

Entry point:

- `generate_html.py::select_ranked_results`

This stage does not learn or rescore. It applies slate policy:

- splits ranked results into HN vs external
- reserves a small fixed external quota
- enforces per-source diversity for external stories
- chooses the final `count`
- sorts the selected slate by active final score:
  - `model_score`

Current external target:

- `round(count * 0.2) + 5`

With `count = 40`, the target is `13` externals when enough are available.

## 4. Post-Selection HN Dupe Filtering

Entry point:

- `generate_html.py::filter_top_ranked_hn_dupes`

This happens after slate selection.

Current status:

- the function is a no-op (body is `return ranked`)
- the call site and function are preserved for future re-enablement
- the rendered page always contains exactly `count` cards when enough candidates exist

## 5. Output

Final outputs:

- `public/index.html`
- `public/clusters.html`
- `public/scores_debug.json` when `debug_scores=true`

`scores_debug.json` is the easiest artifact for inspecting:

- source mix
- active score fields
- classifier feature usage
- single-model ordering behavior

## Current Practical Read

The runtime system is currently one learned ranker plus slate policy:

1. feedback-trained single model in `api/rerank.py`
2. final slate selection in `generate_html.py`

That is simpler to reason about than the old stacked CE/learned-ranker path,
but it still depends heavily on feature quality and feedback-label coverage.
