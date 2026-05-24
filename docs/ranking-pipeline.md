# Ranking Pipeline

This file describes the current runtime ranking flow in the repo as of the
current `main` worktree.

## Overview

The dashboard build in `generate_html.py` uses a staged pipeline:

1. Candidate retrieval
2. First-stage model scoring in `api/rerank.py`
3. Cross-encoder rerank on HN candidates
4. Learned final rerank over the CE-passed pool
5. Final slate selection in `generate_html.py`
6. Post-selection HN dupe filtering
7. HTML/debug output

The main public score field is still named `hybrid_score`, but in the current
runtime path it is mostly "current ranking score", not a legacy hand-tuned HN
blend.

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

- `classifier.scoring_mode = "pairwise_logistic"`
- `classifier.feature_mode = "bottleneck"`

Activation rule:

- the trained model is used only when there are at least:
  - `classifier.min_positive_examples = 5`
  - `classifier.min_negative_examples = 5`
- otherwise the code falls back to centroid-max cosine similarity

What happens:

1. candidate story text is embedded
2. positive and negative history embeddings are prepared
3. positive embeddings are clustered into interest centroids
4. derived similarity features are built
5. optional metadata features are appended
6. a pairwise logistic model is trained on-the-fly
7. candidate scores are produced as normalized first-stage scores

In the current code, `hybrid_score` is initially set to this first-stage score:

- `hybrid_scores = semantic_scores`

## 3. Cross-Encoder Rerank

Entry point:

- `api/rerank.py`, inside `rank_stories`

Current config:

- `cross_encoder.enabled = true`
- `cross_encoder.top_n = 200`
- `cross_encoder.weight = 0.06953`

Important current behavior:

- CE is applied only to HN candidates
- the top `N` HN candidates from the first-stage ranked order are CE-scored
- the CE score is blended back into `hybrid_score`
- once CE is active, the downstream HN pool is restricted to the CE-scored HN
  slice
- external stories are not CE-scored, but they remain in the downstream pool

This was changed to avoid a failure mode where the CE slice eliminated external
stories before quota-based final selection could see them.

## 4. Learned Final Rerank

Entry point:

- `generate_html.py::apply_learned_ranker`
- model code in `api/learned_ranker.py`

Current config:

- `learned_ranker.shadow_enabled = true`
- `learned_ranker.active_enabled = true`

That means the learned ranker is currently active, not shadow-only.

What happens:

1. the CE-passed pool is scored by the learned ranker
2. each `RankResult` gets:
   - `learned_score`
   - `learned_ranker_used`
3. if `active_enabled=true`, the pool is sorted by `learned_score`

At this point the list order is the active final rank order for selection.

## 5. Final Slate Selection

Entry point:

- `generate_html.py::select_ranked_results`

This stage does not learn or rescore. It applies slate policy:

- splits ranked results into HN vs external
- reserves a small fixed external quota
- enforces per-source diversity for external stories
- chooses the final `count`
- sorts the selected slate by active final score:
  - `learned_score` when learned ranker is active
  - otherwise `hybrid_score`

Current external target:

- `round(count * 0.2) + 5`

With `count = 40`, the target is `13` externals when enough are available.

## 6. Post-Selection HN Dupe Filtering

Entry point:

- `generate_html.py::filter_top_ranked_hn_dupes`

This happens after slate selection.

Behavior:

- fetches live HN item-page HTML for selected HN stories
- detects moderator-marked duplicate submissions
- removes dupes from the final list

Important current limitation:

- there is no refill after a dupe is removed
- so the rendered page can contain fewer than `count` cards

## 7. Output

Final outputs:

- `public/index.html`
- `public/clusters.html`
- `public/scores_debug.json` when `debug_scores=true`

`scores_debug.json` is the easiest artifact for inspecting:

- source mix
- active score fields
- CE coverage
- learned-ranker flags

## Current Practical Read

The runtime system is currently two learned rankers in sequence:

1. first-stage pairwise logistic model in `api/rerank.py`
2. learned final reranker in `api/learned_ranker.py`

That is workable, but it also means ranking behavior can be hard to reason
about unless the stage boundaries are kept explicit.
