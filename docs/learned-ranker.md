# Learned Ranker

This file documents the current learned final reranker in `api/learned_ranker.py`.

## Purpose

The learned ranker is a final reranking layer over already-ranked candidates.

It does not operate on raw embeddings directly. It operates on:

- runtime rank outputs
- stored story metadata from dashboard feedback

## Current Runtime Position

Current flow in `generate_html.py`:

1. `rank_stories()` produces the CE-passed ranked pool
2. `apply_learned_ranker()` scores that pool
3. `select_ranked_results()` chooses the final slate from the learned-ranked pool

So the learned ranker currently affects both:

- final ordering
- final slate membership

## Activation

Current config from `hn_rerank.toml`:

- `shadow_enabled = true`
- `active_enabled = true`

Meaning:

- learned scores are computed
- learned ordering is active, not shadow-only

If `active_enabled = false`:

- scores may still be computed in shadow mode
- but downstream ranking stays on `hybrid_score`

## Labels

The model is trained from dashboard feedback records.

Label mapping:

- `down -> 0`
- `neutral -> 1`
- `up -> 2`

Training source constant in code:

- `TRAINING_SOURCE = "dashboard_feedback"`

## Model Shape

Current model in code:

- two-threshold ordinal setup
- implemented as two logistic pipelines:
  - `at_least_neutral`
  - `upvote`

Saved wrapper:

- `OrdinalThresholdModel`

Current constants:

- `MODEL_VERSION = 5`
- `MODEL_KIND = "ordinal_threshold_v1"`

## Feature Set

Current feature names:

1. `semantic_score`
2. `hybrid_score`
3. `max_cluster_score`
4. `knn_score`
5. `max_sim_score`
6. `cross_encoder_score`
7. `log_points`
8. `log_comments`

These are built from:

- `Story`
- `RankResult`

## Training / Load Behavior

Entry point used by runtime:

- `api.learned_ranker.train_or_load_and_score`

`generate_html.py::apply_learned_ranker`:

- trains or loads the model
- computes `learned_score` for each ranked result
- marks `learned_ranker_used`
- sorts by `learned_score` if active

Possible reported modes:

- `trained`
- `loaded`
- `disabled`
- `insufficient_labels`
- `failed`

## Current Risks

This layer is a second learned system on top of the first-stage model.

That means the repo currently stacks:

1. first-stage pairwise logistic ranker
2. learned ordinal final reranker

This can improve personalization, but it also makes behavior harder to explain
because the learned final ranker can counteract the earlier model.

## Practical Interpretation

If dashboard results look surprising, check:

- whether `learned_ranker.active_enabled` is true
- whether `learned_ranker_used` is true in `scores_debug.json`
- whether `learned_score` and `hybrid_score` disagree materially
