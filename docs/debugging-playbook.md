# Debugging Playbook

This file captures the shortest path for common ranking and dashboard failures.

## First Files To Inspect

Start with:

- `hn_rerank.toml`
- `public/index.html`
- `public/scores_debug.json`
- `generate_html.py`
- `api/rerank.py`

Use `scores_debug.json` for selected-result truth first. Use `index.html` to
see what actually rendered.

## 1. "Why are there no external stories?"

Check:

1. `public/scores_debug.json` source mix
2. `generate_html.py::select_ranked_results`
3. `api/rerank.py` scoring behavior

Current expected behavior:

- selection should target 13 externals out of 40 when enough exist
- externals remain in the downstream pool

If selected externals are missing entirely:

- confirm the final slate quota logic is still running

## 2. "Why are many stories old?"

Check:

1. `hn_rerank.toml`:
   - `days`
   - `single_model.model_type`
   - `single_model.svm_kernel`
   - feature flags
2. current active score in `scores_debug.json`

Current reasons old stories can dominate:

- `days = 30`
- HN metadata features favor more mature stories
- the single model can still prefer stories with strong popularity signals

## 3. "Why does a story score strangely after a config change?"

Check:

1. `hn_rerank.toml`
2. `api/rerank.py::_classifier_metadata_features`
3. `api/feedback_single_model.py`

Common causes:

- feature flags changed between training and inference
- the wrong SVM kernel or regularization was selected
- the feedback replay is approximate, not exact vote-time reconstruction

## 4. "Why does `scores_debug.json` disagree with `index.html`?"

First suspect:

- `generate_html.py::filter_top_ranked_hn_dupes`

Current behavior:

- selection happens first
- dupe filter happens afterward
- there is no refill

Result:

- `scores_debug.json` can show 40 selected stories
- `index.html` can render fewer cards

## 5. "Why only 39 cards instead of 40?"

Current most likely cause:

- post-selection dupe filtering removed at least one HN submission

Verify:

- compare count in `scores_debug.json`
- compare count in `index.html`
- inspect generate log for:
  - `Filtered X duplicate HN submissions`

## 6. "Why is evaluation much slower with more data?"

Check:

- `evaluate_quality.py`
- `api/feedback_single_model.py`

Current bottlenecks:

- embedding work
- feature recomputation
- repeated fold training during CV

## 7. "Why does eval look much better than the dashboard?"

Remember:

- `evaluate_quality.py` injects held-out positives into the candidate pool
- cached eval is easier than live production
- `MRR` can look flattering even when top-k quality is mediocre

Trust first:

- `NDCG@10`
- `Precision@10`
- `Recall@10`

## 8. "Why did the trained model not activate?"

Check:

- whether `--classifier` was used in `evaluate_quality.py`
- whether there are at least the configured minimum positive and negative
  labels for the single model

Without enough labels, runtime falls back to centroid-max similarity.

## Quick Command Pattern

Current standard cached quality check:

```bash
uv run python evaluate_quality.py pure_coder --classifier --cache-only --age-matched --candidates 100
```

Current standard single-model sweep:

```bash
uv run python evaluate_quality.py pure_coder --classifier --cache-only --age-matched --final-list --count 40 --model-type svm --svm-kernel rbf --svm-c 3.0 --svm-gamma scale --use-new-features
```
