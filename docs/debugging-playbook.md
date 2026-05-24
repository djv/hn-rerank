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
3. `api/rerank.py` CE behavior

Current expected behavior:

- selection should target 13 externals out of 40 when enough exist
- CE should restrict HN only
- externals should remain in the downstream pool

If selected externals are missing entirely:

- confirm CE did not replace the whole downstream pool with only the CE slice

## 2. "Why are many stories old?"

Check:

1. `hn_rerank.toml`:
   - `days`
   - learned-ranker flags
2. current active score in `scores_debug.json`
3. whether learned ranker is active

Current reasons old stories can dominate:

- `days = 30`
- HN metadata features (`log_points`, `log_comments`) are enabled
- learned ranker is active and can favor mature stories

## 3. "Why do many cards have CE score 0?"

Check:

- whether the story is external
- whether it was inside the top `cross_encoder.top_n` HN slice

Current expected behavior:

- externals bypass CE
- only top-HN CE slice gets nonzero `cross_encoder_score`

So many zero CE scores are expected.

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
- `api/rerank.py::_score_cross_encoder_candidate`

Current bottleneck:

- cross-encoder ONNX scoring during ranking

Large full-data evals often spend most of their time there.

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
- whether there are at least:
  - 5 positives
  - 5 negatives

Without enough negatives, runtime falls back to centroid-max similarity.

## Quick Command Pattern

Current standard cached quality check:

```bash
uv run python evaluate_quality.py pure_coder --classifier --cache-only --age-matched --candidates 100
```

Current standard regeneration:

```bash
uv run python generate_html.py
```
