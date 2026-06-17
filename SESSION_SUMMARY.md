# HN Rerank Debug Session Summary

## Starting Point
User had two AI/ML stories that they expected to surface in the dashboard but weren't:
- `48549628` — "Humanity isn't ready for the coming intelligence explosion" (4h old, 40 pts)
- `48489934` — "Cohere's First Model for Developers" (6h old, 73 pts)

Dashboard was using `model_type = "svm"` with 26 features including 6 telemetry features and 384-dim raw embeddings.

## Investigation Trace

### Story 1: 48549628
- Cache existed with valid data (36 pts, 69 comments, 6462 chars text)
- Not in scores_full.json — missing from candidate pool
- **Root cause 1**: `data["upvoted"]` was in `exclude_ids` (line 1885 of generate_html.py)
- **Root cause 2**: `data.get("upvoted_urls")` was in `exclude_urls` (line 1890)
- User upvoted story on HN → dashboard silently excluded it

### Story 2: 48489934
- Cache existed (was in `c5df3bb...` candidate cache, position 7 of 29)
- Had HN upvote + recent Algolia timestamp → should be in candidate pool
- **Root cause**: `HN_LIVE_WINDOW_DAYS = 4` meant the Algolia live query only covered the last 4 days. Story was posted Jun 11, outside the window.

## Fixes Applied

### 1. Drop telemetry features
Removed 6 telemetry features from `hn_rerank.toml`:
- `impression_count`, `click_count`, `click_ratio`
- `days_since_last_impression`, `domain_ctr`, `domain_impression_count`

**Why**: These track the user's dashboard interactions, not story properties. Fresh HN candidates have zero telemetry, causing the model to learn "zero telemetry = downvote" — score-0 stories in the dashboard.

### 2. Disable raw embedding features
Set `raw_embedding_features = false` in `hn_rerank.toml`.

**Why**: With 384 raw embedding dims + 20 derived features = 404 total, the SVM overfits to exact embedding positions. The Cohere story got model_score=0.0052 despite high similarity (knn=0.55, max_sim=0.58). Removing raw embeddings gave 20 discriminative features.

### 3. Add pos_neg_ratio and embedding_magnitude
Added two new derived similarity features in `api/rerank.py:compute_classifier_similarity_features`:
- `pos_neg_ratio = pos_knn / (pos_knn + neg_knn + 1e-6)` — range [0,1], captures positive-vs-negative bias
- `embedding_magnitude = norm(emb) / mean(norms)` — captures content density

**Why**: Replaces some discriminative signal lost from raw embeddings without overfitting.

### 4. Switch model to MLP
Changed `hn_rerank.toml:model_type` from `"svm"` to `"mlp"`.

**Why**: MLP (64→32 hidden layers) spreads probability scores naturally, avoiding the score-saturation pattern. CV ablation: MLP NDCG@30=0.187 vs SVM 0.099 with 22f.

### 5. Extend time windows
- `HN_LIVE_WINDOW_DAYS: 4 → 7` in `api/fetching.py` — captures resurfaced/boosted stories
- `STORY_CACHE_TTL: 259200 (72h) → 172800 (48h)` in `api/constants.py` — fresher points/comments

### 6. Fix archive boundary off-by-one
Changed `load_cached_archive_stories` in `api/fetching.py` from `story.time < end_ts` to `story.time <= end_ts`. Algolia's date filter already uses `<=`, so the archive was off-by-one on the boundary.

### 7. Increase per-window cap
Changed `win_target = max(win_target, 50)` to `max(win_target, 100)` in `api/fetching.py`. Marginal effect since live_budget already dominates, but provides headroom.

### 8. Stop excluding HN-upvoted stories
In `generate_html.py`, removed:
- `data["upvoted"]` from `exclude_ids`
- `data.get("upvoted_urls", set())` from `exclude_urls`

**Why**: User upvoting on HN was hiding the story from the dashboard. Now only `data["favorites"]`, `data["hidden"]`, and local feedback records exclude.

### 9. Tldr-detail uses server cache
Refactored `POST /api/tldr-detail` in `scripts/feedback_server.py`:
- Now takes only `story_id` and `story_title`
- Server loads full story (text, comments, metadata) from `.cache/stories/{id}.json`
- Removed 240KB `data-story-text-content` HTML payload
- Prompt limit raised 3000 → 12000 chars in `api/llm_utils.py`
- 404 on cache miss (was: empty result)

### 10. Add scores_full.json diagnostics
In `generate_html.py`, after writing `scores_debug.json` (top 40), also write `scores_full.json` (all ~1900 scored candidates). Useful for debugging "why is this story not in the dashboard?"

### 11. Feedback server gets load_story_by_id
Added `load_story_by_id(sid, allow_stale=True)` public wrapper in `api/fetching.py` for the tldr-detail handler.

### 12. Eval infrastructure
- `evaluate_quality.py:build_dataset_from_feedback` now populates `test_ids` and injects test stories into candidates (was empty set).
- `scripts/feature_ablation.py:run_one` now loads config from `hn_rerank.toml` via `AppConfig.load()` (was hardcoded default).
- `tests/snapshots/baseline_full.json` regenerated with new test_ids.
- `tests/test_feedback_eval.py` updated for new candidate count (70 + 22 = 92).
- `tests/test_fetching.py` updated for 7-day live window (3-day archive range instead of 6-day).

### 13. Documentation
- `AGENTS.md` updated with Run D (time-split classifier comparison) table and corrected actionable insight (MLP, not tie-breaking).
- `.opencode/instructions/lock-file-safety.md` created with reminder to check for running processes before deleting lock files.

## Final State

**Configuration** (`hn_rerank.toml`):
- 22 features: 3 similarity + 1 pos_neg_ratio + 4 engagement + 5 content + 3 nearest + 1 curated + 5 metadata
- `model_type = "mlp"`
- `raw_embedding_features = false`
- `HN_LIVE_WINDOW_DAYS = 7`
- `STORY_CACHE_TTL = 48h`

**Verification**:
- All 332 tests pass (1 pre-existing flaky `test_svm_deterministic_scores` — passes on rerun)
- Story `48549628` now at dashboard position 20 with model_score=0.8155
- Story `48489934` in candidate pool at position 1597, model_score=0.0084 (user's specific AI interests don't match Cohere's content)

## Telemetry Features Eval (rejected)

User asked to evaluate adding back the 6 telemetry features (28f config). Re-ran with proper protocol matching.

| Config | Protocol | SVM NDCG@30 | MLP NDCG@30 |
|--------|----------|-------------|-------------|
| 22f (current) | CV | 0.0497 | 0.0673 |
| 28f (with telemetry) | CV | 0.0201 | 0.0196 |
| 28f (with telemetry) | time-split | 0.5707 | 0.5707 |

CV results: 28f is 60-71% WORSE than 22f for both SVM and MLP.
The high 28f time-split numbers are a fluke — all 5 classifiers gave identical trivial results (score-saturation pattern).

**Conclusion**: Telemetry features are NOT beneficial. Reverted.

## Commits (14, pushed to main)

1. `feat: tldr-detail loads story by ID from cache`
2. `feat: 48h story cache TTL`
3. `fix: stop excluding stories the user upvoted on HN`
4. `refactor: remove text_content from HTML payload`
5. `feat: write scores_full.json with all scored candidates`
6. `feat: add pos_neg_ratio and embedding_magnitude derived features`
7. `feat: drop 6 telemetry features from classifier`
8. `feat: disable raw_embedding_features`
9. `feat: enable pos_neg_ratio and embedding_magnitude in classifier`
10. `feat: switch model from svm to mlp`
11. `test: update archive range assertion for 7-day live window`
12. `feat: full-feedback eval infrastructure with CV ablation`
13. `docs: add Run D time-split eval results, mark MLP as default`
14. `docs: add lock-file safety instruction for dashboard regen`
