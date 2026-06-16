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

## Feature Ablation Results (June 2026)

Two ablation runs on the same 26-feature baseline. The snapshots use
different eval paradigms — see "Snapshot protocol" below.

### Run A — single-split on `tests/snapshots/baseline.json` (legacy)

Last run: `uv run scripts/feature_ablation.py tests/snapshots/baseline.json --feedback .cache/user_feedback/dashboard_feedback.json`
Snapshot: 48 train / 12 test / 780 candidates / 1406 vote-age / 809 domain-recency records from feedback.

| Feature | Dropped MRR | ΔMRR | Impact |
|---------|------------|------|--------|
| (all 26) | 0.333 | — | baseline |
| `local_density` | 0.333 | 0.000 | neutral |
| `centroid` | 0.200 | -0.133 | helpful |
| `pos_knn` | 0.200 | -0.133 | helpful |
| `neg_knn` | 0.200 | -0.133 | helpful |
| `log_points` | 0.200 | -0.133 | helpful |
| `log_comments` | 0.200 | -0.133 | helpful |
| `comment_ratio` | 0.200 | -0.133 | helpful |
| `closest_margin` | 0.200 | -0.133 | helpful |
| `closest_pos` | 0.250 | -0.083 | helpful |
| `closest_neg` | 0.200 | -0.133 | helpful |
| `days_since_last_impression` | 0.333 | 0.000 | neutral |
| `impression_count` | 0.200 | -0.133 | helpful |
| `click_count` | 0.200 | -0.133 | helpful |
| `domain_impression_count` | 0.200 | -0.133 | helpful |
| `story_age` | 0.200 | -0.133 | helpful |
| `cluster_size` | 0.200 | -0.133 | **helpful** |
| `domain_recency` | 0.200 | -0.133 | **helpful** |
| `source_trust` | 0.111 | -0.222 | **critical** |

**Run A takeaways:** with a small training set, the model relies on most
features. `source_trust` is the single most important (~67% MRR drop when
removed). `cluster_size` and `domain_recency` are helpful.

### Run B — 5-fold CV on `tests/snapshots/baseline_full.json` (full feedback, drop-one)

Last run: `uv run scripts/feature_ablation.py tests/snapshots/baseline_full.json --feedback .cache/user_feedback/dashboard_feedback.json --cv 5`
Snapshot: 543 train (CV) / 135 test (CV) / 510 candidates / 745 neg / 1419 vote-age / 812 domain-recency records from feedback.

| Feature Removed | MRR | NDCG@30 | hit@30 | mean_rank |
|----------------|------|---------|--------|-----------|
| (none — all 26) | 0.033 | 0.018 | 0.400 | 169.3 |
| `text_len` | 0.168 | 0.043 | 0.600 | 164.8 |
| `closest_pos` | 0.046 | 0.025 | 0.400 | 167.1 |
| `is_hn` | 0.034 | 0.018 | 0.400 | 167.6 |
| `days_since_last_impression` | 0.034 | 0.019 | 0.400 | 153.9 |
| `log_comments` | 0.034 | 0.019 | 0.400 | 166.7 |
| `log_points` | 0.033 | 0.019 | 0.400 | 166.4 |
| `pos_knn` | 0.033 | 0.018 | 0.400 | 166.2 |
| `neg_knn` | 0.033 | 0.018 | 0.400 | 165.8 |
| `comment_ratio` | 0.033 | 0.018 | 0.400 | 165.6 |
| `is_github` | 0.033 | 0.019 | 0.400 | 166.3 |
| `is_pdf` | 0.033 | 0.018 | 0.400 | 165.9 |
| `comments_count` | 0.033 | 0.019 | 0.400 | 166.7 |
| `impression_count` | 0.033 | 0.018 | 0.400 | 163.0 |
| `click_count` | 0.033 | 0.018 | 0.400 | 167.8 |
| `click_ratio` | 0.033 | 0.018 | 0.400 | 167.5 |
| `domain_ctr` | 0.033 | 0.018 | 0.400 | 165.9 |
| `domain_impression_count` | 0.033 | 0.018 | 0.400 | 165.7 |
| `domain_recency` | 0.033 | 0.019 | 0.400 | 166.2 |
| `local_density` | 0.033 | 0.018 | 0.400 | 165.8 |
| `title_len` | 0.032 | 0.018 | 0.400 | 165.8 |
| `closest_margin` | 0.032 | 0.014 | 0.400 | 161.7 |
| `centroid` | 0.031 | 0.018 | 0.400 | 166.8 |
| `cluster_size` | 0.031 | 0.018 | 0.400 | 165.9 |
| `source_trust` | 0.031 | 0.013 | 0.400 | 182.0 |
| `closest_neg` | 0.030 | 0.018 | 0.400 | 159.6 |
| `story_age` | 0.030 | 0.013 | 0.400 | 179.1 |

**Run B interpretation (corrected):** The drop-one CV shows near-uniform
metrics (MRR 0.030-0.046, NDCG@30 0.013-0.043, hit@30 0.400-0.600). This
is **not** because features are unimportant — it is a **score saturation
artifact.** The SVM with 26 features produces overconfident probabilities;
many candidates hit score=1.0, creating massive ties. Tie-breaking
determines the test story's rank, making MRR and NDCG collapse regardless
of which features are present. See Run C below for the proof.

### Run C — 5-fold CV single-feature ranking (full feedback)

Last run: `uv run scripts/feature_ablation.py tests/snapshots/baseline_full.json --feedback .cache/user_feedback/dashboard_feedback.json --cv 5 --single-features --no-drop-one`
Same snapshot / feedback as Run B.

| Feature Only | MRR | NDCG@30 | hit@30 | mean_rank |
|-------------|------|---------|--------|-----------|
| (all 26 — baseline) | 0.033 | 0.018 | 0.400 | 169.3 |
| `story_age` | 1.000 | **0.448** | 1.000 | 164.4 |
| `source_trust` | 1.000 | **0.422** | 1.000 | 163.1 |
| `click_count` | 1.000 | 0.326 | 1.000 | 196.8 |
| `click_ratio` | 1.000 | 0.316 | 1.000 | 198.4 |
| `days_since_last_impression` | 0.900 | 0.314 | 1.000 | 221.9 |
| `domain_recency` | 1.000 | 0.303 | 1.000 | 200.7 |
| `neg_knn` | 1.000 | 0.302 | 1.000 | 201.2 |
| `local_density` | 1.000 | 0.298 | 1.000 | 201.1 |
| `domain_ctr` | 1.000 | 0.297 | 1.000 | 200.4 |
| `pos_knn` | 1.000 | 0.297 | 1.000 | 199.2 |
| `cluster_size` | 1.000 | 0.297 | 1.000 | 201.1 |
| `domain_impression_count` | 1.000 | 0.296 | 1.000 | 202.2 |
| `is_hn` | 1.000 | 0.294 | 1.000 | 202.3 |
| `is_github` | 1.000 | 0.294 | 1.000 | 200.7 |
| `title_len` | 1.000 | 0.293 | 1.000 | 200.8 |
| `comment_ratio` | 1.000 | 0.287 | 1.000 | 195.4 |
| `is_pdf` | 1.000 | 0.286 | 1.000 | 201.5 |
| `impression_count` | 0.800 | 0.279 | 1.000 | 209.9 |
| `log_comments` | 1.000 | 0.272 | 1.000 | 184.8 |
| `comments_count` | 1.000 | 0.272 | 1.000 | 184.8 |
| `centroid` | 1.000 | 0.271 | 1.000 | 198.0 |
| `log_points` | 1.000 | 0.258 | 1.000 | 187.7 |
| `closest_pos` | 0.224 | 0.031 | 0.400 | 199.1 |
| `closest_neg` | 0.073 | 0.024 | 0.600 | 234.6 |
| `text_len` | 0.077 | 0.080 | 1.000 | 188.4 |
| `closest_margin` | 0.029 | 0.009 | 0.200 | 207.7 |

**Run C takeaways — the smoking gun:** Nearly every single feature
achieves MRR=1.000 and hit@30=1.000 when used alone. The 26-feature
model (baseline) gets NDCG@30=0.018 — **25× worse** than `story_age`
alone (0.448) and **23× worse** than `source_trust` alone (0.422).
The model with all features destroys ranking quality through score
saturation.

**Structural explanation:** The `OrdinalThresholdModel` in
`api/ordinal_model.py:371-376` clips `at_least_neutral` and `upvote`
probabilities to `[0,1]` and takes the arithmetic mean. With 26
features, the SVM produces confident probabilities → both values hit
1.0 for many candidates → massive ties → tie-breaking scatters test
stories through the saturated cluster. The linear model with a single
feature does not saturate — the score distribution has room to spread.

**Summary of all three runs:**
- **Run A** (48 train, single-split): clear feature importance;
  `source_trust` critical. This is the right tool for *feature
  importance* questions.
- **Run B** (543 train, CV drop-one): flat MRR/NDCG across all feature
  drops due to score saturation, not feature redundancy.
- **Run C** (543 train, CV single-feature): proves the saturation
  hypothesis. Single features work well; the 26-feature combination is
  broken. This is the right tool for *model quality* assessment.

**Actionable insight:** The SVM saturation is fixed by switching to MLP
(64→32 hidden layers). MLP's non-linear representation spreads probability
scores naturally — no tie-breaking or feature reduction needed.

### Run D — Time-split classifier comparison (543 train, 590 candidates)

Last run: `uv run scripts/feature_ablation.py tests/snapshots/baseline_full.json --feedback .cache/user_feedback/dashboard_feedback.json --no-drop-one --no-single-features --baseline TYPE`

Time-split (80/20) on the full feedback snapshot. Train on older 80% of
upvotes, test on newest 20%. Candidates from legacy snapshot + injected
test stories. This is the most production-representative eval: the model
must rank future user interactions among a fresh candidate pool.

| Classifier | NDCG@30 | mean_rank | P@5 |
|-----------|---------|-----------|-----|
| **MLP** | **0.814** | 197.4 | 1.000 |
| SVM (RBF) | 0.777 | 232.6 | 1.000 |
| Random Forest | 0.723 | **189.3** | 1.000 |
| Gradient Boost | 0.629 | 205.5 | 1.000 |
| Logistic | 0.604 | 200.5 | 0.800 |

**Run D takeaways:** All classifiers achieve hit@30=1.000 (trivially, with
134 test stories among 590 candidates). NDCG@30 is the discriminating metric.
MLP leads by 4.8% over SVM. All tree/linear alternatives underperform.

The current `hn_rerank.toml` uses `model_type = "mlp"` (switched 2026-06-15).

### Snapshot protocol

- `tests/snapshots/baseline.json` (48 train / 12 test / 780 candidates) — the
  legacy small eval. Time-split 80/20 of an older feedback snapshot.
- `tests/snapshots/baseline_full.json` (543 train / 135 test / 591 candidates) —
  regenerated from all cached feedback. `test_ids` populated and test stories
  injected into candidates (see `build_dataset_from_feedback` in
  `evaluate_quality.py`). CV re-splits internally from all_stories.
- Regen the full snapshot: `uv run scripts/regen_full_snapshot.py` (no network).

### Operational notes

- **Harness must always be run with `--feedback .cache/user_feedback/dashboard_feedback.json`** for representative `story_age` and `domain_recency` measurements.
- Use `--cv 5` for the full-feedback eval (Run B protocol).
- Use `--cv 5 --single-features --no-drop-one` for single-feature quality assessment (Run C protocol).
- Drop `--cv` for the legacy small eval (Run A protocol).
- Without `--cv`, uses the stored time-split (Run D protocol — most realistic).
- `feature_ablation.py:run_one()` now reads `model_type` from `hn_rerank.toml` (`AppConfig.load()`), not the hardcoded default. Changes in TOML are reflected in ablation runs.
- Both runs include the `cluster_size` and `domain_recency` features.

## Repo notes

- `hn_rerank.toml` controls most runtime behavior.
- `README.md` is the canonical user-facing setup reference.
- `public/index.html` and `public/clusters.html` are generated artifacts.
- Tests marked `@pytest.mark.slow` (6 tests: integration, pagination, SVM, clustering stability) are skipped with `-m "not slow"`.
- `api/rerank.py` module-global caches for metadata features: `_local_density_cache`, `_story_age_at_vote_map`, `_cluster_size_cache`, `_domain_recency_map`. All reset at the start of `rank_stories` (or set externally by `main()` in `generate_html.py`). The story_age and domain_recency maps are populated from FeedbackRecord in `generate_html.py:1955-1975` and cleared in a `finally` block.
- `scripts/feature_ablation.py` accepts `--feedback path` to populate both `story_age` and `domain_recency` caches during ablation. Always use `--feedback .cache/user_feedback/dashboard_feedback.json` for representative measurements.
- `tests/snapshots/baseline_full.json` (543 train / 135 test / 510 candidates) is generated from all cached feedback via `uv run scripts/regen_full_snapshot.py`. No network required. `tests/snapshots/baseline.json` is kept for fast unit-test evals.
- `scripts/feature_ablation.py` accepts `--cv N` to use k-fold cross-validation via `evaluate_cv` (more realistic, harder metric). Without `--cv`, it uses a single 80/20 time-split.

## Long-running services

All persistent services run as **systemd user services**. Never start long-running Python processes with `&` or `nohup` — use systemd.

| Service | File | Manage |
|---------|------|--------|
| Feedback API | `~/.config/systemd/user/hn_rerank_feedback.service` | `systemctl --user {start\|stop\|restart\|status} hn_rerank_feedback` |
| Dashboard regen timer | `~/.config/systemd/user/hn_rerank.timer` + `hn_rerank.service` | `systemctl --user {start\|stop} hn_rerank.timer` |
| Caddy reverse proxy | `/etc/systemd/system/hn-dashboard.service` | `sudo systemctl {start\|stop\|restart\|status} hn-dashboard` |

The feedback server auto-triggers dashboard regeneration on upvote/neutral/down via `api/regen_scheduler.py`. Regen logs to `.cache/regen.log`.

## Engineering philosophy (Harness Engineering)

Derived from OpenAI's "Harness engineering: leveraging Codex in an agent-first world" (Feb 2026):

- **Progressive disclosure** — layered context; pointer-rich, not dump-everything. Directory-level instructions first, deeper pointers on request.
- **Repository-first artifacts** — plans, decisions, tech debt, progress tracked as versioned files in the repo, not external tools.
- **Rigid boundaries, flexible internals** — enforce invariants and dependency direction via automated checks (ruff, ty, pytest); let the agent choose *how* inside the boundary.
- **Autonomous loop** — user describes the goal; agent gathers context, writes code/tests, self-reviews, iterates until all gates pass.
- **Mechanical enforcement, not micromanagement** — automated gates replace manual prescription of implementation details.
