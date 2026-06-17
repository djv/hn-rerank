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

**⚠️ Superseded:** Runs A–D were executed on the buggy eval pipeline
(thread-unsafe caches, fold leakage, random CV, metric saturation).
Results are retained for historical reference but should not be used
for decisions. The 4×4 ablation below is authoritative.

### Run E — 4×4 Feature×Model Ablation (639 train, 481 candidates)

Last run: `bash scripts/run_4x4_ablation.sh`
Snapshot: `tests/snapshots/baseline_full.json` (639 train / 50 test / 481 candidates).
Eval: time-split (80/20), cleaned pipeline (thread-safe caches, no fold leakage,
deterministic tie-breaking, `max_test=50`). All cells use `--no-drop-one
--no-single-features --cv 0`.

**4 feature configs:** 22f (current TOML — 22 metadata features), 22f+raw (406 dims),
16f (no engagement/signal features), 16f+raw (400 dims).

**4 model configs:** mlp-relu (default), mlp-sigmoid, mlp-alpha (L2=0.01), svm-rbf.

All cells achieved MRR=1.000, hit@30=1.000, nonhn_at_0.5=0.00 (no plateau).

| Cell | NDCG@30 | mean_rank | P@5 |
|------|---------|-----------|-----|
| **16f+raw-svm-rbf** | **0.737** | 116.6 | 1.000 |
| 22f-svm-rbf | 0.701 | 157.3 | 1.000 |
| 22f+raw-svm-rbf | 0.681 | 118.5 | 1.000 |
| 16f-mlp-alpha | 0.650 | 162.1 | 1.000 |
| 16f-mlp-sigmoid | 0.649 | 136.2 | 1.000 |
| 16f-mlp-relu | 0.646 | 166.8 | 1.000 |
| 16f+raw-mlp-alpha | 0.635 | 114.0 | 0.600 |
| 16f+raw-mlp-relu | 0.635 | 114.8 | 0.600 |
| 16f-svm-rbf | 0.626 | 150.9 | 1.000 |
| 22f+raw-mlp-alpha | 0.597 | 124.0 | 0.800 |
| 22f-mlp-alpha | 0.589 | 167.4 | 1.000 |
| 16f+raw-mlp-sigmoid | 0.582 | 106.3 | 0.800 |
| 22f+raw-mlp-relu | 0.575 | 124.4 | 0.800 |
| 22f-mlp-relu | 0.569 | 166.5 | 1.000 |
| 22f+raw-mlp-sigmoid | 0.526 | 115.6 | 0.800 |
| 22f-mlp-sigmoid | 0.463 | 176.2 | 1.000 |

**Key takeaways:**
- **SVM RBF dominates** — all 4 SVM configs top the table. SVM with 400+ dims
  spreads probability scores effectively on this dataset.
- **16f (no engagement features) > 22f** — removing log_points, log_comments,
  comment_ratio, comments_count, story_age, domain_recency helps.
- **Raw embeddings + sparse metadata is optimal** — 16f+raw-svm-rbf wins.
- **MLP-sigmoid is consistently worst** — sigmoid degrades NDCG@30 by 18-22%.
- **MLP-alpha (L2=0.01) helps** — outperforms default MLP (alpha=0.0001) in all
  feature configs, but still behind SVM.
- **No plateau on any config** — the MLP ReLU dead zone was only a production
  dashboard issue (imbalanced training set at deployment).

**Actionable insight:** Switch `hn_rerank.toml` to `model_type="svm"` with
`raw_embedding_features=true` and the 16-feature metadata set.

### Run F — Single-feature CV on winner (16f+raw-svm-rbf)

Last run: `uv run python scripts/feature_ablation.py tests/snapshots/baseline_full.json --feedback .cache/user_feedback/dashboard_feedback.json --cv 5 --single-features --no-drop-one --model-type svm --raw-embedding-features --features ...`

5-fold CV with walk-forward chronological folds. Each cell = one metadata feature
+ 384 raw embedding dims (400 total). All cells hit hit@30=1.000.

| Feature Only | NDCG@30 | mean_rank | P@5 |
|-------------|---------|-----------|-----|
| source_trust | **0.568** | **89.1** | 0.760 |
| closest_pos | 0.539 | 104.3 | 0.560 |
| closest_margin | 0.495 | 117.9 | 0.560 |
| centroid | 0.490 | 109.7 | 0.520 |
| pos_neg_ratio | 0.490 | 116.3 | 0.560 |
| pos_knn | 0.488 | 114.9 | 0.520 |
| is_pdf | 0.480 | 117.1 | 0.480 |
| cluster_size | 0.476 | 116.9 | 0.480 |
| embedding_magnitude | 0.475 | 117.4 | 0.480 |
| neg_knn | 0.474 | 116.8 | 0.480 |
| local_density | 0.474 | 117.0 | 0.480 |
| is_github | 0.473 | 116.3 | 0.480 |
| title_len | 0.458 | 117.6 | 0.480 |
| is_hn | 0.429 | 129.9 | 0.480 |
| closest_neg | 0.386 | 149.5 | 0.360 |
| text_len | 0.366 | 164.4 | 0.480 |
| **Baseline** (all 16 + raw) | 0.442 | 145.0 | 0.600 |

**Key takeaways:**
- **source_trust dominates** — alone it achieves 0.568 NDCG@30, beating the
  combined model (0.442) by 28.5%.
- **Most single features outperform the combined model** — the 16 metadata
  features are redundant/interfering when paired with 384-d raw embeddings.
- **Weakest features:** text_len (0.366) and closest_neg (0.386).
- **All features pass the plateau test** — nonhn_at_0.5 ≤ 0.07 on every config.

**Actionable insight:** Consider further reducing metadata features to
[source_trust, closest_pos, closest_margin, centroid, pos_neg_ratio] — the
top-5 all beat the combined model.

The current `hn_rerank.toml` uses `model_type = "svm"` with
`raw_embedding_features = true`, `svm_c = 0.3`, and 16 metadata features
(switched 2026-06-16, C optimized from 1.0→0.3 in Run G).

### Run G — SVM C × gamma tuning on 16f+raw-svm-rbf

Last run: `uv run python scripts/run_svm_tuning.py`

Grid search over C ∈ {0.3, 1.0, 3.0, 10.0} × gamma ∈ {scale, 0.001, 0.01, 0.1}
on the 16-feature metadata set + 384-d raw embeddings (400 total dims).
Time-split eval (Run D protocol). All cells hit MRR=1.000, hit@30=1.000.

| Cell | NDCG@30 | mean_rank | Note |
|------|---------|-----------|------|
| **C=0.3, gamma=scale** | **0.876** | 91.1 | **winner** |
| C=0.3, gamma=scale (5-fold CV) | 0.526 | 117.9 | CV confirmation |
| C=1.0, gamma=scale (previous winner) | 0.737 | 116.6 | baseline |
| C=1.0, gamma=scale (5-fold CV) | 0.442 | 145.0 | CV baseline |
| C=0.3, gamma=0.01 | 0.761 | 100.3 | |
| C=1.0, gamma=0.01 | 0.761 | 100.3 | |
| C=3.0, gamma=scale | 0.712 | 123.6 | |
| C=0.3, gamma=0.001 | 0.710 | 116.0 | |
| C=10.0, gamma=scale | 0.690 | 127.0 | |
| C=*, gamma=0.1 | 0.666 | 239.8 | plateau (nonhn_0.5=1.00) |

**Kernel variants (C=0.3, gamma=scale):** all 4 kernels (rbf, linear, poly, sigmoid)
achieved NDCG@30=0.876. Kernel choice is irrelevant at C=0.3 — regularization
dominates over kernel geometry in 400-d space.

**5-feature subset** [source_trust, closest_pos, closest_margin, centroid,
pos_neg_ratio] + raw (C=0.3): **NDCG@30=0.822**, mean_rank=59.1. ~6% worse than
16-feature on NDCG@30, but 35% better mean_rank. Not adopted — NDCG@30 is primary.

**Run G takeaways:**
- **C=0.3 is optimal** — lower C (more regularization) outperforms C=1.0 by
  19% (time-split: 0.876 vs 0.737; CV: 0.526 vs 0.442).
- **gamma=scale is optimal** — higher gamma (0.01) converges to 0.761 regardless
  of C; gamma=0.1 causes full plateau.
- **C=0.3, gamma=scale is now in hn_rerank.toml** — replaces old C=1.0.
- **5-feature subset not adopted** — 16 features are worth the extra dims for
  top-30 ranking even though mean_rank is worse.

### Run H — MLP + RF tuning (3-fold CV, content-based)

Last run: `uv run python scripts/run_mlp_rf_tuning.py`

27 configs across MLP alpha, hidden layers, solver, lr_init and RF params.
3-fold content-based CV — no temporal ordering, all stories used for
train/test. All cells hit hit@30=1.000, no plateau.

| Cell | NDCG@30 | MRR | mean_rank | P@5 |
|------|---------|-----|-----------|-----|
| **MLP lbfgs (64,) α=0.1** | **0.516** | **1.000** | **124.8** | 0.467 |
| MLP on 16f+raw (same config) | 0.516 | 1.000 | 124.8 | 0.467 |
| MLP adam (64,) α=0.1 | 0.378 | 0.472 | 134.3 | 0.267 |
| RF max_depth=5, min_samples_leaf=5 | 0.428 | 0.722 | 166.4 | 0.400 |
| RF baseline (unlimited depth) | 0.405 | 0.583 | 171.9 | 0.533 |
| **SVM C=0.3 (16f±raw, CV=3)** | **0.556** | 0.556 | 108.3 | 0.600 |

**Key findings:**
- **lbfgs dominates adam** for MLP — switching adam→lbfgs gives +36%
  (0.378→0.516)
- **Single layer (64,) > two-layer (64,32)** — simpler architecture wins on
  16 features
- **Higher L2 (α=0.1) is best** — controls overfitting on 639 samples
- **Raw embeddings don't help MLP or RF on CV=3** — unlike time-split where
  raw boosts SVM
- **MLP lbfgs beats RF** (0.516 vs 0.428) but still trails SVM (0.556)
- **SVM C=0.3 stays optimal** — gap narrows on CV=3 (+7.7%) vs time-split (+54%)

**Config changes made:**
- Added `mlp_hidden_layers: str = "64,32"` to `SingleModelConfig`
  (comma-separated layer sizes)
- Added `mlp_solver: str = "adam"` — supports "adam", "lbfgs", "sgd"
- Added `mlp_learning_rate_init: float = 0.001`
- Fixed `_convert_override_value` str bug in `scripts/feature_ablation.py`
  — numeric strings (e.g. "32") no longer converted to int when field type is str
- Wired all new MLP fields into `api/ordinal_model.py:_make_pipeline()`
- Results: `results/mlp_rf_tuning_results.json`

### Snapshot protocol

- `tests/snapshots/baseline.json` (48 train / 12 test / 780 candidates) — the
  legacy small eval. Time-split 80/20 of an older feedback snapshot.
- `tests/snapshots/baseline_full.json` (~639 train / ~160 test / ~481 candidates) —
  regenerated from all cached feedback. `test_ids` populated and test stories
  injected into candidates (see `build_dataset_from_feedback` in
  `evaluate_quality.py`).
- Regen the full snapshot: `uv run scripts/regen_full_snapshot.py` (no network).

### Operational notes

- **Harness must always be run with `--feedback .cache/user_feedback/dashboard_feedback.json`** for representative `story_age` and `domain_recency` measurements.
- `evaluate_cv` uses **content-based k-fold CV**, not temporal folds.
- `--cv N` uses content-based cross-validation.
- Without `--cv`, uses the stored time-split (Run D protocol — most realistic).
- `feature_ablation.py:run_one()` now reads `model_type` from `hn_rerank.toml` (`AppConfig.load()`), not the hardcoded default. Changes in TOML are reflected in ablation runs.

## Repo notes

- `hn_rerank.toml` controls most runtime behavior.
- `README.md` is the canonical user-facing setup reference.
- `public/index.html` and `public/clusters.html` are generated artifacts.
- Tests marked `@pytest.mark.slow` (6 tests: integration, pagination, SVM, clustering stability) are skipped with `-m "not slow"`.
- `api/rerank.py` thread-safe per-call caches via `_RankCache` (threading.local): `_story_age_at_vote_map` and `_domain_recency_map` are module globals set externally; `local_density` and `cluster_size` are per-`rank_stories` call. The story_age and domain_recency maps are populated from FeedbackRecord in `generate_html.py:1955-1975` and cleared in a `finally` block.
- `scripts/feature_ablation.py` accepts `--feedback path` to populate both `story_age` and `domain_recency` caches during ablation. Always use `--feedback .cache/user_feedback/dashboard_feedback.json` for representative measurements.
- `tests/snapshots/baseline_full.json` (~639 train / ~160 test / ~481 candidates) is generated from all cached feedback via `uv run scripts/regen_full_snapshot.py`. No network required. `tests/snapshots/baseline.json` is kept for fast unit-test evals.
- `scripts/feature_ablation.py` accepts `--cv N` to use content-based k-fold cross-validation via `evaluate_cv`. Without `--cv`, it uses a single 80/20 time-split.
- `scripts/feature_ablation.py:_convert_override_value()` now handles `str` fields correctly — prevents numeric strings like "32" from being converted to int when they should remain strings (e.g. `mlp_hidden_layers`).
- `api/config.py:SingleModelConfig` exposes `mlp_hidden_layers` (comma-separated str), `mlp_solver`, `mlp_learning_rate_init` for tuning.

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

### Run I — Content-based Re-evaluation (CV=3)

Last run: `bash scripts/run_4x4_ablation.sh`, `uv run python scripts/run_svm_tuning.py`, `uv run python scripts/run_mlp_rf_tuning.py`
Eval: 3-fold cross validation (content-based, chronological boundaries removed). Fixed `raw_embedding_features` flag parsing bug where the default TOML value was bleeding into `--no-raw` tests.

**4x4 Ablation highlights:**
- **Raw embeddings are essential:** `16f+raw-svm-rbf` (0.935) vs `16f-svm-rbf` (0.407). Raw embeddings provide massive lift.
- **Engagement features help under CV:** `22f` (which includes log_points, story_age) outscores `16f` under CV (0.992 vs ~0.935). However, to prioritize true "story content" prediction over temporal engagement leakage, the `16f+raw` set is strongly preferred.

**Tuning winners (16f + raw embeddings):**
- **MLP Winner:** `mlp_solver="lbfgs"`, `mlp_alpha=0.1`, `mlp_hidden_layers="32"`, `mlp_learning_rate_init=0.01` achieved **NDCG@30=0.992**, MRR=1.000, MeanRank=212.3, nonhn@0.5=0.00.
- **SVM Winner:** `svm_c=0.3`, `svm_gamma="scale"` achieved **NDCG@30=0.935**, MRR=1.000, MeanRank=192.4, nonhn@0.5=0.00.

**Actionable insight:** Switched `hn_rerank.toml` to the tuned MLP model (`model_type="mlp"`, `mlp_solver="lbfgs"`, `mlp_alpha=0.1`, `mlp_hidden_layers="32"`, `mlp_learning_rate_init=0.01`) keeping the content-focused `16-feature` set and `raw_embedding_features=true`.

### Run J — CV=3 Metric Saturation Discovery

The `NDCG@30` metric in the `CV=3` ablation runs was discovered to be artificially saturated.
In 3-fold CV on ~800 total positive stories, each test fold contains ~266 positive items mixed into a candidate pool of ~481 neutral items.
Because `NDCG@30` only evaluates the top 30 slots, a model only needs to identify the 30 "easiest" positive stories out of 266 (a ~11% recall rate) to achieve a perfect `1.000` score.
Highly expressive models like MLP (with raw embeddings) easily isolate 30 highly confident matches that align with training clusters, gaming the `NDCG@30` metric.

When evaluating the robust **Mean Rank** across all test positives (lower is better):
- `16f+raw-svm-rbf` (C=0.3): Mean Rank = **192.4**
- `16f+raw-mlp` (LBFGS, α=0.1): Mean Rank = **212.3**

**Actionable Insight:** SVM RBF remains the globally superior model for ranking the entire positive distribution, outperforming MLP by ~20 positions on average. `hn_rerank.toml` has been reverted back to the `svm` architecture.

### Run K — Evaluation Pipeline Audit and Bug Fix

An audit of the evaluation pipeline revealed a bug in `compute_classifier_similarity_features` during training: `np.fill_diagonal(sim_n, -1.0)` was incorrectly masking positive story rows instead of negative story rows due to row offsets when `embs` was vertically stacked `[pos; neg]`. This gave negative training stories an artificially inflated `closest_neg ≈ 1.0`, creating an optimistic bias.

After replacing `fill_diagonal` with an offset-aware `_mask_self_similarity` function, the SVM `16f+raw` model showed global improvement:
- **Mean Rank**: improved from 192.4 → **184.5**
- **Median Rank**: improved from 150.0 → **145.0**

The bug fix removed the optimistic bias, allowing the SVM to learn a better decision boundary.

