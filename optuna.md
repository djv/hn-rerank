# Optuna Hyperparameter Optimization Plan

## Baseline

| Metric | Value | Source |
|--------|-------|--------|
| Best score (cv=3) | 0.5069 | `runs/optuna_auto/20260206_075555_seed42/optuna_20260206_075607.json` |
| Previous best (cv=5) | 0.3676 | `runs/optuna_auto/smoke3/optuna_20260205_231157.json` |

**Caveat:** The 0.5069 score used cv=3. All remaining runs use cv=5 for stronger robustness. Compare only within the same CV setting.

## Execution Principles

- Always run with: `--cache-only --n-jobs 4 --candidates 500`
- Keep stage outputs isolated by directory.
- Compare only runs with the same CV setting (cv=5) when deciding promotion.
- Treat single-run spikes as suspect; rely on multi-seed consistency.

---

## Stage 3: Core Refinement (Primary)

**Purpose:** Exploit a lower-dimensional, high-impact parameter space for stable gains.

| Run | Space | Trials | CV | Seed | Dir |
|-----|-------|--------|----|------|-----|
| 3a  | core  | 300    | 5  | 60   | `runs/stage3_core_refine_seed60/` |
| 3b  | core  | 300    | 5  | 61   | `runs/stage3_core_refine_seed61/` |

**Success criteria:**
- At least one run improves the best cv=5 score.
- Best params are not wildly divergent across seeds.

**Status:** Running (task IDs: `b6d3051`, `b68b00c`)

---

## Stage 4: Full Expansion (Validation of Added Complexity)

**Purpose:** Test whether broader/full space yields real incremental value over core.

| Run | Space | Trials | CV | Seed | Dir |
|-----|-------|--------|----|------|-----|
| 4a  | full  | 250    | 5  | 70   | `runs/stage4_full_expand_seed70/` |
| 4b  | full  | 250    | 5  | 71   | `runs/stage4_full_expand_seed71/` |

**Success criteria:**
- Full-space best exceeds Stage 3 best by a meaningful margin and is not a one-off.
- If no clear gain, keep core-space winner (simpler and lower variance).

**Status:** Running (task IDs: `b936948`, `bc6ddc2`)

---

## Decision Gate (After Stages 3+4)

Choose final candidate by:

1. **Highest `best_score`** under cv=5.
2. **Consistency** across seeds (both seeds should converge to similar regions).
3. **Practical simplicity** — prefer core if full gain is marginal.

---

## Promotion Workflow

1. Apply winning params to `api/constants.py`.
2. Validate:
   ```bash
   uv run pytest -q
   uv run ruff check .
   ```
3. Save artifacts (winner log/json + brief notes) in run folder for traceability.
4. Commit once checks pass.

---

## Operational Notes

- `n_jobs=4` parallelism supported by tokenizer/thread-safety fixes already applied.
- Background runners may terminate unexpectedly; monitor with `tail -f` on output files.
- Warm-start: all runs seeded from `runs/optuna_auto/20260206_075555_seed42/optuna_20260206_075607.{json,log}`.
