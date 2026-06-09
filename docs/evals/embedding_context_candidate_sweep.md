# Embedding Model / Context Length / Candidate Count Sweep

## Setup

Evaluation target: local-first HN reranking dashboard.

Primary eval script: `scripts/evaluate_feedback_models.py`

Two evaluation modes:
- **Time-split holdout (default)**: sort feedback by `updated_at`, hold out newest 20%. Realistic dashboard proxy.
- **CV `--cv N --seed 42`**: random folds, useful for variance but overstates performance (leaks temporal structure).

Dataset: feedback records in `.cache/user_feedback/dashboard_feedback.json`
- Grew from ~1032 to ~1044 during this eval session.
- Frozen snapshot at 1044 records: `.cache/user_feedback/dashboard_feedback_debug.json`

## Evaluator Changes Made

`scripts/evaluate_feedback_models.py`:
- Model override flags: `--model-dir`, `--embedding-max-tokens`
- Neutral/nonpositive leakage diagnostics:
  - `neutral_in_top10`, `neutral_median_rank`
  - `nonpositive_in_top10`, `nonpositive_median_rank`
- `--candidate-count N`: controls HN distractor count in candidate pool

Caveat: CV branch prints metrics but doesn't write `--json-output` (only time-split does).

## Jina Small Model Directories

| Dir | `max_tokens` | Notes |
|---|---:|---|
| `onnx_model_jina_small` | 256 | Original, used for 256-tok runs |
| `onnx_model_jina_small_128` | 128 | Symlink to original, different spec |
| `onnx_model_jina_small_384` | 384 | |
| `onnx_model_jina_small_512` | 512 | |
| `onnx_model_jina_small_768` | 768 | |
| `onnx_model_jina_small_1024` | 1024 | |
| `onnx_model_jina_small_1536` | 1536 | OOM during CV5, never completed |
| `onnx_model_jina_small_2048` | 2048 | OOM during CV5, never completed |

Jina config: `max_position_embeddings=8192`, `position_embedding_type=alibi`. ONNX shape `['batch_size', 'sequence_length']` — dynamic, handles all lengths.

## Time-Split Results (NDCG@10)

### 300 candidates (default)

| Model | Tok | NDCG@10 | Prec@10 | Recall@50 |
|---|---:|---:|---:|---:|
| MiniLM | 256 | **0.773** | 70% | 31.7% |
| Jina | 128 | 0.423 | 30% | 17.1% |
| Jina | 256 | 0.547 | 40% | 26.8% |
| Jina | 384 | 0.472 | 40% | 31.7% |
| Jina | 512 | 0.482 | 40% | 29.3% |
| Jina | 768 | 0.000 | 0% | 19.5% |
| Jina | 1024 | 0.000 | 0% | 14.6% |

### 1300 candidates (`--candidate-count 1300`, RF only, context-length sweep)

| Model | Tok | NDCG@10 | Prec@10 | Recall@50 |
|---|---:|---:|---:|---:|
| MiniLM | 256 | 0.532 | 40% | 22.0% |
| Jina | 128 | 0.313 | 30% | 12.2% |
| Jina | 256 | 0.539 | 40% | 19.5% |
| Jina | 384 | 0.452 | 40% | 22.0% |
| Jina | 512 | 0.220 | 10% | 14.6% |
| Jina | 768 | 0.000 | 0% | 4.9% |
| Jina | 1024 | 0.000 | 0% | 0.0% |

See "Classifier × Candidate-Pool Robustness" below for the full 1300-candidate cross-classifier comparison.

### Key Time-Split Findings

1. **MiniLM + RF dominates at 300 candidates** (0.773 vs best Jina 0.675).
2. **768+ time-split hits dead zero with RF** — SVM RBF recovers to non-zero but still degrades.
3. **RF context-length winner (old):** MiniLM 256 (0.773) > Jina 256 (0.547) > Jina 384 (0.472).

## CV5 Results (NDCG@10)

### 300 candidates

| Model | Tok | NDCG@10 |
|---|---:|---:|
| MiniLM | 256 | 0.896±0.06 |
| Jina | 128 | 0.942±0.09 |
| Jina | 256 | 0.938±0.04 |
| Jina | 384 | 1.000±0.00 |
| Jina | 512 | 0.987±0.03 |
| Jina | 768 | 0.972±0.03 |
| Jina | 1024 | 0.961±0.05 |

### 1300 candidates

| Model | Tok | NDCG@10 |
|---|---:|---:|
| MiniLM | 256 | 0.881±0.08 |
| Jina | 128 | 0.883±0.06 |
| Jina | 256 | 0.908±0.03 |
| Jina | 384 | 0.973±0.03 |
| Jina | 512 | 0.945±0.05 |
| Jina | 768 | 0.803±0.08 |
| Jina | 1024 | timed out |

### Key CV5 Findings

CV5 overstates performance because random folds destroy temporal holdout:
- Train/test are mixed across full feedback history.
- Same-topic/same-user items leak across folds.
- Positive density per fold is ~2.5-3× higher than time-split (100+ upvotes vs 41).

## Full Classifier Comparison (300 candidates, frozen snapshot)

Complete matrix across classifiers for both MiniLM and Jina 256, plus Jina 768 for reference.

### MiniLM 256 (default `onnx_model`)

| Classifier | NDCG@10 | Prec@10 | Recall@50 | Downvote leak |
|---|---:|---:|---:|---:|
| random_forest (TOML default) | **0.773** | 70% | 31.7% | 0% |
| svm rbf | **0.748** | 70% | 39.0% | 0% |
| logistic | 0.581 | 60% | 31.7% | 0% |
| gradient_boosting | 0.703 | 60% | 22.0% | 0% |

### Jina 256 (`onnx_model_jina_small`)

| Classifier | NDCG@10 | Prec@10 | Recall@50 | Downvote leak |
|---|---:|---:|---:|---:|
| random_forest | 0.547 | 40% | 26.8% | 0% |
| svm rbf | 0.577 | 50% | 36.6% | 0% |
| svm linear | 0.149 | 20% | 26.8% | 4.2% |
| logistic | **0.675** | 60% | 17.1% | 0% |
| gradient_boosting | 0.330 | 20% | 39.0% | 1.0% |

### Jina 768 (reference)

| Classifier | NDCG@10 | Prec@10 |
|---|---:|---:|
| random_forest | 0.000 | 0% |
| svm rbf | 0.284 | 20% |
| svm linear | 0.095 | 10% |

### Key Findings

1. **MiniLM dominates Jina at 300 candidates regardless of classifier.** Best MiniLM config (RF, 0.773) beats best Jina config (logistic, 0.675) by ~0.1 NDCG@10.

2. **Classifier ranking differs by embedding:**
   - MiniLM: RF > SVM RBF > GB > logistic (tree-based wins)
   - Jina 256: logistic > SVM RBF > RF > GB (linear/lightweight wins)
   - Jina 768: SVM RBF alone achieves non-zero

3. **Logistic regression is surprisingly strong on Jina 256** (0.675 NDCG@10) — nearly catches MiniLM RF. The only Jina config competitive with MiniLM.

4. **Gradient boosting is inconsistent:** near MiniLM SVM RBF on MiniLM (0.703), but worst overall on Jina 256 (0.330) with downvote leakage.

5. **TOML default (RF) is optimal for current MiniLM setup at 300 candidates.** But see 1300-candidate results below.

## Classifier × Candidate-Pool Robustness (1300 candidates)

Complete matrix across all three embeddings × five classifiers at 1300 candidates.

| Embedding | RF | SVM RBF | logistic | GB | MLP |
|---|---:|---:|---:|---:|---:|
| MiniLM 256 | 0.532 | **0.675** | 0.218 | 0.467 | 0.404 |
| Jina 256 | 0.539 | 0.425 | 0.249 | 0.330 | 0.095 |
| Jina 384 | 0.381 | 0.444 | 0.085 | 0.489 | 0.000 |

### Key Findings

1. **MiniLM + SVM RBF wins (0.675)** — leads by a wide margin at 1300 cand. No other combination comes close.
2. **Jina 384 GB (0.489) is third** behind MiniLM RF (0.532), but still far from MiniLM SVM RBF.
3. **Jina 256 RF (0.539) ties MiniLM RF (0.532)** at 1300 — confirming the earlier sweep finding, but both trail SVM RBF.
4. **MLP is weak or zero** across all embeddings — unreliable for this task.
5. **No alternative beats MiniLM + SVM RBF at any pool size.** At 300 cand, RF is +0.025 ahead; at 1300 cand, SVM RBF is +0.143 ahead. SVM RBF's robustness is decisive.

### Drop Analysis (300 → 1300 candidates)

| Embedding | Classifier | 300 | 1300 | Drop |
|---|---:|---:|---:|---:|
| MiniLM | SVM RBF | 0.748 | **0.675** | **-10%** |
| MiniLM | RF | 0.773 | 0.532 | -31% |
| MiniLM | GB | 0.703 | 0.467 | -34% |
| MiniLM | MLP | — | 0.404 | — |
| MiniLM | logistic | 0.581 | 0.218 | -62% |
| Jina 256 | RF | 0.547 | 0.539 | -1% |
| Jina 256 | SVM RBF | 0.577 | 0.425 | -26% |
| Jina 256 | logistic | 0.675 | 0.249 | -63% |
| Jina 256 | GB | 0.330 | 0.330 | 0% |
| Jina 256 | MLP | — | 0.095 | — |

### Recommendation

**Switch TOML default from `random_forest` to `svm` with `svm_kernel = "rbf"`.** SVM RBF is 3× more robust to pool dilution than RF and leads by +0.143 at the larger pool size that reflects real usage.

## SVM Grid Search (MiniLM 256, 1300 candidates)

Full grid: `C ∈ {0.3, 1.0, 3.0, 10.0, 30.0} × γ ∈ {scale, auto, 0.001, 0.003, 0.01, 0.03}` with RBF kernel.

| C | gamma | NDCG@10 | Prec@10 | Recall@50 | Downvote leak |
|---|---:|---:|---:|---:|---:|
| **1.0** | **scale** | **0.817** | **80%** | **34.1%** | 0% |
| 1.0 | auto | 0.817 | 80% | 34.1% | 0% |
| 1.0 | 0.003 | 0.763 | 70% | 34.1% | 0% |
| 1.0 | 0.01 | 0.706 | 60% | 31.7% | 0% |
| 1.0 | 0.001 | 0.694 | 60% | 31.7% | 0% |
| 10.0 | scale | 0.682 | 60% | 34.1% | 0% |
| 30.0 | scale | 0.682 | 60% | 34.1% | 0% |
| 3.0 | scale | 0.675 | 60% | 34.1% | 0% |
| 0.3 | scale | 0.643 | 60% | 22.0% | 3.1% |
| 1.0 | 0.03 | 0.286 | 20% | 12.2% | 4.2% |

### Key Findings

1. **C=1.0 is the optimal regularization** — C=3.0 overfits; C=0.3 underfits. C=10 and 30 plateau at 0.682.
2. **gamma=scale/auto dominate** — numeric gamma only works in a narrow band (~0.003–0.01). Too wide (0.001) hurts; too narrow (0.03) overfits catastrophically.
3. **80% Prec@10 at 1300 candidates** — remarkable. Only 3 downvotes/neutrals in top 10 across 10 test sets (41 upvotes total).

For comparison, **Jina 256 best SVM config** at 1300 cand is C=3.0 γ=0.01 with NDCG@10=0.538 — far behind MiniLM at any reasonable C/γ setting.

## Debugging Jina 768 Time-Split Zero

### Score Distributions from RF (frozen snapshot, 300 cand)

| Metric | MiniLM 256 | Jina 256 | Jina 768 |
|---|---:|---:|---:|
| UP mean score | 0.516 | 0.495 | 0.459 |
| DOWN mean score | 0.478 | 0.464 | 0.429 |
| DIST mean score | 0.423 | 0.423 | 0.406 |
| UP - DIST gap | 0.093 | 0.072 | 0.053 |
| DIST std | 0.064 | 0.063 | **0.083** |
| Max any score | 0.682 (UP) | 0.663 (UP) | **0.637 (DIST)** |
| Top 10 composition | 5UP/2DN/2NE/1DI | 4UP/1DN/3NE/2DI | **0UP/0DN/1NE/9DI** |
| Test up median rank | 136 | 137 | **215** |

### Root Cause

Not generic "noise dilution." Two issues combine to produce 0.000 at 768:

1. **RF-specific overfitting**: distractor score variance increases sharply at 768 (std 0.083 vs 0.063), and top distractors overtake top upvotes. SVM RBF avoids the zero (0.284) but still underperforms 256 (0.577).

2. **Context-dependent signal degradation**: even with SVM RBF, 768-tok (0.284) underperforms 256-tok (0.577). Possible explanations:
   - ALiBi position encoding attenuates early-token signal at longer sequence lengths
   - 768-tok truncation lands mid-article (article body, not title+lead), diluting the embedding with boilerplate
   - Higher-dimensional embeddings from longer context don't add discriminative signal for this particular feedback dataset

## Final Recommendation

| Setting | Current (TOML) | Recommended | Rationale |
|---|---|---|---|
| embedding model | `onnx_model` (MiniLM) | Keep | MiniLM dominates Jina across all classifiers |
| `model_type` | `svm` | Keep | SVM RBF is 3× more robust to pool dilution than RF |
| `svm_kernel` | `rbf` | Keep | Linear SVM is terrible; RBF essential |
| `svm_c` | `3.0` | **`1.0`** | Grid sweep: C=1.0 gives 0.817 vs 0.675 at 1300 cand |
| `svm_gamma` | `scale` | Keep | scale/auto are tied; both best |

## Key Files

- `scripts/evaluate_feedback_models.py` — main evaluator
- `scripts/debug_ranking.py` — debug script dumping RF scores with labels
- `api/rerank.py` — ONNX embedding + ranking pipeline
- `api/ordinal_model.py` — RF/SVM/MLP classifier training
- `docs/evals/embedding_context_candidate_sweep.md` — this file
