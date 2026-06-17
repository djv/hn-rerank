#!/usr/bin/env -S uv run
"""MLP + Random Forest tuning: alpha, hidden layers, solver, lr_init, RF params."""

from tuning_runner import OUTDIR, F16_RAW, run_eval
import json

SNAPSHOT = "tests/snapshots/baseline_full.json"
FEEDBACK = ".cache/user_feedback/dashboard_feedback.json"

MLP_BASE = [
    "uv",
    "run",
    "python",
    "scripts/feature_ablation.py",
    SNAPSHOT,
    "--feedback",
    FEEDBACK,
    "--no-drop-one",
    "--no-single-features",
    "--cv",
    "3",
]


def run_mlp(label, features, overrides=None, raw=True):
    base = MLP_BASE + ["--model-type", "mlp"]
    if raw:
        base += ["--raw-embedding-features"]
    else:
        base += ["--no-raw-embedding-features"]
    return run_eval(label, features, base, overrides)


def run_rf(label, features, overrides=None, raw=True):
    base = MLP_BASE + ["--model-type", "random_forest"]
    if raw:
        base += ["--raw-embedding-features"]
    else:
        base += ["--no-raw-embedding-features"]
    return run_eval(label, features, base, overrides)


# ── Phase 1: MLP alpha sweep (16f+raw, relu, (64,32), adam) ────────────
print("=== Phase 1: MLP alpha sweep (16f+raw, relu, (64,32), adam) ===", flush=True)
alpha_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
results = []
for a in alpha_values:
    r = run_mlp(f"mlp16f_a{a}", F16_RAW, {"mlp_alpha": str(a)})
    r["mlp_alpha"] = a
    r["mlp_hidden_layers"] = "64,32"
    results.append(r)

results.sort(key=lambda r: -r["aggregate_score"])
best_alpha = results[0]["mlp_alpha"]
print(
    f"  BEST alpha: {best_alpha} -> NDCG@30={results[0]['ndcg_at_30']:.3f}\n",
    flush=True,
)

# ── Phase 2: MLP hidden layer sweep (16f+raw, best alpha, relu, adam) ──
print(
    f"=== Phase 2: MLP hidden layer sweep (16f+raw, alpha={best_alpha}, relu, adam) ===",
    flush=True,
)
hl_values = ["32", "64", "128", "32,16", "64,32", "128,64", "128,64,32"]
for hl in hl_values:
    label = "mlp16f_h" + hl.replace(",", "_")
    r = run_mlp(label, F16_RAW, {"mlp_alpha": str(best_alpha), "mlp_hidden_layers": hl})
    r["mlp_alpha"] = best_alpha
    r["mlp_hidden_layers"] = hl
    results.append(r)

results.sort(key=lambda r: -r["aggregate_score"])
best_hl = results[0]["mlp_hidden_layers"]
print(
    f"  BEST hidden_layers: {best_hl} -> NDCG@30={results[0]['ndcg_at_30']:.3f}\n",
    flush=True,
)

# ── Phase 3: MLP solver test (16f+raw, best alpha, best hl, relu) ──────
print(
    f"=== Phase 3: MLP solver test (16f+raw, alpha={best_alpha}, hl={best_hl}, relu) ===",
    flush=True,
)
for solver in ["adam", "lbfgs"]:
    label = f"mlp16f_s{solver}"
    r = run_mlp(
        label,
        F16_RAW,
        {
            "mlp_alpha": str(best_alpha),
            "mlp_hidden_layers": best_hl,
            "mlp_solver": solver,
        },
    )
    r["mlp_alpha"] = best_alpha
    r["mlp_hidden_layers"] = best_hl
    r["mlp_solver"] = solver
    results.append(r)

results.sort(key=lambda r: -r["aggregate_score"])
print(
    f"  BEST config so far: {results[0]['name']} -> NDCG@30={results[0]['ndcg_at_30']:.3f}\n",
    flush=True,
)

# ── Phase 4: MLP learning_rate_init sweep (16f+raw, best alpha, best hl, best solver) ──
# Pick the best solver from Phase 3 results
solver_results = [r for r in results if r.get("mlp_solver")]
best_solver = (
    max(solver_results, key=lambda r: r["aggregate_score"]).get("mlp_solver", "adam")
    if solver_results
    else "adam"
)

print(
    f"=== Phase 4: MLP lr_init sweep (16f+raw, alpha={best_alpha}, hl={best_hl}, {best_solver}) ===",
    flush=True,
)
for lr in [0.0001, 0.001, 0.01]:
    label = f"mlp16f_lr{lr}"
    r = run_mlp(
        label,
        F16_RAW,
        {
            "mlp_alpha": str(best_alpha),
            "mlp_hidden_layers": best_hl,
            "mlp_solver": best_solver,
            "mlp_learning_rate_init": str(lr),
        },
    )
    r["mlp_alpha"] = best_alpha
    r["mlp_hidden_layers"] = best_hl
    r["mlp_solver"] = best_solver
    r["mlp_lr_init"] = lr
    results.append(r)

results.sort(key=lambda r: -r["aggregate_score"])
print(
    f"  BEST MLP on 16f+raw: {results[0]['name']} -> NDCG@30={results[0]['ndcg_at_30']:.3f}\n",
    flush=True,
)

# ── Phase 5: MLP winner recap ──────────────────────────────────
# Best MLP config on 16f (no raw). Test it on 16f+raw (400 dims).
best_mlp = results[0]
print(
    f"=== Phase 5: MLP winner recap (alpha={best_mlp.get('mlp_alpha', '?')}, hl={best_mlp.get('mlp_hidden_layers', '?')}) ===",
    flush=True,
)
ov = {}
if best_mlp.get("mlp_alpha") is not None:
    ov["mlp_alpha"] = str(best_mlp["mlp_alpha"])
if best_mlp.get("mlp_hidden_layers") is not None:
    ov["mlp_hidden_layers"] = str(best_mlp["mlp_hidden_layers"])
if best_mlp.get("mlp_solver") is not None:
    ov["mlp_solver"] = str(best_mlp["mlp_solver"])
if best_mlp.get("mlp_lr_init") is not None:
    ov["mlp_learning_rate_init"] = str(best_mlp["mlp_lr_init"])
r = run_mlp("mlp_16f+raw_winner", F16_RAW, ov, raw=True)
r["mlp_alpha"] = best_mlp.get("mlp_alpha")
r["mlp_hidden_layers"] = best_mlp.get("mlp_hidden_layers")
r["mlp_solver"] = best_mlp.get("mlp_solver")
r["mlp_lr_init"] = best_mlp.get("mlp_lr_init")
results.append(r)

# ── Phase 6: RF sweep (16f+raw, no raw) ─────────────────────────────────
print("\n=== Phase 6: RF sweep (16f+raw, no raw) ===", flush=True)
rf_configs = [
    (
        "rf16f_baseline",
        {"rf_n_estimators": "200", "rf_max_depth": "0", "rf_min_samples_leaf": "2"},
    ),
    (
        "rf16f_md10",
        {"rf_n_estimators": "200", "rf_max_depth": "10", "rf_min_samples_leaf": "2"},
    ),
    (
        "rf16f_md5_ml5",
        {"rf_n_estimators": "200", "rf_max_depth": "5", "rf_min_samples_leaf": "5"},
    ),
    (
        "rf16f_md20",
        {"rf_n_estimators": "200", "rf_max_depth": "20", "rf_min_samples_leaf": "2"},
    ),
    (
        "rf16f_n500_md10",
        {"rf_n_estimators": "500", "rf_max_depth": "10", "rf_min_samples_leaf": "2"},
    ),
    (
        "rf16f_n500",
        {"rf_n_estimators": "500", "rf_max_depth": "0", "rf_min_samples_leaf": "2"},
    ),
    (
        "rf16f_mdNone_ml10",
        {"rf_n_estimators": "200", "rf_max_depth": "0", "rf_min_samples_leaf": "10"},
    ),
]
for label, ov in rf_configs:
    r = run_rf(label, F16_RAW, ov)
    r["rf_overrides"] = ov
    results.append(r)

results.sort(key=lambda r: -r["aggregate_score"])

# ── Phase 7: Best RF on 16f+raw (sanity check) ──────────────────────
rf_best = max(
    (r for r in results if r.get("rf_overrides")), key=lambda r: r["aggregate_score"]
)
print(f"\n=== Phase 7: RF winner recap (from {rf_best['name']}) ===", flush=True)
r = run_rf("rf_16f+raw_winner", F16_RAW, rf_best["rf_overrides"], raw=True)
r["rf_overrides"] = rf_best["rf_overrides"]
results.append(r)

# ── Final table ──────────────────────────────────────────────────────
results.sort(key=lambda r: -r["aggregate_score"])
print("\n=== Final results ===", flush=True)
print(
    f"{'name':<35} {'NDCG@30':>7} {'MRR':>5} {'mean_rank':>7} {'P@5':>5} {'nonhn@0.5':>9}"
)
print("-" * 66)
for r in results:
    name = r.get("name", "?")[:34]
    nonhn = r.get("nonhn_at_0_5_fraction", -1)
    print(
        f"{name:<35} {r['ndcg_at_30']:>7.3f} {r['mrr']:>5.3f} "
        f"{r['mean_rank']:>7.1f} {r['precision_at_5']:>5.3f} "
        f"{nonhn:>9.2f}"
    )

winner = results[0]
print(
    f"\n>>> WINNER: {winner['name']} "
    f"(NDCG@30={winner['ndcg_at_30']:.3f}, "
    f"mean_rank={winner['mean_rank']:.1f})",
    flush=True,
)

# Save consolidated
with open(OUTDIR / "mlp_rf_tuning_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUTDIR / 'mlp_rf_tuning_results.json'}")
