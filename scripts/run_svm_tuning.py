#!/usr/bin/env -S uv run
"""SVM tuning: C x gamma grid, kernel variants, 5-feature subset."""

import subprocess
import json
import time
from pathlib import Path

SNAPSHOT = "tests/snapshots/baseline_full.json"
FEEDBACK = ".cache/user_feedback/dashboard_feedback.json"
OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

F16_RAW = (
    "centroid,pos_knn,neg_knn,pos_neg_ratio,"
    "title_len,text_len,is_github,is_pdf,"
    "closest_pos,closest_neg,closest_margin,"
    "is_hn,source_trust,local_density,cluster_size,"
    "embedding_magnitude"
)
F5_RAW = "source_trust,closest_pos,closest_margin,centroid,pos_neg_ratio"

BASE = [
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
    "0",
    "--model-type",
    "svm",
    "--raw-embedding-features",
]


def run_eval(label, features, overrides=None):
    cmd = BASE + [
        "--output",
        str(OUTDIR / f"{label}.json"),
        "--features",
        f"{label}={features}",
    ]
    if overrides:
        for k, v in overrides.items():
            cmd += ["--override", f"{k}={v}"]
    print(f"  [{label}] overrides={overrides}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    with open(OUTDIR / f"{label}.json") as f:
        data = json.load(f)
    r = data[0]
    print(
        f"  -> NDCG@30={r['ndcg_at_30']:.3f} mean_rank={r['mean_rank']:.1f} "
        f"P@5={r['precision_at_5']:.3f} ({elapsed:.1f}s)",
        flush=True,
    )
    return r


# Phase 1: C x gamma grid
print("=== Phase 1: SVM C x gamma grid ===", flush=True)
c_values = [0.3, 1.0, 3.0, 10.0]
gamma_values = ["scale", 0.001, 0.01, 0.1]
grid_results = []

for c in c_values:
    for g in gamma_values:
        label = f"svm_c{c}_g{g}"
        overrides = {"svm_c": str(c)}
        if g != "scale":
            overrides["svm_gamma"] = str(g)
        r = run_eval(label, F16_RAW, overrides)
        r["svm_c"] = c
        r["svm_gamma"] = g
        grid_results.append(r)

grid_results.sort(key=lambda r: -r["ndcg_at_30"])
best = grid_results[0]
best_c = best["svm_c"]
best_g = best["svm_gamma"]
print(
    f"\n  BEST: C={best_c}, gamma={best_g} -> NDCG@30={best['ndcg_at_30']:.3f}\n",
    flush=True,
)

# Phase 2: Kernel variants with best C, gamma
print("=== Phase 2: SVM kernel variants ===", flush=True)
kernels = ["rbf", "linear", "poly", "sigmoid"]
kernel_results = []
for kernel in kernels:
    overrides = {"svm_kernel": kernel}
    if best_c != 1.0:
        overrides["svm_c"] = str(best_c)
    if best_g != "scale":
        overrides["svm_gamma"] = str(best_g)
    r = run_eval(f"kernel_{kernel}", F16_RAW, overrides)
    r["svm_kernel"] = kernel
    kernel_results.append(r)

# Phase 3: 5-feature subset with best C, gamma
print("=== Phase 3: 5-feature subset ===", flush=True)
overrides = {}
if best_c != 1.0:
    overrides["svm_c"] = str(best_c)
if best_g != "scale":
    overrides["svm_gamma"] = str(best_g)
r5 = run_eval("svm_5feat_raw", F5_RAW, overrides)

# Consolidate
all_cells = grid_results + kernel_results + [r5]
all_cells.sort(key=lambda r: -r["ndcg_at_30"])
winner_cell = all_cells[0]

print("\n=== Results table ===", flush=True)
print(f"{'name':<35} {'NDCG@30':>7} {'MRR':>5} {'mean_rank':>7} {'P@5':>5}")
print("-" * 59)
for r in all_cells:
    name = r["name"][:34]
    print(
        f"{name:<35} {r['ndcg_at_30']:>7.3f} {r['mrr']:>5.3f} "
        f"{r['mean_rank']:>7.1f} {r['precision_at_5']:>5.3f}"
    )

print(
    f"\n>>> WINNER: {winner_cell['name']} "
    f"(NDCG@30={winner_cell['ndcg_at_30']:.3f}, "
    f"mean_rank={winner_cell['mean_rank']:.1f})",
    flush=True,
)

# Save consolidated
with open(OUTDIR / "svm_tuning_results.json", "w") as f:
    json.dump(all_cells, f, indent=2, default=str)
print(f"\nConsolidated results saved to {OUTDIR / 'svm_tuning_results.json'}")
