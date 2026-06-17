#!/usr/bin/env -S uv run
import subprocess
import json
import time
from pathlib import Path

OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

F16_RAW = (
    "centroid,pos_knn,neg_knn,pos_neg_ratio,"
    "title_len,text_len,is_github,is_pdf,"
    "closest_pos,closest_neg,closest_margin,"
    "is_hn,source_trust,local_density,cluster_size,"
    "embedding_magnitude"
)

BASE = [
    "uv", "run", "python", "scripts/feature_ablation.py",
    "tests/snapshots/baseline_full.json",
    "--feedback", ".cache/user_feedback/dashboard_feedback.json",
    "--no-drop-one", "--no-single-features", "--cv", "3",
    "--model-type", "svm", "--raw-embedding-features",
]

def run_eval(label, hn_count):
    cmd = BASE + [
        "--output", str(OUTDIR / f"{label}.json"),
        "--features", f"{label}={F16_RAW}",
        "--override", "svm_c=0.3",
        "--override", "svm_gamma=scale",
        "--override", f"hard_negative_mining_count={hn_count}"
    ]
    print(f"[{label}] hn_count={hn_count}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    
    with open(OUTDIR / f"{label}.json") as f:
        data = json.load(f)
    r = data[0]
    r['aggregate_score'] = (r.get('recall_at_50', 0) * 1000) - r['median_rank']
    print(
        f"  -> Median={r['median_rank']:.1f} mean={r['mean_rank']:.1f} "
        f"Recall@50={r.get('recall_at_50', 0):.3f} Agg={r['aggregate_score']:.1f} ({elapsed:.1f}s)",
        flush=True,
    )
    return r

results = []
for count in [0, 10, 20, 50]:
    results.append(run_eval(f"hn_mining_{count}", count))

print("\n=== Results ===")
results.sort(key=lambda r: -r["aggregate_score"])
for r in results:
    print(f"{r['name']:<15} Agg={r['aggregate_score']:>5.1f} Median={r['median_rank']:>5.1f} Recall@50={r.get('recall_at_50', 0):.3f}")
