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
F5_RAW = "source_trust,closest_pos,closest_margin,centroid,pos_neg_ratio"

def run_eval(
    label: str,
    features: str,
    base_cmd: list[str],
    overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    cmd = base_cmd + [
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
    r["aggregate_score"] = (r.get("recall_at_50", 0) * 1000) - r["median_rank"]
    print(
        f"  -> Median={r.get('median_rank', 0):.1f} mean={r.get('mean_rank', 0):.1f} "
        f"Recall@50={r.get('recall_at_50', 0):.3f} Agg={r['aggregate_score']:.1f} ({elapsed:.1f}s)",
        flush=True,
    )
    return r
