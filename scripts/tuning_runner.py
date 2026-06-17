import subprocess
import json
import time
from pathlib import Path

OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)


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
