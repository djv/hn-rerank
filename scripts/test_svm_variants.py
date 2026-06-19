"""Test SVM variants: C sweep, isotonic calibration, softmax margins."""

import json
import re
import subprocess
import sys
from pathlib import Path

CONFIG = Path("/home/dev/hn-rewrite/config.toml")
PIPELINE = Path("/home/dev/hn-rewrite/pipeline.py")
EVAL = Path("/home/dev/hn-rewrite/eval.py")
DASHBOARD = Path("/home/dev/hn-rewrite/public/index.html")
REPORT = Path("/home/dev/hn-rewrite/eval_report.json")

config_base = CONFIG.read_text()
pipeline_base = PIPELINE.read_text()
eval_base = EVAL.read_text()

trials = []
for kernel in ("linear", "rbf"):
    for c in (0.1, 1.0, 10.0):
        trials.append((f"{kernel}_C{c:g}_sig", kernel, c, "sigmoid", None))
    trials.append((f"{kernel}_C03_iso", kernel, 0.3, "isotonic", None))
    trials.append((f"{kernel}_softmax", kernel, 0.3, None, "softmax"))

results = {}

for label, kernel, c, method, margin_mode in trials:
    print(f"\n{'=' * 60}")
    print(f"TRIAL: {label}")
    print("=" * 60)

    # Patch config
    cfg_lines = []
    for line in config_base.splitlines():
        if line.strip().startswith("svm_kernel"):
            cfg_lines.append(f'svm_kernel = "{kernel}"')
        elif line.strip().startswith("svm_c"):
            cfg_lines.append(f"svm_c = {c}")
        elif line.strip().startswith("# svm_c"):
            cfg_lines.append(f"svm_c = {c}")
        elif line.strip().startswith("svm_gamma"):
            if kernel == "linear":
                cfg_lines.append("# svm_gamma = 0.1")
            else:
                cfg_lines.append("svm_gamma = 0.1")
        else:
            cfg_lines.append(line)
    CONFIG.write_text("\n".join(cfg_lines))

    # Patch pipeline.py + eval.py for calibration method
    if margin_mode == "softmax":
        # Replace calibration block with softmax in pipeline.py
        pipe_code = pipeline_base
        pipe_code = re.sub(
            r"n_train = len\(fb_features_scaled\)\n\s+calibrated = CalibratedClassifierCV\([^)]+\)\n\s+calibrated\.fit\(fb_features_scaled, labels, sample_weight=sample_weights\)",
            "",  # Remove entire calibration block
            pipe_code,
        )
        pipe_code = re.sub(
            r"probs = calibrated\.predict_proba\(cand_features_scaled\)\n\s+class_order = list\(calibrated\.classes_\)",
            "margins = svm.decision_function(cand_features_scaled)\nmargins_exp = np.exp(margins - margins.max(axis=1, keepdims=True))\nprobs = margins_exp / margins_exp.sum(axis=1, keepdims=True)\nclass_order = list(svm.classes_)",
            pipe_code,
        )

        eval_code = eval_base
        eval_code = re.sub(
            r"n_train = len\(X_train_scaled\)\n\s+calibrated = CalibratedClassifierCV\([^)]+\)\n\s+calibrated\.fit\(X_train_scaled, y_train, sample_weight=weights\)",
            "",
            eval_code,
        )
        eval_code = re.sub(
            r"probs = calibrated\.predict_proba\(X_all_scaled\)",
            "margins = svm.decision_function(X_all_scaled)\nmargins_exp = np.exp(margins - margins.max(axis=1, keepdims=True))\nprobs = margins_exp / margins_exp.sum(axis=1, keepdims=True)",
            eval_code,
        )

        PIPELINE.write_text(pipe_code)
        EVAL.write_text(eval_code)

    elif method != "sigmoid":
        # Change method="sigmoid" → method="isotonic"
        pipe_code = pipeline_base.replace('method="sigmoid"', f'method="{method}"')
        PIPELINE.write_text(pipe_code)
        eval_code = eval_base.replace('method="sigmoid"', f'method="{method}"')
        EVAL.write_text(eval_code)
    else:
        # Restore base files
        PIPELINE.write_text(pipeline_base)
        EVAL.write_text(eval_base)

    # Run generate.py
    print("\n--- generate.py ---")
    gen = subprocess.run(
        [sys.executable, "generate.py"],
        capture_output=True,
        text=True,
        timeout=240,
    )
    if gen.returncode != 0:
        print(f"ERR: {gen.stderr[-500:]}")
        results[label] = {"error": gen.stderr[-200:]}
        # Restore files for next trial
        PIPELINE.write_text(pipeline_base)
        EVAL.write_text(eval_base)
        continue

    # Extract scores
    scores = []
    if DASHBOARD.exists():
        dash = DASHBOARD.read_text()
        found = re.findall(r'data-score="([^"]*)"', dash)
        scores = [float(s) for s in found[:40]]
        print(
            f"Score spread (top 40): min={min(scores):.4f}  max={max(scores):.4f}  median={sorted(scores)[len(scores) // 2]:.4f}"
        )
        distinct = len(set(round(s, 3) for s in scores))
        print(f"Distinct tiers (0.1% resolution): {distinct}")

    # Run eval
    print("\n--- eval.py ---")
    eval_run = subprocess.run(
        [sys.executable, "eval.py"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if eval_run.returncode != 0:
        print(f"ERR: {eval_run.stderr[-300:]}")
        results[label] = {"error": eval_run.stderr[-200:]}
        PIPELINE.write_text(pipeline_base)
        EVAL.write_text(eval_base)
        continue

    # Parse metrics
    if REPORT.exists() and scores:
        r = json.loads(REPORT.read_text())
        soft = r["formulas"]["soft"]
        results[label] = {
            "soft_ndcg10": soft["mean"]["ndcg_at_10"],
            "soft_ndcg10_std": soft["std"]["ndcg_at_10"],
            "soft_p10": soft["mean"]["precision_at_10"],
            "soft_median_rank": soft["mean"]["median_rank"],
            "score_max": max(scores),
            "score_min": min(scores),
            "score_median": sorted(scores)[len(scores) // 2],
            "tiers": distinct,
        }
        print(
            f">>> soft NDCG@10: {results[label]['soft_ndcg10']:.4f} ± {results[label]['soft_ndcg10_std']:.4f}"
        )
        print(
            f">>> score spread: [{results[label]['score_min']:.4f}, {results[label]['score_max']:.4f}] median={results[label]['score_median']:.4f}"
        )

    # Restore files
    PIPELINE.write_text(pipeline_base)
    EVAL.write_text(eval_base)

# Restore config
CONFIG.write_text(config_base)

print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(
    f"{'Variant':20s} {'NDCG@10':>10s} {'P@10':>8s} {'med_rk':>8s} {'s_max':>8s} {'s_min':>8s} {'tiers':>6s}"
)
print("-" * 70)
for label, data in results.items():
    if "error" in data:
        print(f"{label:20s} ERROR: {data['error'][:40]}")
    else:
        print(
            f"{label:20s} {data['soft_ndcg10']:10.4f} {data['soft_p10']:8.3f} {data['soft_median_rank']:8.1f} {data['score_max']:8.4f} {data['score_min']:8.4f} {data['tiers']:6d}"
        )

# Cleanup temp files
for f in ["_test_kernels.py", "_diag_scores.py"]:
    Path(f).unlink(missing_ok=True)
