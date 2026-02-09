#!/usr/bin/env python3
"""Collect best params from prior Optuna runs and emit enqueue candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_run_jsons(root: Path):
    for p in root.rglob("optuna_*.json"):
        if p.is_file():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/final_enqueue_candidates.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows: list[tuple[float, dict[str, float], str]] = []

    for fp in _iter_run_jsons(runs_root):
        try:
            payload = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("cv_folds") != args.cv_folds:
            continue
        best_score = payload.get("best_score")
        best_params = payload.get("best_params")
        if not isinstance(best_score, (int, float)):
            continue
        if not isinstance(best_params, dict):
            continue
        parsed: dict[str, float] = {}
        for k, v in best_params.items():
            try:
                parsed[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        if not parsed:
            continue
        rows.append((float(best_score), parsed, str(fp)))

    rows.sort(key=lambda x: x[0], reverse=True)

    # Dedup by exact params dict; keep highest-score instance.
    dedup: dict[str, tuple[float, dict[str, float], str]] = {}
    for score, params, source in rows:
        sig = json.dumps(params, sort_keys=True)
        if sig not in dedup:
            dedup[sig] = (score, params, source)

    top = sorted(dedup.values(), key=lambda x: x[0], reverse=True)[: args.top_k]
    candidates = [params for _, params, _ in top]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidates, indent=2))

    summary_path = out_path.with_suffix(".meta.json")
    summary_path.write_text(
        json.dumps(
            {
                "count": len(candidates),
                "top_k": args.top_k,
                "cv_folds": args.cv_folds,
                "sources": [
                    {"best_score": score, "source": source}
                    for score, _, source in top
                ],
            },
            indent=2,
        )
    )

    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
