#!/usr/bin/env -S uv run
"""Regenerate tests/snapshots/baseline_full.json from full feedback (no network).

Usage:
    uv run scripts/regen_full_snapshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from evaluate_quality import RankingEvaluator


def main() -> None:
    ev = RankingEvaluator(username="ablation_full")
    ok = ev.build_dataset_from_feedback(
        feedback_path=Path(".cache/user_feedback/dashboard_feedback.json"),
        snapshot_candidates_path=Path("tests/snapshots/baseline.json"),
        holdout=0.20,
    )
    if not ok:
        raise SystemExit("build_dataset_from_feedback failed")
    out = Path("tests/snapshots/baseline_full.json")
    ev.save_snapshot(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
