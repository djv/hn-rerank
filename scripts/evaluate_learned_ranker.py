"""Evaluate learned-ranker feedback fit against the current hybrid scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from api.config import AppConfig
from api.feedback import load_feedback
from api.learned_ranker import build_labels_from_feedback, evaluate_labeled_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated learned-ranker report from dashboard feedback."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("hn_rerank.toml"),
        help="Path to hn_rerank TOML config.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for a JSON copy of the report.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Maximum number of stratified CV folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.load(toml_path=args.config)
    labels = build_labels_from_feedback(load_feedback())
    report = evaluate_labeled_order(
        labels,
        config.learned_ranker,
        max_folds=args.folds,
    )
    payload = asdict(report)

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
