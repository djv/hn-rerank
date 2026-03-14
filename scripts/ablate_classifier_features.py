#!/usr/bin/env -S uv run
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import ExitStack
from pathlib import Path
from statistics import fmean, pstdev
from typing import cast
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.rerank  # noqa: E402
from api.constants import (  # noqa: E402
    CLASSIFIER_K_FEAT,
    CLASSIFIER_NEG_SAMPLE_WEIGHT,
    KNN_NEIGHBORS,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_NEGATIVE_WEIGHT,
)
from evaluate_quality import RankingEvaluator  # noqa: E402


ABLATIONS: list[tuple[str, dict[str, bool]]] = [
    (
        "full",
        {
            "CLASSIFIER_USE_CENTROID_FEATURE": True,
            "CLASSIFIER_USE_POS_KNN_FEATURE": True,
            "CLASSIFIER_USE_NEG_KNN_FEATURE": True,
        },
    ),
    (
        "no_centroid",
        {
            "CLASSIFIER_USE_CENTROID_FEATURE": False,
            "CLASSIFIER_USE_POS_KNN_FEATURE": True,
            "CLASSIFIER_USE_NEG_KNN_FEATURE": True,
        },
    ),
    (
        "no_pos_knn",
        {
            "CLASSIFIER_USE_CENTROID_FEATURE": True,
            "CLASSIFIER_USE_POS_KNN_FEATURE": False,
            "CLASSIFIER_USE_NEG_KNN_FEATURE": True,
        },
    ),
    (
        "no_neg_knn",
        {
            "CLASSIFIER_USE_CENTROID_FEATURE": True,
            "CLASSIFIER_USE_POS_KNN_FEATURE": True,
            "CLASSIFIER_USE_NEG_KNN_FEATURE": False,
        },
    ),
    (
        "embedding_only",
        {
            "CLASSIFIER_USE_CENTROID_FEATURE": False,
            "CLASSIFIER_USE_POS_KNN_FEATURE": False,
            "CLASSIFIER_USE_NEG_KNN_FEATURE": False,
        },
    ),
]

IMPORTANT_METRICS = [
    "mrr",
    "ndcg@10",
    "ndcg@30",
    "map@30",
    "precision@20",
    "recall@30",
    "recall@50",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run classifier feature ablations against a frozen benchmark snapshot."
    )
    parser.add_argument("snapshot", type=Path, help="Path to benchmark snapshot JSON")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--seed",
        dest="seeds",
        action="append",
        type=int,
        default=None,
        help="Repeatable RNG seeds (default: 0,1,2)",
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=RANKING_DIVERSITY_LAMBDA,
        help="Diversity lambda to evaluate",
    )
    parser.add_argument(
        "--neg-weight",
        type=float,
        default=RANKING_NEGATIVE_WEIGHT,
        help="Negative penalty weight to evaluate",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=KNN_NEIGHBORS,
        help="k-NN neighbor count to evaluate",
    )
    parser.add_argument(
        "--classifier-k-feat",
        type=int,
        default=CLASSIFIER_K_FEAT,
        help="Classifier k_feat to evaluate",
    )
    parser.add_argument(
        "--classifier-neg-sample-weight",
        type=float,
        default=CLASSIFIER_NEG_SAMPLE_WEIGHT,
        help="Classifier negative sample weight to evaluate",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser


def normalize_seeds(seeds: list[int] | None) -> list[int]:
    if not seeds:
        return [0, 1, 2]
    out: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        out.append(seed)
    return out


def summarize(metrics_per_seed: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not metrics_per_seed:
        return summary
    for key in metrics_per_seed[0]:
        values = [metrics[key] for metrics in metrics_per_seed]
        summary[key] = float(fmean(values))
        if len(values) > 1 and not key.endswith("_std"):
            summary[f"{key}_seed_std"] = float(pstdev(values))
    return summary


async def main() -> int:
    args = build_parser().parse_args()
    args.seeds = normalize_seeds(args.seeds)

    evaluator = RankingEvaluator("snapshot")
    if not evaluator.load_snapshot(args.snapshot):
        raise SystemExit(f"Failed to load snapshot: {args.snapshot}")

    results: list[dict[str, object]] = []
    for label, flags in ABLATIONS:
        per_seed: list[dict[str, float]] = []
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(api.rerank, "CLASSIFIER_K_FEAT", args.classifier_k_feat)
            )
            stack.enter_context(
                patch.object(
                    api.rerank,
                    "CLASSIFIER_NEG_SAMPLE_WEIGHT",
                    args.classifier_neg_sample_weight,
                )
            )
            for attr, value in flags.items():
                stack.enter_context(patch.object(api.rerank, attr, value))
            for seed in args.seeds:
                np.random.seed(seed)
                metrics = evaluator.evaluate_cv(
                    n_folds=args.cv_folds,
                    diversity=args.diversity,
                    knn=args.knn,
                    neg_weight=args.neg_weight,
                    use_classifier=True,
                    report_each=False,
                    parallel=False,
                )
                per_seed.append(metrics)
        summary = summarize(per_seed)
        results.append(
            {
                "label": label,
                "flags": flags,
                "summary": summary,
                "per_seed": [
                    {"seed": seed, "metrics": metrics}
                    for seed, metrics in zip(args.seeds, per_seed, strict=True)
                ],
            }
        )

    full_summary = next(
        cast(dict[str, float], item["summary"]) for item in results if item["label"] == "full"
    )
    for item in results:
        summary = cast(dict[str, float], item["summary"])
        deltas = {
            metric: summary.get(metric, 0.0) - full_summary.get(metric, 0.0)
            for metric in IMPORTANT_METRICS
        }
        item["delta_vs_full"] = deltas

    payload = {
        "snapshot": str(args.snapshot),
        "seeds": args.seeds,
        "cv_folds": args.cv_folds,
        "params": {
            "diversity": args.diversity,
            "neg_weight": args.neg_weight,
            "knn": args.knn,
            "classifier_k_feat": args.classifier_k_feat,
            "classifier_neg_sample_weight": args.classifier_neg_sample_weight,
        },
        "results": results,
    }

    for item in results:
        summary = cast(dict[str, float], item["summary"])
        print(f"[{item['label']}]")
        for metric in IMPORTANT_METRICS:
            value = summary.get(metric)
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        if item["label"] != "full":
            delta = cast(dict[str, float], item["delta_vs_full"])
            print(f"  delta ndcg@30: {delta['ndcg@30']:+.4f}")
            print(f"  delta recall@30: {delta['recall@30']:+.4f}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Wrote JSON results to {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
