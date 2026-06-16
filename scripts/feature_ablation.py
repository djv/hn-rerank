#!/usr/bin/env -S uv run
"""Run ranking evaluation with different feature sets to measure feature importance.

Usage:
    uv run python scripts/feature_ablation.py [snapshot_path]
        [--output results.json]
        [--baseline default]
        [--drop-one]           [default: true]
        [--single-features]    [default: false]
        [--features name ...]  [default: use baseline features from config]
        [--feedback path]     [populate story_age cache from feedback records]
        [--top-k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path (for module imports when run from any directory)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from api.config import AppConfig

from evaluate_quality import RankingEvaluator


@dataclass
class AblationResult:
    name: str
    features: list[str]
    mrr: float
    precision_at_5: float
    recall_at_5: float
    ndcg_at_30: float
    hit_at_30: float
    mean_rank: float
    median_rank: float
    n_train: int
    n_test: int
    n_candidates: int
    classifier_used: bool
    elapsed_seconds: float
    n_folds: int = 0
    nonhn_score_stddev: float = 0.0
    nonhn_at_0_5_fraction: float = 0.0


_DEFAULT_K = [5, 30]
_METRIC_KEYS = [
    "mrr",
    "precision@5",
    "recall@5",
    "ndcg@30",
    "hit@30",
    "mean_rank",
    "median_rank",
    "nonhn_score_stddev",
    "nonhn_at_0_5_fraction",
]


def _parse_override_args(
    raw: list[str] | None,
) -> dict[str, str] | None:
    if not raw:
        return None
    parsed: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        parsed[k.strip()] = v.strip()
    return parsed or None


def _convert_override_value(key: str, value: str, config_obj: object) -> Any:
    """Convert a CLI string value to the correct type based on field annotation."""
    field_type = type(getattr(config_obj, key))
    if field_type is float:
        return float(value)
    elif field_type is int:
        return int(value)
    elif field_type is bool:
        return value.lower() in ("true", "1", "yes")
    elif field_type is str:
        return value
    # Fallback: try numeric conversion for union types (e.g. str | float)
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def run_one(
    evaluator: RankingEvaluator,
    features: list[str],
    name: str,
    n_folds: int = 0,
    model_type: str | None = None,
    raw_embedding_features: bool | None = None,
    overrides: dict[str, str] | None = None,
) -> AblationResult:
    """Train and evaluate with the given feature set.

    When n_folds > 0, use k-fold CV (non-saturating for large training sets).
    """
    from dataclasses import replace

    config = AppConfig.load()
    classifier_updates: dict[str, Any] = {"features": tuple(features)}
    if raw_embedding_features is not None:
        classifier_updates["raw_embedding_features"] = raw_embedding_features
    config = replace(
        config,
        classifier=replace(config.classifier, **classifier_updates),
    )
    single_model_updates: dict[str, Any] = {}
    if model_type is not None:
        single_model_updates["model_type"] = model_type
    if overrides:
        for k, v in overrides.items():
            single_model_updates[k] = _convert_override_value(k, v, config.single_model)
    if single_model_updates:
        config = replace(
            config,
            single_model=replace(config.single_model, **single_model_updates),
        )

    start = time.time()
    if n_folds > 0:
        metrics = evaluator.evaluate_cv(
            n_folds=n_folds, config=config, k_metrics=_DEFAULT_K, parallel=True
        )
    else:
        metrics = evaluator.evaluate(config=config, k_metrics=_DEFAULT_K)
    elapsed = time.time() - start

    dataset = evaluator.dataset
    return AblationResult(
        name=name,
        features=list(features),
        mrr=metrics.get("mrr", 0.0),
        precision_at_5=metrics.get("precision@5", 0.0),
        recall_at_5=metrics.get("recall@5", 0.0),
        ndcg_at_30=metrics.get("ndcg@30", 0.0),
        hit_at_30=metrics.get("hit@30", 0.0),
        mean_rank=metrics.get("mean_rank", 0.0),
        median_rank=metrics.get("median_rank", 0.0),
        n_train=len(dataset.train_stories) if dataset else 0,
        n_test=len(dataset.test_stories) if dataset else 0,
        n_candidates=len(dataset.candidates) if dataset else 0,
        classifier_used=True,
        elapsed_seconds=elapsed,
        n_folds=n_folds,
        nonhn_score_stddev=metrics.get("nonhn_score_stddev", 0.0),
        nonhn_at_0_5_fraction=metrics.get("nonhn_at_0.5_fraction", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot",
        nargs="?",
        default="tests/snapshots/baseline.json",
        help="Path to a ranking evaluation snapshot (JSON).",
    )
    parser.add_argument("--output", type=Path, help="Output JSON file path.")
    parser.add_argument(
        "--baseline",
        default="default",
        help="Label for the baseline feature set.",
    )
    parser.add_argument(
        "--drop-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run baseline minus each individual feature (default: true).",
    )
    parser.add_argument(
        "--single-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run each feature in isolation (default: false).",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        help="Run only these specific feature sets (format: name=feat1,feat2).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="K for precision@K, recall@K."
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=None,
        help="Path to feedback JSON (populates story_age cache).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Use k-fold cross-validation (n_folds) instead of single time-split.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Override model type: logistic, random_forest, gradient_boosting, svm, mlp.",
    )
    parser.add_argument(
        "--raw-embedding-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable raw embedding features.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Key=value overrides for SingleModelConfig (e.g. mlp_activation=logistic).",
    )
    args = parser.parse_args()

    # Load snapshot
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise SystemExit(f"Snapshot not found: {snapshot_path}")

    evaluator = RankingEvaluator(username="ablation")
    if not evaluator.load_snapshot(snapshot_path):
        raise SystemExit(f"Failed to load snapshot from {snapshot_path}")

    # Resolve baseline features from toml config
    baseline_config = AppConfig.load()
    baseline_features = list(baseline_config.classifier.features)

    # Note: Evaluator now populates metadata features dynamically per fold
    # to avoid target leakage, so we do not preload feedback files globally.

    # Build feature sets = evaluator.dataset
    dataset = evaluator.dataset
    assert dataset is not None, (
        "evaluator.dataset should be populated after load_snapshot"
    )
    print(
        f"Dataset: {len(dataset.train_stories)} train, "
        f"{len(dataset.test_stories)} test, "
        f"{len(dataset.candidates)} candidates\n"
    )

    # Build the list of feature-set experiments
    sets: list[tuple[str, list[str]]] = []

    if args.features:
        for spec in args.features:
            if "=" in spec:
                name, feat_list = spec.split("=", 1)
                feats = [f.strip() for f in feat_list.split(",") if f.strip()]
                sets.append((name, feats))
            else:
                sets.append((spec, baseline_features))
    else:
        # Baseline first
        sets.append((args.baseline, baseline_features))

        # Drop-one
        if args.drop_one:
            for f in baseline_features:
                dropped = [x for x in baseline_features if x != f]
                sets.append((f"{args.baseline} - {f}", dropped))

        # Single-feature
        if args.single_features:
            for f in baseline_features:
                sets.append((f"only {f}", [f]))

    # Deduplicate by name (last wins)
    seen: set[str] = set()
    deduped: list[tuple[str, list[str]]] = []
    for name, feats in sets:
        if name not in seen:
            seen.add(name)
            deduped.append((name, feats))
    sets = deduped

    # Run experiments
    results: list[AblationResult] = []
    print(f"Running {len(sets)} evaluations...\n")
    for i, (name, feats) in enumerate(sets, 1):
        print(f"  [{i}/{len(sets)}] {name}...", end=" ", flush=True)
        result = run_one(
            evaluator,
            feats,
            name,
            n_folds=args.cv,
            model_type=args.model_type,
            raw_embedding_features=(
                args.raw_embedding_features
                if args.raw_embedding_features is not None
                else None
            ),
            overrides=_parse_override_args(args.override),
        )
        results.append(result)
        print(
            f"MRR={result.mrr:.3f} NDCG@30={result.ndcg_at_30:.3f} "
            f"hit@30={result.hit_at_30:.3f} mean_rank={result.mean_rank:.1f} "
            f"nonhn_0.5={result.nonhn_at_0_5_fraction:.2f}"
        )

    # Print comparison table
    print()
    header = (
        f"{'name':<40} {'MRR':>6} {'NDCG@30':>7} {'hit@30':>7} "
        f"{'mean_rank':>9} {'P@5':>6} {'R@5':>6} {'train':>6} {'elapsed':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        name_col = r.name[:39]
        print(
            f"{name_col:<40} {r.mrr:>6.3f} {r.ndcg_at_30:>7.3f} "
            f"{r.hit_at_30:>7.3f} {r.mean_rank:>8.1f} "
            f"{r.precision_at_5:>6.3f} "
            f"{r.recall_at_5:>6.3f} {r.n_train:>6} {r.elapsed_seconds:>7.1f}s"
        )

    # Save JSON
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
