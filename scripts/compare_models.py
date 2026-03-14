#!/usr/bin/env -S uv run
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.constants import POSITIVE_RECENCY_ENABLED


IMPORTANT_METRICS = [
    "mrr",
    "ndcg@10",
    "ndcg@30",
    "precision@20",
    "recall@30",
    "recall@50",
    "mean_rank",
]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class ModelRuntime:
    fingerprint: str
    cache_dir: Path
    cache_version: str


def parse_model_arg(raw: str) -> ModelSpec:
    label, sep, path_text = raw.partition("=")
    label = label.strip()
    path_text = path_text.strip()

    if not sep or not label or not path_text:
        raise argparse.ArgumentTypeError(
            "model must be provided as label=path"
        )

    path = Path(path_text).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"model path is not a directory: {path}")
    if not (path / "model.onnx").is_file():
        raise argparse.ArgumentTypeError(f"model directory missing model.onnx: {path}")

    return ModelSpec(label=label, path=path.resolve())


def normalize_seeds(seeds: list[int] | None) -> list[int]:
    if not seeds:
        return [0]

    deduped: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        deduped.append(seed)
    return deduped


def summarize_seed_metrics(seed_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not seed_metrics:
        return {}

    summary: dict[str, float] = {}
    keys = sorted(seed_metrics[0])
    for key in keys:
        values = [metrics[key] for metrics in seed_metrics]
        summary[key] = float(statistics.fmean(values))
        if len(values) > 1 and not key.endswith("_std"):
            summary[f"{key}_seed_std"] = float(statistics.pstdev(values))

    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple local ONNX model directories on the same evaluation dataset."
    )
    parser.add_argument("username", help="HN username to evaluate")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        type=parse_model_arg,
        required=True,
        help="Repeatable model spec in label=path form",
    )
    parser.add_argument(
        "--classifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable classifier mode",
    )
    parser.add_argument(
        "--recency",
        action=argparse.BooleanOptionalAction,
        default=POSITIVE_RECENCY_ENABLED,
        help="Enable or disable recency weighting for positive signals",
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        action="append",
        type=int,
        default=None,
        help="Repeatable RNG seed for cross-validation splits (default: 0)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for RankingEvaluator.evaluate_cv()",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=200,
        help="Candidate pool size passed to dataset loading",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="Holdout fraction used when preparing the shared dataset",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use cached HN/story data only and allow stale cached entries",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the full benchmark payload as JSON",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.models) < 2:
        parser.error("provide at least two --model entries")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if args.candidates < 1:
        parser.error("--candidates must be at least 1")
    if not 0.0 < args.holdout < 1.0:
        parser.error("--holdout must be between 0 and 1")

    args.seeds = normalize_seeds(args.seeds)
    return args


@contextlib.contextmanager
def override_rerank_model(
    rerank_module: Any,
    spec: ModelSpec,
    cache_root: Path,
) -> Any:
    fingerprint = sha256_file(spec.path / "model.onnx")
    cache_version = f"compare-{fingerprint[:12]}"
    cache_dir = cache_root / cache_version
    cache_dir.mkdir(parents=True, exist_ok=True)

    original_model = getattr(rerank_module, "_model", None)
    original_version = rerank_module.EMBEDDING_MODEL_VERSION
    original_cache_dir = rerank_module.CACHE_DIR

    model = rerank_module.ONNXEmbeddingModel(model_dir=str(spec.path))
    rerank_module._model = model
    rerank_module.EMBEDDING_MODEL_VERSION = cache_version
    rerank_module.CACHE_DIR = cache_dir
    try:
        yield ModelRuntime(
            fingerprint=fingerprint,
            cache_dir=cache_dir,
            cache_version=cache_version,
        )
    finally:
        rerank_module._model = original_model
        rerank_module.EMBEDDING_MODEL_VERSION = original_version
        rerank_module.CACHE_DIR = original_cache_dir


def clone_dataset_for_model(eq_module: Any, rerank_module: Any, dataset: Any) -> Any:
    neg_embeddings = None
    if dataset.neg_stories:
        neg_embeddings = rerank_module.get_embeddings(
            [story.text_content for story in dataset.neg_stories]
        )

    # evaluate_cv() recomputes positive embeddings per fold from story text, so the
    # shared train_embeddings from dataset loading are not reused across models.
    return eq_module.EvaluationDataset(
        train_stories=list(dataset.train_stories),
        test_stories=list(dataset.test_stories),
        neg_stories=list(dataset.neg_stories),
        candidates=list(dataset.candidates),
        train_embeddings=dataset.train_embeddings,
        neg_embeddings=neg_embeddings,
        pos_weights=None if dataset.pos_weights is None else dataset.pos_weights.copy(),
        test_ids=set(dataset.test_ids),
    )


def build_output_payload(
    args: argparse.Namespace,
    dataset: Any,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "username": args.username,
        "classifier": args.classifier,
        "recency": args.recency,
        "cv_folds": args.cv_folds,
        "candidate_count": args.candidates,
        "holdout": args.holdout,
        "cache_only": args.cache_only,
        "seeds": args.seeds,
        "dataset": {
            "train_stories": len(dataset.train_stories),
            "test_stories": len(dataset.test_stories),
            "negative_stories": len(dataset.neg_stories),
            "candidates": len(dataset.candidates),
        },
        "models": results,
    }


def print_dataset_summary(dataset: Any) -> None:
    print("Shared dataset:")
    print(f"  train stories: {len(dataset.train_stories)}")
    print(f"  test stories: {len(dataset.test_stories)}")
    print(f"  negative stories: {len(dataset.neg_stories)}")
    print(f"  candidates: {len(dataset.candidates)}")


def print_model_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(f"\n[{result['label']}] {result['path']}")
    print(f"  sha256: {result['model_sha256']}")
    for key in IMPORTANT_METRICS:
        if key not in summary:
            continue
        value = summary[key]
        seed_std_key = f"{key}_seed_std"
        if seed_std_key in summary:
            print(f"  {key}: {value:.4f} (seed std {summary[seed_std_key]:.4f})")
        else:
            print(f"  {key}: {value:.4f}")


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    import evaluate_quality as eq
    from api import rerank

    cache_root = Path(".cache/model_compare_embeddings")

    first_spec = args.models[0]
    with override_rerank_model(rerank, first_spec, cache_root):
        evaluator = eq.RankingEvaluator(args.username)
        success = await evaluator.load_data(
            holdout=args.holdout,
            limit_pos=10_000,
            limit_neg=10_000,
            candidate_count=args.candidates,
            use_classifier=args.classifier,
            use_recency=args.recency,
            cache_only=args.cache_only,
            allow_stale=args.cache_only,
        )
        if not success or evaluator.dataset is None:
            raise SystemExit(1)
        shared_dataset = evaluator.dataset

    print_dataset_summary(shared_dataset)

    results: list[dict[str, Any]] = []
    for spec in args.models:
        with override_rerank_model(rerank, spec, cache_root) as runtime:
            evaluator = eq.RankingEvaluator(args.username)
            evaluator.dataset = clone_dataset_for_model(eq, rerank, shared_dataset)

            per_seed: list[dict[str, Any]] = []
            for seed in args.seeds:
                np.random.seed(seed)
                metrics = evaluator.evaluate_cv(
                    n_folds=args.cv_folds,
                    use_classifier=args.classifier,
                    report_each=False,
                    parallel=False,
                )
                per_seed.append({"seed": seed, "metrics": metrics})

            summary = summarize_seed_metrics([item["metrics"] for item in per_seed])
            result = {
                "label": spec.label,
                "path": str(spec.path),
                "model_sha256": runtime.fingerprint,
                "cache_dir": str(runtime.cache_dir),
                "cache_version": runtime.cache_version,
                "summary": summary,
                "per_seed": per_seed,
            }
            results.append(result)
            print_model_summary(result)

    payload = build_output_payload(args, shared_dataset, results)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nWrote JSON results to {args.json_output}")

    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run_benchmark(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
