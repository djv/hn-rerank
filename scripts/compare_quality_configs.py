#!/usr/bin/env -S uv run
"""Compare two ranking configs on the same quality-evaluation dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np

from api.config import AppConfig
from evaluate_quality import RankingEvaluator
from tuning_common import (
    VALIDATION_GUARD_METRICS,
    VALIDATION_PRIMARY_METRICS,
    score_metrics,
)


DEFAULT_K_METRICS = [10, 20, 30, 40]


@dataclass(frozen=True)
class ConfigMetrics:
    label: str
    metrics: dict[str, float]
    score: float


@dataclass(frozen=True)
class ComparisonResult:
    baseline: ConfigMetrics
    candidate: ConfigMetrics
    relative_delta: float
    primary_failures: list[str]
    guard_failures: list[str]
    passed: bool


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        seeds.append(int(text))
    return seeds or [0]


def average_metric_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    if not runs:
        return {}
    keys = sorted({key for run in runs for key in run})
    return {
        key: float(fmean(run[key] for run in runs if key in run))
        for key in keys
    }


def evaluate_config(
    evaluator: RankingEvaluator,
    config: AppConfig,
    *,
    seeds: list[int],
    cv_folds: int,
    count: int,
    k_metrics: list[int],
) -> dict[str, float]:
    runs: list[dict[str, float]] = []
    for seed in seeds:
        np.random.seed(seed)
        runs.append(
            evaluator.evaluate_cv(
                n_folds=cv_folds,
                config=config,
                k_metrics=k_metrics,
                report_each=False,
                parallel=False,
                final_list_count=count,
            )
        )
    return average_metric_runs(runs)


def compare_metrics(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    *,
    min_relative_improvement: float,
    std_penalty: float,
    primary_metrics: tuple[str, ...] = VALIDATION_PRIMARY_METRICS,
    guard_metrics: tuple[str, ...] = VALIDATION_GUARD_METRICS,
) -> ComparisonResult:
    baseline_score = score_metrics(baseline_metrics, std_penalty=std_penalty)
    candidate_score = score_metrics(candidate_metrics, std_penalty=std_penalty)
    relative_delta = (
        (candidate_score - baseline_score) / abs(baseline_score)
        if baseline_score
        else float("inf")
    )

    primary_failures = [
        metric
        for metric in primary_metrics
        if candidate_metrics.get(metric, 0.0) < baseline_metrics.get(metric, 0.0)
    ]
    guard_failures = [
        metric
        for metric in guard_metrics
        if candidate_metrics.get(metric, 0.0) < baseline_metrics.get(metric, 0.0)
    ]
    passed = (
        relative_delta >= min_relative_improvement
        and not primary_failures
        and not guard_failures
    )
    return ComparisonResult(
        baseline=ConfigMetrics(
            label="baseline",
            metrics=baseline_metrics,
            score=baseline_score,
        ),
        candidate=ConfigMetrics(
            label="candidate",
            metrics=candidate_metrics,
            score=candidate_score,
        ),
        relative_delta=float(relative_delta),
        primary_failures=primary_failures,
        guard_failures=guard_failures,
        passed=passed,
    )


def print_metrics(label: str, config_metrics: ConfigMetrics, k_metrics: list[int]) -> None:
    print(f"\n{label}:")
    print(f"  weighted_score: {config_metrics.score:.4f}")
    print(f"  mrr: {config_metrics.metrics.get('mrr', 0.0):.4f}")
    for k in k_metrics:
        print(
            f"  @{k}: "
            f"ndcg={config_metrics.metrics.get(f'ndcg@{k}', 0.0):.4f}, "
            f"map={config_metrics.metrics.get(f'map@{k}', 0.0):.4f}, "
            f"precision={config_metrics.metrics.get(f'precision@{k}', 0.0):.1%}, "
            f"recall={config_metrics.metrics.get(f'recall@{k}', 0.0):.1%}"
        )


def result_to_json(result: ComparisonResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "relative_delta": result.relative_delta,
        "primary_failures": result.primary_failures,
        "guard_failures": result.guard_failures,
        "baseline": {
            "score": result.baseline.score,
            "metrics": result.baseline.metrics,
        },
        "candidate": {
            "score": result.candidate.score,
            "metrics": result.candidate.metrics,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two hn_rerank TOML configs on the same CV folds."
    )
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--username", default="pure_coder")
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--limit-pos", type=int, default=200)
    parser.add_argument("--limit-neg", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--std-penalty", type=float, default=0.5)
    parser.add_argument("--min-relative-improvement", type=float, default=0.05)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--age-matched", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    baseline_config = AppConfig.load(args.baseline_config, username=args.username)
    candidate_config = AppConfig.load(args.candidate_config, username=args.username)
    seeds = parse_seeds(args.seeds)
    k_metrics = [k for k in DEFAULT_K_METRICS if k <= args.count] or [args.count]

    evaluator = RankingEvaluator(args.username)
    loaded = await evaluator.load_data(
        limit_pos=args.limit_pos,
        limit_neg=args.limit_neg,
        candidate_count=args.candidates,
        use_classifier=True,
        cache_only=args.cache_only,
        allow_stale=args.cache_only,
        age_matched=args.age_matched,
    )
    if not loaded:
        raise SystemExit("failed to load evaluation data")

    print(
        "Comparing configs on shared data: "
        f"cv_folds={args.cv_folds}, count={args.count}, "
        f"candidates={args.candidates}, seeds={','.join(map(str, seeds))}"
    )
    baseline_metrics = evaluate_config(
        evaluator,
        baseline_config,
        seeds=seeds,
        cv_folds=args.cv_folds,
        count=args.count,
        k_metrics=k_metrics,
    )
    candidate_metrics = evaluate_config(
        evaluator,
        candidate_config,
        seeds=seeds,
        cv_folds=args.cv_folds,
        count=args.count,
        k_metrics=k_metrics,
    )
    result = compare_metrics(
        baseline_metrics,
        candidate_metrics,
        min_relative_improvement=args.min_relative_improvement,
        std_penalty=args.std_penalty,
    )

    print_metrics("Baseline", result.baseline, k_metrics)
    print_metrics("Candidate", result.candidate, k_metrics)
    print(
        "\nDecision: "
        f"{'PASS' if result.passed else 'FAIL'} "
        f"(relative_delta={result.relative_delta:.2%}, "
        f"required={args.min_relative_improvement:.2%})"
    )
    if result.primary_failures:
        print(f"  primary regressions: {', '.join(result.primary_failures)}")
    if result.guard_failures:
        print(f"  guard regressions: {', '.join(result.guard_failures)}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result_to_json(result), indent=2, sort_keys=True))
        print(f"\nWrote comparison JSON: {args.json_out}")

    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
