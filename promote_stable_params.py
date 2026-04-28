#!/usr/bin/env -S uv run
"""Run multi-seed Optuna tuning and promote only statistically stable params."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_quality import RankingEvaluator, _merge_rank_diagnostic_summaries
from tuning_common import (
    average_seed_metrics as _average_seed_metrics,
    patched_rerank_params,
    render_promoted_toml as _render_promoted_toml,
    resolve_params as _resolved_params,
    score_metrics as _score_metrics,
    validate_candidate_metrics as _validate_candidate_metrics,
)

Z_95: float = 1.96


@dataclass(frozen=True)
class SeedRun:
    seed: int
    run_dir: str
    json_path: str
    best_score: float
    best_params: dict[str, float]
    candidate_params: tuple[dict[str, float], ...] = field(default_factory=tuple)


@dataclass
class CandidateSummary:
    name: str
    params: dict[str, float]
    scores: list[float]
    deltas: list[float]
    metrics: dict[str, float]
    diagnostics: dict[str, object]
    mean_score: float
    mean_delta: float
    std_delta: float
    lcb_delta_95: float
    regressions: int
    validation: dict[str, Any]
    stable: bool


def _parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        seed = int(item)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("No valid seeds parsed")
    return seeds


def _latest_optuna_json(log_dir: Path) -> Path | None:
    files = sorted(
        log_dir.glob("optuna_*.json"),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
        reverse=True,
    )
    return files[0] if files else None


def _parse_numeric_params(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _load_optuna_result(
    path: Path, top_k_per_seed: int
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload in {path}")

    best_score = payload.get("best_score")
    best_params = payload.get("best_params")
    if not isinstance(best_score, (int, float)):
        raise ValueError(f"Missing best_score in {path}")
    if not isinstance(best_params, dict):
        raise ValueError(f"Missing best_params in {path}")

    parsed_best = _parse_numeric_params(best_params)
    if not parsed_best:
        raise ValueError(f"No numeric best_params in {path}")

    top_candidates: list[dict[str, float]] = []
    top_trials = payload.get("top_trials")
    if isinstance(top_trials, list):
        for item in top_trials:
            if len(top_candidates) >= top_k_per_seed:
                break
            if not isinstance(item, dict):
                continue
            parsed = _parse_numeric_params(item.get("params"))
            if parsed:
                top_candidates.append(parsed)

    # Backward compatibility: older optimizer JSON contains only best_params.
    if not top_candidates:
        top_candidates.append(parsed_best)

    # Ensure best params are always represented in the candidate set.
    seen_signatures = {_param_signature(c) for c in top_candidates}
    best_sig = _param_signature(parsed_best)
    if best_sig not in seen_signatures:
        top_candidates.append(parsed_best)

    return float(best_score), parsed_best, top_candidates
def _param_signature(params: dict[str, float]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _collect_candidates(seed_runs: list[SeedRun]) -> list[tuple[str, dict[str, float]]]:
    unique: dict[str, tuple[str, dict[str, float]]] = {}
    for run in seed_runs:
        ranked_params = list(run.candidate_params) or [run.best_params]
        for rank, params in enumerate(ranked_params, start=1):
            name = f"seed_{run.seed}_top{rank}"
            sig = _param_signature(params)
            if sig not in unique:
                unique[sig] = (name, params)
    return list(unique.values())
def _eval_candidate_scores(
    evaluator: RankingEvaluator,
    params: dict[str, float],
    eval_seeds: list[int],
    cv_folds: int,
    std_penalty: float,
) -> tuple[list[float], dict[str, float], dict[str, object]]:
    scores: list[float] = []
    metrics_per_seed: list[dict[str, float]] = []
    diagnostics_per_seed: list[dict[str, object]] = []
    with patched_rerank_params(params) as resolved:
        ranking = resolved["ranking"]
        semantic = resolved["semantic"]
        for seed in eval_seeds:
            np.random.seed(seed)
            diagnostics_summary: dict[str, object] = {}
            metrics = evaluator.evaluate_cv(
                n_folds=cv_folds,
                diversity=float(ranking["diversity_lambda"]),
                knn=int(semantic["knn_neighbors"]),
                neg_weight=float(ranking["negative_weight"]),
                use_classifier=True,
                k_metrics=[10, 20, 30, 40],
                report_each=False,
                parallel=False,
                final_list_count=40,
                diagnostics_summary=diagnostics_summary,
            )
            metrics_per_seed.append(metrics)
            diagnostics_per_seed.append(diagnostics_summary)
            scores.append(_score_metrics(metrics, std_penalty=std_penalty))
    return (
        scores,
        _average_seed_metrics(metrics_per_seed),
        _merge_rank_diagnostic_summaries(diagnostics_per_seed),
    )


def _is_stable(
    deltas: list[float],
    min_improvement: float,
    max_seed_regressions: int,
) -> tuple[bool, float, float, float, int]:
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas))
    regressions = int(sum(1 for d in deltas if d < 0.0))
    if len(deltas) > 1:
        lcb_delta_95 = float(mean_delta - Z_95 * std_delta / math.sqrt(len(deltas)))
    else:
        lcb_delta_95 = mean_delta

    stable = (
        mean_delta >= min_improvement
        and lcb_delta_95 > 0.0
        and regressions <= max_seed_regressions
    )
    return stable, mean_delta, std_delta, lcb_delta_95, regressions


def _run_optuna_seed(
    username: str,
    seed: int,
    log_dir: Path,
    trials: int,
    space: str,
    cv_folds: int,
    candidates: int,
    n_jobs: int,
    cache_only: bool,
    baseline: str,
    guard_tolerance: float,
    std_penalty: float,
    warm_start_log_dir: Path | None,
    top_k_per_seed: int,
    snapshot: Path | None,
) -> SeedRun:
    cmd = [
        sys.executable,
        "optimize_hyperparameters.py",
        username,
        "--trials",
        str(trials),
        "--space",
        space,
        "--cv-folds",
        str(cv_folds),
        "--std-penalty",
        str(std_penalty),
        "--candidates",
        str(candidates),
        "--n-jobs",
        str(n_jobs),
        "--seed",
        str(seed),
        "--log-dir",
        str(log_dir),
        "--baseline",
        baseline,
        "--guard-tolerance",
        str(guard_tolerance),
    ]
    if warm_start_log_dir is not None:
        cmd.extend(["--warm-start-log-dir", str(warm_start_log_dir)])
    if snapshot is not None:
        cmd.extend(["--snapshot", str(snapshot)])
    if cache_only:
        cmd.append("--cache-only")

    env = os.environ.copy()
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    subprocess.run(cmd, check=True, env=env)

    json_path = _latest_optuna_json(log_dir)
    if json_path is None:
        raise RuntimeError(f"No Optuna JSON output found in {log_dir}")
    best_score, best_params, candidate_params = _load_optuna_result(
        json_path, top_k_per_seed=top_k_per_seed
    )
    return SeedRun(
        seed=seed,
        run_dir=str(log_dir),
        json_path=str(json_path),
        best_score=best_score,
        best_params=best_params,
        candidate_params=tuple(candidate_params),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed Optuna, then promote params only when gains are stable."
        )
    )
    parser.add_argument("username", help="HN username")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    parser.add_argument("--trials", type=int, default=120, help="Trials per seed")
    parser.add_argument(
        "--space",
        choices=[
            "core",
            "full",
            "cat_relevance",
            "cat_freshness",
            "cat_semantic",
            "cat_hn",
        ],
        default="core",
        help="Optuna space to search",
    )
    parser.add_argument("--cv-folds", type=int, default=8, help="CV folds")
    parser.add_argument(
        "--std-penalty",
        type=float,
        default=0.5,
        help="Penalty multiplier for weighted metric std in objective",
    )
    parser.add_argument("--candidates", type=int, default=500, help="Candidate pool size")
    parser.add_argument("--n-jobs", type=int, default=4, help="Optuna parallel workers")
    parser.add_argument(
        "--baseline",
        default=".cache/metrics_baseline.json",
        help="Baseline metrics path for optimizer guard output",
    )
    parser.add_argument(
        "--guard-tolerance",
        type=float,
        default=0.0,
        help="Allowed drop for optimizer guard logging",
    )
    parser.add_argument(
        "--run-root",
        default="runs/promotion",
        help="Root directory for promotion runs/artifacts",
    )
    parser.add_argument(
        "--warm-start-run",
        default=None,
        help="Optional previous promotion run dir for per-seed warm start (expects seed_<n>/ subdirs)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help="Minimum mean objective gain over baseline",
    )
    parser.add_argument(
        "--max-seed-regressions",
        type=int,
        default=0,
        help="Max seeds allowed to regress vs baseline",
    )
    parser.add_argument(
        "--top-k-per-seed",
        type=int,
        default=5,
        help="Evaluate top-K Optuna trials per seed as promotion candidates",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use cached data only for all evaluation/loading",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Frozen benchmark snapshot to use for optimization and validation.",
    )
    args = parser.parse_args()
    if args.top_k_per_seed < 1:
        raise SystemExit("--top-k-per-seed must be >= 1")

    seeds = _parse_seed_list(args.seeds)
    warm_start_root = Path(args.warm_start_run) if args.warm_start_run else None
    if warm_start_root is not None and not warm_start_root.exists():
        raise SystemExit(f"Warm-start run directory not found: {warm_start_root}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root) / f"{ts}_{args.space}_{len(seeds)}seeds"
    run_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[promotion] start username={args.username} space={args.space} "
        f"seeds={seeds} trials={args.trials} cv={args.cv_folds} "
        f"std_penalty={args.std_penalty} top_k_per_seed={args.top_k_per_seed}"
    )
    print(f"[promotion] artifacts: {run_root}")

    seed_runs: list[SeedRun] = []
    for seed in seeds:
        seed_dir = run_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"[promotion] optimize seed={seed} log_dir={seed_dir}")
        warm_seed_dir: Path | None = None
        if warm_start_root is not None:
            candidate_warm_dir = warm_start_root / f"seed_{seed}"
            if candidate_warm_dir.exists():
                warm_seed_dir = candidate_warm_dir
                print(
                    f"[promotion] seed={seed} warm_start_log_dir={warm_seed_dir}"
                )
            else:
                print(
                    f"[promotion] seed={seed} no warm start dir at "
                    f"{candidate_warm_dir}; using default ranges"
                )
        run = _run_optuna_seed(
            username=args.username,
            seed=seed,
            log_dir=seed_dir,
            trials=args.trials,
            space=args.space,
            cv_folds=args.cv_folds,
            candidates=args.candidates,
            n_jobs=args.n_jobs,
            cache_only=args.cache_only,
            baseline=args.baseline,
            guard_tolerance=args.guard_tolerance,
            std_penalty=args.std_penalty,
            warm_start_log_dir=warm_seed_dir,
            top_k_per_seed=args.top_k_per_seed,
            snapshot=args.snapshot,
        )
        seed_runs.append(run)
        print(
            f"[promotion] seed={seed} best_score={run.best_score:.6f} "
            f"json={run.json_path} candidates={len(run.candidate_params)}"
        )

    evaluator = RankingEvaluator(args.username)
    print("[promotion] loading evaluation dataset once...")
    if args.snapshot is not None:
        success = evaluator.load_snapshot(args.snapshot)
    else:
        success = await evaluator.load_data(
            candidate_count=args.candidates,
            use_classifier=True,
            cache_only=args.cache_only,
            allow_stale=args.cache_only,
        )
    if not success:
        raise SystemExit("Failed to load evaluation data")

    baseline_scores, baseline_metrics, baseline_diagnostics = _eval_candidate_scores(
        evaluator,
        params={},
        eval_seeds=seeds,
        cv_folds=args.cv_folds,
        std_penalty=args.std_penalty,
    )
    baseline_mean = float(np.mean(baseline_scores))
    print(
        f"[promotion] baseline mean score={baseline_mean:.6f} "
        f"per_seed={[round(s, 6) for s in baseline_scores]}"
    )

    candidates = _collect_candidates(seed_runs)
    print(f"[promotion] evaluating {len(candidates)} unique candidate param sets")
    summaries: list[CandidateSummary] = []
    for name, params in candidates:
        scores, mean_metrics, diagnostics = _eval_candidate_scores(
            evaluator,
            params=params,
            eval_seeds=seeds,
            cv_folds=args.cv_folds,
            std_penalty=args.std_penalty,
        )
        deltas = [score - base for score, base in zip(scores, baseline_scores, strict=True)]
        stable, mean_delta, std_delta, lcb_delta_95, regressions = _is_stable(
            deltas=deltas,
            min_improvement=args.min_improvement,
            max_seed_regressions=args.max_seed_regressions,
        )
        validation = _validate_candidate_metrics(
            mean_metrics,
            baseline_metrics,
            std_penalty=args.std_penalty,
        )
        summary = CandidateSummary(
            name=name,
            params=params,
            scores=scores,
            deltas=deltas,
            metrics=mean_metrics,
            diagnostics=diagnostics,
            mean_score=float(np.mean(scores)),
            mean_delta=mean_delta,
            std_delta=std_delta,
            lcb_delta_95=lcb_delta_95,
            regressions=regressions,
            validation=validation,
            stable=stable and bool(validation["promotable"]),
        )
        summaries.append(summary)
        print(
            f"[promotion] candidate={name} mean={summary.mean_score:.6f} "
            f"mean_delta={summary.mean_delta:.6f} lcb95={summary.lcb_delta_95:.6f} "
            f"regressions={summary.regressions} stable={summary.stable} "
            f"promotable={summary.validation['promotable']}"
        )

    summaries.sort(key=lambda s: s.mean_score, reverse=True)
    promotable_summaries = [summary for summary in summaries if summary.stable]
    winner = promotable_summaries[0] if promotable_summaries else summaries[0]
    promoted = bool(promotable_summaries)

    report: dict[str, Any] = {
        "timestamp": ts,
        "username": args.username,
        "space": args.space,
        "seeds": seeds,
        "trials_per_seed": args.trials,
        "cv_folds": args.cv_folds,
        "std_penalty": args.std_penalty,
        "top_k_per_seed": args.top_k_per_seed,
        "candidates": args.candidates,
        "candidate_count": len(candidates),
        "baseline_scores": baseline_scores,
        "baseline_mean": baseline_mean,
        "baseline_metrics": baseline_metrics,
        "baseline_diagnostics": baseline_diagnostics,
        "snapshot": None if args.snapshot is None else str(args.snapshot),
        "seed_runs": [
            {
                "seed": run.seed,
                "run_dir": run.run_dir,
                "json_path": run.json_path,
                "best_score": run.best_score,
                "best_params": run.best_params,
                "candidate_param_count": len(run.candidate_params),
                "candidate_params": list(run.candidate_params),
            }
            for run in seed_runs
        ],
        "candidate_summaries": [
            {
                "name": s.name,
                "params": s.params,
                "scores": s.scores,
                "deltas": s.deltas,
                "metrics": s.metrics,
                "diagnostics": s.diagnostics,
                "mean_score": s.mean_score,
                "mean_delta": s.mean_delta,
                "std_delta": s.std_delta,
                "lcb_delta_95": s.lcb_delta_95,
                "regressions": s.regressions,
                "validation": s.validation,
                "stable": s.stable,
            }
            for s in summaries
        ],
        "winner": {
            "name": winner.name,
            "params": winner.params,
            "metrics": winner.metrics,
            "diagnostics": winner.diagnostics,
            "mean_score": winner.mean_score,
            "mean_delta": winner.mean_delta,
            "lcb_delta_95": winner.lcb_delta_95,
            "regressions": winner.regressions,
            "validation": winner.validation,
            "stable": winner.stable,
        },
        "promotion": {
            "promoted": promoted,
            "min_improvement": args.min_improvement,
            "max_seed_regressions": args.max_seed_regressions,
        },
    }

    report_path = run_root / "promotion_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[promotion] wrote {report_path}")

    if promoted:
        resolved = _resolved_params(winner.params)
        promoted_json = run_root / "promoted_params.json"
        promoted_toml = run_root / "promoted_params.toml"
        promoted_json.write_text(json.dumps(resolved, indent=2))
        promoted_toml.write_text(_render_promoted_toml(resolved))
        print(f"[promotion] promoted params written to {promoted_json}")
        print(f"[promotion] promoted TOML snippet written to {promoted_toml}")
        return 0

    print("[promotion] no promotion: stability gate not met")
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
