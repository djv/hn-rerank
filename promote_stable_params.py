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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from evaluate_quality import RankingEvaluator
import api.rerank
from api.constants import (
    ADAPTIVE_HN_THRESHOLD_YOUNG,
    ADAPTIVE_HN_WEIGHT_MIN,
    CLASSIFIER_K_FEAT,
    CLASSIFIER_NEG_SAMPLE_WEIGHT,
    CLUSTER_OUTLIER_SIMILARITY_THRESHOLD,
    CLUSTER_SPECTRAL_NEIGHBORS,
    DEFAULT_CLUSTER_COUNT,
    FRESHNESS_HALF_LIFE_HOURS,
    FRESHNESS_MAX_BOOST,
    HN_SCORE_NORMALIZATION_CAP,
    KNN_MAXSIM_WEIGHT,
    KNN_NEIGHBORS,
    KNN_SIGMOID_K,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_NEGATIVE_WEIGHT,
)
from optimize_hyperparameters import (
    ADAPTIVE_HN_DELTA,
    HN_THRESHOLD_GAP,
    _derive_classifier_diversity_lambda,
)

OBJECTIVE_WEIGHTS: dict[str, float] = {
    "mrr": 0.30,
    "ndcg@10": 0.35,
    "ndcg@30": 0.20,
    "recall@50": 0.15,
}

Z_95: float = 1.96


@dataclass(frozen=True)
class SeedRun:
    seed: int
    run_dir: str
    json_path: str
    best_score: float
    best_params: dict[str, float]


@dataclass
class CandidateSummary:
    name: str
    params: dict[str, float]
    scores: list[float]
    deltas: list[float]
    mean_score: float
    mean_delta: float
    std_delta: float
    lcb_delta_95: float
    regressions: int
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


def _load_optuna_result(path: Path) -> tuple[float, dict[str, float]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload in {path}")

    best_score = payload.get("best_score")
    best_params = payload.get("best_params")
    if not isinstance(best_score, (int, float)):
        raise ValueError(f"Missing best_score in {path}")
    if not isinstance(best_params, dict):
        raise ValueError(f"Missing best_params in {path}")

    parsed: dict[str, float] = {}
    for key, value in best_params.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not parsed:
        raise ValueError(f"No numeric best_params in {path}")
    return float(best_score), parsed


def _score_metrics(metrics: dict[str, float]) -> float:
    mean = sum(w * metrics.get(k, 0.0) for k, w in OBJECTIVE_WEIGHTS.items())
    std = sum(w * metrics.get(f"{k}_std", 0.0) for k, w in OBJECTIVE_WEIGHTS.items())
    return float(mean - 0.5 * std)


def _param_signature(params: dict[str, float]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _collect_candidates(seed_runs: list[SeedRun]) -> list[tuple[str, dict[str, float]]]:
    unique: dict[str, tuple[str, dict[str, float]]] = {}
    for run in seed_runs:
        name = f"seed_{run.seed}"
        sig = _param_signature(run.best_params)
        if sig not in unique:
            unique[sig] = (name, run.best_params)
    return list(unique.values())


def _resolved_params(params: dict[str, float]) -> dict[str, dict[str, float]]:
    diversity_lambda = float(params.get("diversity_lambda", RANKING_DIVERSITY_LAMBDA))
    ranking = {
        "negative_weight": float(params.get("neg_weight", RANKING_NEGATIVE_WEIGHT)),
        "diversity_lambda": diversity_lambda,
        "diversity_lambda_classifier": float(
            _derive_classifier_diversity_lambda(diversity_lambda)
        ),
    }

    adaptive_hn_min = float(params.get("adaptive_hn_min", ADAPTIVE_HN_WEIGHT_MIN))
    threshold_young = float(
        params.get("hn_threshold_young", ADAPTIVE_HN_THRESHOLD_YOUNG)
    )
    adaptive_hn = {
        "weight_min": adaptive_hn_min,
        "weight_max": adaptive_hn_min + ADAPTIVE_HN_DELTA,
        "threshold_young": threshold_young,
        "threshold_old": threshold_young + HN_THRESHOLD_GAP,
        "score_normalization_cap": float(
            params.get("hn_score_cap", HN_SCORE_NORMALIZATION_CAP)
        ),
    }

    freshness = {
        "half_life_hours": float(
            params.get("freshness_half_life", FRESHNESS_HALF_LIFE_HOURS)
        ),
        "max_boost": float(params.get("freshness_boost", FRESHNESS_MAX_BOOST)),
    }

    semantic = {
        "knn_sigmoid_k": float(params.get("knn_sigmoid_k", KNN_SIGMOID_K)),
        "knn_maxsim_weight": float(params.get("knn_maxsim_weight", KNN_MAXSIM_WEIGHT)),
        "knn_neighbors": int(round(float(params.get("knn_k", KNN_NEIGHBORS)))),
    }

    classifier = {
        "k_feat": int(round(float(params.get("classifier_k_feat", CLASSIFIER_K_FEAT)))),
        "neg_sample_weight": float(
            params.get("classifier_neg_sample_weight", CLASSIFIER_NEG_SAMPLE_WEIGHT)
        ),
    }

    return {
        "ranking": ranking,
        "adaptive_hn": adaptive_hn,
        "freshness": freshness,
        "semantic": semantic,
        "classifier": classifier,
    }


def _render_promoted_toml(resolved: dict[str, dict[str, float]]) -> str:
    ranking = resolved["ranking"]
    adaptive_hn = resolved["adaptive_hn"]
    freshness = resolved["freshness"]
    semantic = resolved["semantic"]
    classifier = resolved["classifier"]

    return (
        "# Auto-generated promoted params.\n"
        "# Merge this into hn_rerank.toml under [hn_rerank.*] sections.\n\n"
        "[hn_rerank.ranking]\n"
        f"negative_weight = {ranking['negative_weight']:.10f}\n"
        f"diversity_lambda = {ranking['diversity_lambda']:.10f}\n"
        f"diversity_lambda_classifier = {ranking['diversity_lambda_classifier']:.10f}\n\n"
        "[hn_rerank.adaptive_hn]\n"
        f"weight_min = {adaptive_hn['weight_min']:.10f}\n"
        f"weight_max = {adaptive_hn['weight_max']:.10f}\n"
        f"threshold_young = {adaptive_hn['threshold_young']:.10f}\n"
        f"threshold_old = {adaptive_hn['threshold_old']:.10f}\n"
        f"score_normalization_cap = {adaptive_hn['score_normalization_cap']:.10f}\n\n"
        "[hn_rerank.freshness]\n"
        f"half_life_hours = {freshness['half_life_hours']:.10f}\n"
        f"max_boost = {freshness['max_boost']:.10f}\n\n"
        "[hn_rerank.semantic]\n"
        f"knn_sigmoid_k = {semantic['knn_sigmoid_k']:.10f}\n"
        f"knn_maxsim_weight = {semantic['knn_maxsim_weight']:.10f}\n"
        f"knn_neighbors = {semantic['knn_neighbors']}\n\n"
        "[hn_rerank.classifier]\n"
        f"k_feat = {classifier['k_feat']}\n"
        f"neg_sample_weight = {classifier['neg_sample_weight']:.10f}\n"
    )


def _eval_candidate_scores(
    evaluator: RankingEvaluator,
    params: dict[str, float],
    eval_seeds: list[int],
    cv_folds: int,
) -> list[float]:
    resolved = _resolved_params(params)
    ranking = resolved["ranking"]
    adaptive_hn = resolved["adaptive_hn"]
    freshness = resolved["freshness"]
    semantic = resolved["semantic"]
    classifier = resolved["classifier"]

    patch_kwargs: dict[str, Any] = {
        "ADAPTIVE_HN_WEIGHT_MIN": adaptive_hn["weight_min"],
        "ADAPTIVE_HN_WEIGHT_MAX": adaptive_hn["weight_max"],
        "ADAPTIVE_HN_THRESHOLD_YOUNG": adaptive_hn["threshold_young"],
        "ADAPTIVE_HN_THRESHOLD_OLD": adaptive_hn["threshold_old"],
        "HN_SCORE_NORMALIZATION_CAP": adaptive_hn["score_normalization_cap"],
        "FRESHNESS_MAX_BOOST": freshness["max_boost"],
        "FRESHNESS_HALF_LIFE_HOURS": freshness["half_life_hours"],
        "KNN_SIGMOID_K": semantic["knn_sigmoid_k"],
        "KNN_MAXSIM_WEIGHT": semantic["knn_maxsim_weight"],
        "KNN_NEIGHBORS": semantic["knn_neighbors"],
        "RANKING_DIVERSITY_LAMBDA_CLASSIFIER": ranking[
            "diversity_lambda_classifier"
        ],
        "CLASSIFIER_K_FEAT": classifier["k_feat"],
        "CLASSIFIER_NEG_SAMPLE_WEIGHT": classifier["neg_sample_weight"],
        # Kept fixed by optimization script; keep parity.
        "CLUSTER_OUTLIER_SIMILARITY_THRESHOLD": CLUSTER_OUTLIER_SIMILARITY_THRESHOLD,
        "DEFAULT_CLUSTER_COUNT": DEFAULT_CLUSTER_COUNT,
        "CLUSTER_SPECTRAL_NEIGHBORS": CLUSTER_SPECTRAL_NEIGHBORS,
    }

    scores: list[float] = []
    with patch.multiple(api.rerank, **patch_kwargs):
        for seed in eval_seeds:
            np.random.seed(seed)
            metrics = evaluator.evaluate_cv(
                n_folds=cv_folds,
                diversity=ranking["diversity_lambda"],
                knn=semantic["knn_neighbors"],
                neg_weight=ranking["negative_weight"],
                use_classifier=True,
                k_metrics=[10, 30, 50],
                report_each=False,
                parallel=False,
            )
            scores.append(_score_metrics(metrics))
    return scores


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
    if cache_only:
        cmd.append("--cache-only")

    env = os.environ.copy()
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    subprocess.run(cmd, check=True, env=env)

    json_path = _latest_optuna_json(log_dir)
    if json_path is None:
        raise RuntimeError(f"No Optuna JSON output found in {log_dir}")
    best_score, best_params = _load_optuna_result(json_path)
    return SeedRun(
        seed=seed,
        run_dir=str(log_dir),
        json_path=str(json_path),
        best_score=best_score,
        best_params=best_params,
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
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")
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
        "--cache-only",
        action="store_true",
        help="Use cached data only for all evaluation/loading",
    )
    args = parser.parse_args()

    seeds = _parse_seed_list(args.seeds)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root) / f"{ts}_{args.space}_{len(seeds)}seeds"
    run_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[promotion] start username={args.username} space={args.space} "
        f"seeds={seeds} trials={args.trials} cv={args.cv_folds}"
    )
    print(f"[promotion] artifacts: {run_root}")

    seed_runs: list[SeedRun] = []
    for seed in seeds:
        seed_dir = run_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"[promotion] optimize seed={seed} log_dir={seed_dir}")
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
        )
        seed_runs.append(run)
        print(
            f"[promotion] seed={seed} best_score={run.best_score:.6f} "
            f"json={run.json_path}"
        )

    evaluator = RankingEvaluator(args.username)
    print("[promotion] loading evaluation dataset once...")
    success = await evaluator.load_data(
        holdout=0.2,
        candidate_count=args.candidates,
        use_classifier=True,
        use_recency=True,
        cache_only=args.cache_only,
        allow_stale=args.cache_only,
    )
    if not success:
        raise SystemExit("Failed to load evaluation data")

    baseline_scores = _eval_candidate_scores(
        evaluator,
        params={},
        eval_seeds=seeds,
        cv_folds=args.cv_folds,
    )
    baseline_mean = float(np.mean(baseline_scores))
    print(
        f"[promotion] baseline mean score={baseline_mean:.6f} "
        f"per_seed={[round(s, 6) for s in baseline_scores]}"
    )

    candidates = _collect_candidates(seed_runs)
    summaries: list[CandidateSummary] = []
    for name, params in candidates:
        scores = _eval_candidate_scores(
            evaluator,
            params=params,
            eval_seeds=seeds,
            cv_folds=args.cv_folds,
        )
        deltas = [score - base for score, base in zip(scores, baseline_scores, strict=True)]
        stable, mean_delta, std_delta, lcb_delta_95, regressions = _is_stable(
            deltas=deltas,
            min_improvement=args.min_improvement,
            max_seed_regressions=args.max_seed_regressions,
        )
        summary = CandidateSummary(
            name=name,
            params=params,
            scores=scores,
            deltas=deltas,
            mean_score=float(np.mean(scores)),
            mean_delta=mean_delta,
            std_delta=std_delta,
            lcb_delta_95=lcb_delta_95,
            regressions=regressions,
            stable=stable,
        )
        summaries.append(summary)
        print(
            f"[promotion] candidate={name} mean={summary.mean_score:.6f} "
            f"mean_delta={summary.mean_delta:.6f} lcb95={summary.lcb_delta_95:.6f} "
            f"regressions={summary.regressions} stable={summary.stable}"
        )

    summaries.sort(key=lambda s: s.mean_score, reverse=True)
    winner = summaries[0]
    promoted = bool(winner.stable)

    report: dict[str, Any] = {
        "timestamp": ts,
        "username": args.username,
        "space": args.space,
        "seeds": seeds,
        "trials_per_seed": args.trials,
        "cv_folds": args.cv_folds,
        "candidates": args.candidates,
        "baseline_scores": baseline_scores,
        "baseline_mean": baseline_mean,
        "seed_runs": [
            {
                "seed": run.seed,
                "run_dir": run.run_dir,
                "json_path": run.json_path,
                "best_score": run.best_score,
                "best_params": run.best_params,
            }
            for run in seed_runs
        ],
        "candidate_summaries": [
            {
                "name": s.name,
                "params": s.params,
                "scores": s.scores,
                "deltas": s.deltas,
                "mean_score": s.mean_score,
                "mean_delta": s.mean_delta,
                "std_delta": s.std_delta,
                "lcb_delta_95": s.lcb_delta_95,
                "regressions": s.regressions,
                "stable": s.stable,
            }
            for s in summaries
        ],
        "winner": {
            "name": winner.name,
            "mean_score": winner.mean_score,
            "mean_delta": winner.mean_delta,
            "lcb_delta_95": winner.lcb_delta_95,
            "regressions": winner.regressions,
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
