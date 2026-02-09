#!/usr/bin/env -S uv run
"""
Optimize HN Rerank hyperparameters using Optuna.
Usage: uv run optimize_hyperparameters.py <username>
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, NopPruner


def _ensure_joblib_settings() -> None:
    # Ensure joblib temp dirs are writable for parallel execution.
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    tmp = os.environ.get("JOBLIB_TEMP_FOLDER") or os.environ.get("LOKY_TEMP_FOLDER")
    if not tmp:
        tmp = str(Path(__file__).resolve().parent / ".cache" / "joblib")
        os.environ["JOBLIB_TEMP_FOLDER"] = tmp
        os.environ["LOKY_TEMP_FOLDER"] = tmp
    Path(tmp).mkdir(parents=True, exist_ok=True)


_ensure_joblib_settings()

from evaluate_quality import RankingEvaluator, DEFAULT_GUARD_METRICS, _guard_metrics, _load_baseline  # noqa: E402
import api.rerank  # noqa: E402
from api.constants import (  # noqa: E402
    ADAPTIVE_HN_THRESHOLD_YOUNG,
    ADAPTIVE_HN_WEIGHT_MIN,
    CLASSIFIER_K_FEAT,
    CLASSIFIER_NEG_SAMPLE_WEIGHT,
    CLUSTER_OUTLIER_SIMILARITY_THRESHOLD,
    DEFAULT_CLUSTER_COUNT,
    CLUSTER_SPECTRAL_NEIGHBORS,
    FRESHNESS_MAX_BOOST,
    FRESHNESS_HALF_LIFE_HOURS,
    HN_SCORE_NORMALIZATION_CAP,
    KNN_NEIGHBORS,
    KNN_SIGMOID_K,
    KNN_MAXSIM_WEIGHT,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_NEGATIVE_WEIGHT,
)

# --- Derived-parameter constants ---
# Gap between young/old HN thresholds (hours). Top trials: 20–80h, default 42h.
HN_THRESHOLD_GAP: float = 42.0
# Delta between adaptive HN min/max. Top trials: 0.01–0.06, default ≈0.033.
ADAPTIVE_HN_DELTA: float = 0.035

# Configure logging to suppress verbose output during optimization
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.INFO)


def _parse_last_log(log_dir: Path) -> dict[str, float] | None:
    """Parse most recent completed Optuna output (log or JSON) for best params."""
    logs = sorted(
        log_dir.glob("optuna_*.log"),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
        reverse=True,
    )
    for best_file in logs:
        best_params: dict[str, float] = {}
        text = best_file.read_text()

        # Find "Best Parameters:" section
        match = re.search(r"Best Parameters:\n((?:\s+\S+:.*\n)+)", text)
        if not match:
            continue

        for line in match.group(1).strip().splitlines():
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    best_params[key.strip()] = float(val.strip())
                except ValueError:
                    pass

        if best_params:
            return best_params

    # Fallback to most recent structured JSON if logs are incomplete/in-progress.
    jsons = sorted(
        log_dir.glob("optuna_*.json"),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
        reverse=True,
    )
    for json_file in jsons:
        try:
            payload = json.loads(json_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        best = payload.get("best_params") if isinstance(payload, dict) else None
        if not isinstance(best, dict):
            continue
        parsed: dict[str, float] = {}
        for key, val in best.items():
            try:
                parsed[str(key)] = float(val)
            except (TypeError, ValueError):
                pass
        if parsed:
            return parsed

    return None


def _build_ranges(
    prev: dict[str, float] | None,
    space: str,
) -> dict[str, tuple[float, float]]:
    """Build search ranges centered on previous best for selected space.

    Defaults informed by 140+ trial empirical analysis (2026-02-05):
    - Tight where top-10 converged.
    """
    full_defaults: dict[str, tuple[float, float]] = {
        "diversity_lambda": (0.20, 0.60),
        "neg_weight": (0.30, 0.80),
        "knn_k": (1, 4),
        "adaptive_hn_min": (0.0, 0.07),
        "freshness_boost": (0.04, 0.15),
        "freshness_half_life": (45.0, 100.0),
        "knn_sigmoid_k": (4.0, 8.0),
        "knn_maxsim_weight": (0.25, 0.65),
        "hn_threshold_young": (4.0, 16.0),
        "hn_score_cap": (120.0, 1600.0),
        "classifier_k_feat": (2, 9),
        "classifier_neg_sample_weight": (0.7, 2.0),
    }
    spaces: dict[str, set[str]] = {
        "full": set(full_defaults.keys()),
        "core": set(full_defaults.keys()) - {"freshness_boost", "knn_sigmoid_k", "knn_maxsim_weight"},
        "cat_relevance": {
            "diversity_lambda",
            "neg_weight",
            "knn_k",
            "classifier_k_feat",
            "classifier_neg_sample_weight",
        },
        "cat_freshness": {"freshness_boost", "freshness_half_life"},
        "cat_semantic": {"knn_sigmoid_k", "knn_maxsim_weight"},
        "cat_hn": {"adaptive_hn_min", "hn_threshold_young", "hn_score_cap"},
    }
    tuned_keys = spaces.get(space)
    if tuned_keys is None:
        raise ValueError(f"Unknown space: {space}")
    defaults = {k: v for k, v in full_defaults.items() if k in tuned_keys}

    if prev is None:
        return defaults

    # Narrow ranges to ±20% around previous best, clamped to defaults.
    narrowed: dict[str, tuple[float, float]] = {}
    for key, (lo, hi) in defaults.items():
        if key in prev:
            v = prev[key]
            span = (hi - lo) * 0.2
            narrowed[key] = (max(lo, v - span), min(hi, v + span))
        else:
            narrowed[key] = (lo, hi)
    return narrowed


def _derive_classifier_diversity_lambda(diversity_lambda: float) -> float:
    """Classifier diversity ≥ 0.30 floor (runtime already does this max)."""
    return max(diversity_lambda, 0.30)


def _derive_hn_threshold_old(hn_threshold_young: float) -> float:
    """Old threshold = young + fixed gap."""
    return hn_threshold_young + HN_THRESHOLD_GAP


def _derive_adaptive_hn_max(adaptive_hn_base: float) -> float:
    """Adaptive HN max = base + fixed delta."""
    return adaptive_hn_base + ADAPTIVE_HN_DELTA


def _load_enqueued_params(path: str | None) -> list[dict[str, float]]:
    """Load optional list of parameter dicts to enqueue before optimization."""
    if not path:
        return []
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    out: list[dict[str, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        parsed: dict[str, float] = {}
        for key, val in item.items():
            try:
                parsed[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
        if parsed:
            out.append(parsed)
    return out


async def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters")
    parser.add_argument("username", help="HN username")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--candidates", type=int, default=200, help="Candidate pool size")
    parser.add_argument("--baseline", default=".cache/metrics_baseline.json", help="Path to metrics baseline JSON")
    parser.add_argument("--guard-tolerance", type=float, default=0.0, help="Allowed drop vs baseline")
    parser.add_argument("--cache-only", action="store_true", help="Use cached data only (ignore TTL, no RSS)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler")
    parser.add_argument("--startup-trials", type=int, default=10, help="Random trials before TPE kicks in")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel Optuna workers (threads)")
    parser.add_argument("--no-prune", action="store_true", help="Disable Optuna pruning")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--log-dir", type=str, default=".", help="Directory for log files")
    parser.add_argument(
        "--warm-start-log-dir",
        type=str,
        default=None,
        help="Directory with previous optuna logs/json used for warm-start (defaults to --log-dir).",
    )
    parser.add_argument(
        "--enqueue-params-file",
        type=str,
        default=None,
        help="JSON file containing a list of parameter dicts to enqueue before sampling.",
    )
    parser.add_argument(
        "--space",
        choices=["core", "full", "cat_relevance", "cat_freshness", "cat_semantic", "cat_hn"],
        default="core",
        help="Search-space size: core tunes fewer high-impact knobs; full tunes all non-cluster knobs.",
    )
    args = parser.parse_args()

    # Timestamped log filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"optuna_{ts}.log"

    # Read previous best params to narrow search
    warm_dir = Path(args.warm_start_log_dir) if args.warm_start_log_dir else Path(args.log_dir)
    prev_best = _parse_last_log(warm_dir)
    ranges = _build_ranges(prev_best, args.space)
    if prev_best:
        print(f"Loaded previous completed run from {warm_dir}, narrowing search ranges")
        for k, v in prev_best.items():
            print(f"  {k}: {v:.6f}")
    else:
        print("No previous log found, using default ranges")
    print(f"Search space '{args.space}': {len(ranges)} tuned parameters")

    # 1. Load Data (Once)
    evaluator = RankingEvaluator(args.username)
    print("Loading data...")
    success = await evaluator.load_data(
        holdout=0.2,
        candidate_count=args.candidates,
        use_classifier=True,
        use_recency=True,
        cache_only=args.cache_only,
        allow_stale=args.cache_only,
    )
    if not success:
        print("Failed to load data.")
        sys.exit(1)

    print(f"Data loaded. Starting {args.trials}-trial optimization "
          f"({args.cv_folds}-fold CV, {args.n_jobs} workers)...")

    baseline = _load_baseline(args.baseline)

    def score_metrics(metrics: dict[str, float]) -> float:
        # Reweighted objective: MRR + NDCG@10 + NDCG@30 + Recall@50
        # Variance penalty discourages configs that are unstable across folds
        mean = (
            0.30 * metrics.get("mrr", 0.0)
            + 0.35 * metrics.get("ndcg@10", 0.0)
            + 0.20 * metrics.get("ndcg@30", 0.0)
            + 0.15 * metrics.get("recall@50", 0.0)
        )
        std = (
            0.30 * metrics.get("mrr_std", 0.0)
            + 0.35 * metrics.get("ndcg@10_std", 0.0)
            + 0.20 * metrics.get("ndcg@30_std", 0.0)
            + 0.15 * metrics.get("recall@50_std", 0.0)
        )
        return mean - 0.5 * std

    # 2. Define Objective Function
    def objective(trial: Trial) -> float:
        r = ranges  # alias

        # Ranking weights — classifier diversity derived from diversity_lambda
        diversity_lambda = (
            trial.suggest_float("diversity_lambda", *r["diversity_lambda"])
            if "diversity_lambda" in r
            else RANKING_DIVERSITY_LAMBDA
        )
        classifier_diversity_lambda = _derive_classifier_diversity_lambda(diversity_lambda)
        neg_weight = (
            trial.suggest_float("neg_weight", *r["neg_weight"])
            if "neg_weight" in r
            else RANKING_NEGATIVE_WEIGHT
        )

        # k-NN
        if "knn_k" in r:
            knn_lo, knn_hi = int(r["knn_k"][0]), int(r["knn_k"][1])
            knn_k = trial.suggest_int("knn_k", knn_lo, knn_hi)
        else:
            knn_k = KNN_NEIGHBORS

        # Adaptive HN Weighting — max derived from min + fixed delta
        adaptive_hn_min = (
            trial.suggest_float("adaptive_hn_min", *r["adaptive_hn_min"])
            if "adaptive_hn_min" in r
            else ADAPTIVE_HN_WEIGHT_MIN
        )
        adaptive_hn_max = _derive_adaptive_hn_max(adaptive_hn_min)

        # Freshness
        freshness_boost = (
            trial.suggest_float("freshness_boost", *r["freshness_boost"])
            if "freshness_boost" in r
            else FRESHNESS_MAX_BOOST
        )
        freshness_half_life = (
            trial.suggest_float("freshness_half_life", *r["freshness_half_life"])
            if "freshness_half_life" in r
            else FRESHNESS_HALF_LIFE_HOURS
        )

        # Semantic scoring (k-NN path)
        knn_sigmoid_k = (
            trial.suggest_float("knn_sigmoid_k", *r["knn_sigmoid_k"])
            if "knn_sigmoid_k" in r
            else KNN_SIGMOID_K
        )
        knn_maxsim_weight = (
            trial.suggest_float("knn_maxsim_weight", *r["knn_maxsim_weight"])
            if "knn_maxsim_weight" in r
            else KNN_MAXSIM_WEIGHT
        )

        # Adaptive HN thresholds — old derived from young + fixed gap
        hn_threshold_young = (
            trial.suggest_float("hn_threshold_young", *r["hn_threshold_young"])
            if "hn_threshold_young" in r
            else ADAPTIVE_HN_THRESHOLD_YOUNG
        )
        hn_threshold_old = _derive_hn_threshold_old(hn_threshold_young)

        # HN score normalization
        hn_score_cap = (
            trial.suggest_float("hn_score_cap", *r["hn_score_cap"])
            if "hn_score_cap" in r
            else HN_SCORE_NORMALIZATION_CAP
        )

        # Classifier tuning
        if "classifier_k_feat" in r:
            k_feat_lo, k_feat_hi = int(r["classifier_k_feat"][0]), int(r["classifier_k_feat"][1])
            classifier_k_feat = trial.suggest_int("classifier_k_feat", k_feat_lo, k_feat_hi)
        else:
            classifier_k_feat = CLASSIFIER_K_FEAT
        classifier_neg_sample_weight = (
            trial.suggest_float("classifier_neg_sample_weight", *r["classifier_neg_sample_weight"])
            if "classifier_neg_sample_weight" in r
            else CLASSIFIER_NEG_SAMPLE_WEIGHT
        )

        # Clustering (frozen to stable defaults to reduce search dimensionality)
        cluster_outlier_threshold = CLUSTER_OUTLIER_SIMILARITY_THRESHOLD
        cluster_count = DEFAULT_CLUSTER_COUNT
        cluster_spectral_neighbors = CLUSTER_SPECTRAL_NEIGHBORS

        # Patch the constants in api.rerank
        with patch.multiple(
            api.rerank,
            ADAPTIVE_HN_WEIGHT_MIN=adaptive_hn_min,
            ADAPTIVE_HN_WEIGHT_MAX=adaptive_hn_max,
            ADAPTIVE_HN_THRESHOLD_YOUNG=hn_threshold_young,
            ADAPTIVE_HN_THRESHOLD_OLD=hn_threshold_old,
            HN_SCORE_NORMALIZATION_CAP=hn_score_cap,
            FRESHNESS_MAX_BOOST=freshness_boost,
            FRESHNESS_HALF_LIFE_HOURS=freshness_half_life,
            KNN_SIGMOID_K=knn_sigmoid_k,
            KNN_MAXSIM_WEIGHT=knn_maxsim_weight,
            KNN_NEIGHBORS=knn_k,
            RANKING_DIVERSITY_LAMBDA_CLASSIFIER=classifier_diversity_lambda,
            CLASSIFIER_K_FEAT=classifier_k_feat,
            CLASSIFIER_NEG_SAMPLE_WEIGHT=classifier_neg_sample_weight,
            CLUSTER_OUTLIER_SIMILARITY_THRESHOLD=cluster_outlier_threshold,
            DEFAULT_CLUSTER_COUNT=cluster_count,
            CLUSTER_SPECTRAL_NEIGHBORS=cluster_spectral_neighbors,
        ):
            def report_callback(step: int, interim_metrics: dict[str, float]) -> None:
                score = score_metrics(interim_metrics)
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            metrics = evaluator.evaluate_cv(
                n_folds=args.cv_folds,
                diversity=diversity_lambda,
                knn=knn_k,
                neg_weight=neg_weight,
                use_classifier=True,
                k_metrics=[10, 30, 50],
                report_each=False,
                report_callback=report_callback,
            )

        line = (
            f"Trial {trial.number}: mrr={metrics['mrr']:.3f} "
            f"ndcg@10={metrics['ndcg@10']:.3f} ndcg@30={metrics['ndcg@30']:.3f} "
            f"recall@50={metrics.get('recall@50', 0):.3f}"
        )
        print(line)

        if baseline:
            failures = _guard_metrics(
                metrics,
                baseline,
                guard_metrics=DEFAULT_GUARD_METRICS,
                tolerance=args.guard_tolerance,
            )
            if failures:
                print("  [guard] regressions vs baseline:")
                for msg in failures:
                    print(f"  - {msg}")

        return score_metrics(metrics)

    # 3. Run Optimization
    sampler = TPESampler(
        n_startup_trials=args.startup_trials,
        multivariate=True,
        group=True,
        seed=args.seed,
    )
    if args.no_prune:
        pruner = NopPruner()
    else:
        pruner = MedianPruner(
            n_startup_trials=args.startup_trials,
            n_warmup_steps=1,
            interval_steps=1,
            n_min_trials=5,
        )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Enqueue previous best as first trial if available
    int_params = {"knn_k", "classifier_k_feat"}

    def _normalize_enqueue(raw_params: dict[str, float]) -> dict[str, float | int]:
        enqueue_params: dict[str, float | int] = {}
        for k, v in raw_params.items():
            if k not in ranges:
                continue
            enqueue_params[k] = int(v) if k in int_params else float(v)
        return enqueue_params

    enqueued_signatures: set[str] = set()

    if prev_best:
        enqueue_params = _normalize_enqueue(prev_best)
        if enqueue_params:
            study.enqueue_trial(enqueue_params)
            enqueued_signatures.add(json.dumps(enqueue_params, sort_keys=True))

    # Optionally enqueue additional candidate parameter sets.
    for raw in _load_enqueued_params(args.enqueue_params_file):
        enqueue_params = _normalize_enqueue(raw)
        if not enqueue_params:
            continue
        sig = json.dumps(enqueue_params, sort_keys=True)
        if sig in enqueued_signatures:
            continue
        study.enqueue_trial(enqueue_params)
        enqueued_signatures.add(sig)

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True, n_jobs=args.n_jobs)

    # 4. Report Results
    report_lines: list[str] = []

    def rprint(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    rprint(f"\nOptimization Complete! ({args.trials} trials, {args.cv_folds}-fold CV)")
    rprint(f"Best Combined Score: {study.best_value:.4f}")
    rprint("Best Parameters:")
    for key, value in study.best_params.items():
        rprint(f"  {key}: {value}")

    # Run final evaluation with best params to show all metrics
    best = study.best_params
    best_diversity = float(best.get("diversity_lambda", RANKING_DIVERSITY_LAMBDA))
    best_neg_weight = float(best.get("neg_weight", RANKING_NEGATIVE_WEIGHT))
    best_knn = int(best.get("knn_k", KNN_NEIGHBORS))
    best_adaptive_hn_min = float(best.get("adaptive_hn_min", ADAPTIVE_HN_WEIGHT_MIN))
    best_hn_threshold_young = float(best.get("hn_threshold_young", ADAPTIVE_HN_THRESHOLD_YOUNG))
    best_hn_score_cap = float(best.get("hn_score_cap", HN_SCORE_NORMALIZATION_CAP))
    best_classifier_k_feat = int(best.get("classifier_k_feat", CLASSIFIER_K_FEAT))
    best_classifier_neg_sample_weight = float(best.get("classifier_neg_sample_weight", CLASSIFIER_NEG_SAMPLE_WEIGHT))
    best_classifier_div = _derive_classifier_diversity_lambda(best_diversity)
    best_adaptive_hn_max = _derive_adaptive_hn_max(best_adaptive_hn_min)
    best_hn_threshold_old = _derive_hn_threshold_old(best_hn_threshold_young)
    with patch.multiple(
        api.rerank,
        ADAPTIVE_HN_WEIGHT_MIN=best_adaptive_hn_min,
        ADAPTIVE_HN_WEIGHT_MAX=best_adaptive_hn_max,
        ADAPTIVE_HN_THRESHOLD_YOUNG=best_hn_threshold_young,
        ADAPTIVE_HN_THRESHOLD_OLD=best_hn_threshold_old,
        HN_SCORE_NORMALIZATION_CAP=best_hn_score_cap,
        FRESHNESS_MAX_BOOST=best["freshness_boost"] if "freshness_boost" in best else FRESHNESS_MAX_BOOST,
        FRESHNESS_HALF_LIFE_HOURS=best["freshness_half_life"] if "freshness_half_life" in best else FRESHNESS_HALF_LIFE_HOURS,
        KNN_SIGMOID_K=best["knn_sigmoid_k"] if "knn_sigmoid_k" in best else KNN_SIGMOID_K,
        KNN_MAXSIM_WEIGHT=best["knn_maxsim_weight"] if "knn_maxsim_weight" in best else KNN_MAXSIM_WEIGHT,
        KNN_NEIGHBORS=best_knn,
        RANKING_DIVERSITY_LAMBDA_CLASSIFIER=best_classifier_div,
        CLASSIFIER_K_FEAT=best_classifier_k_feat,
        CLASSIFIER_NEG_SAMPLE_WEIGHT=best_classifier_neg_sample_weight,
        CLUSTER_OUTLIER_SIMILARITY_THRESHOLD=CLUSTER_OUTLIER_SIMILARITY_THRESHOLD,
        DEFAULT_CLUSTER_COUNT=DEFAULT_CLUSTER_COUNT,
        CLUSTER_SPECTRAL_NEIGHBORS=CLUSTER_SPECTRAL_NEIGHBORS,
    ):
        final_metrics = evaluator.evaluate(
            diversity=best_diversity,
            knn=best_knn,
            neg_weight=best_neg_weight,
            use_classifier=True,
            k_metrics=[10, 20, 30, 50]
        )

    rprint("\nFinal Metrics:")
    rprint(f"  MRR: {final_metrics.get('mrr', 0):.3f}")
    for k in [10, 20, 30, 50]:
        rprint(
            f"  @{k}: NDCG={final_metrics.get(f'ndcg@{k}', 0):.3f}, "
            f"MAP={final_metrics.get(f'map@{k}', 0):.3f}, "
            f"P={final_metrics.get(f'precision@{k}', 0):.1%}, "
            f"R={final_metrics.get(f'recall@{k}', 0):.1%}"
        )

    # Write log with timestamp
    log_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nLog saved: {log_path}")

    # Write best params to file for easy reference
    with open("optimized_params.txt", "w") as f:
        f.write(f"Best Combined Score: {study.best_value:.4f}\n")
        f.write(f"MRR: {final_metrics.get('mrr', 0):.3f}\n")
        f.write(f"NDCG@10: {final_metrics.get('ndcg@10', 0):.3f}\n")
        f.write(f"NDCG@30: {final_metrics.get('ndcg@30', 0):.3f}\n")
        f.write(f"Recall@50: {final_metrics.get('recall@50', 0):.3f}\n\n")
        for key, value in study.best_params.items():
            f.write(f"{key} = {value}\n")

    # Also write structured JSON for programmatic consumption
    json_path = log_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "best_score": study.best_value,
        "best_params": study.best_params,
        "final_metrics": final_metrics,
        "n_trials": args.trials,
        "cv_folds": args.cv_folds,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
