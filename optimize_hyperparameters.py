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
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import numpy as np

TRAIN_EXTRA_HINT = (
    "optimize_hyperparameters.py requires the 'train' extra. "
    "Run: uv sync --extra train"
)

try:
    import optuna
    from optuna.trial import Trial
    from optuna.trial import TrialState
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, NopPruner
except ModuleNotFoundError as exc:
    raise SystemExit(TRAIN_EXTRA_HINT) from exc


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

from evaluate_quality import (  # noqa: E402
    RankingEvaluator,
    DEFAULT_GUARD_METRICS,
    _guard_metrics,
    _load_baseline,
    _merge_rank_diagnostic_summaries,
)
from api.constants import (  # noqa: E402
    ADAPTIVE_HN_THRESHOLD_YOUNG,
    ADAPTIVE_HN_WEIGHT_MIN,
    CLASSIFIER_K_FEAT,
    CLASSIFIER_NEG_SAMPLE_WEIGHT,
    FRESHNESS_MAX_BOOST,
    FRESHNESS_HALF_LIFE_HOURS,
    HN_SCORE_NORMALIZATION_CAP,
    KNN_NEIGHBORS,
)
import tuning_common as _tuning_common  # noqa: E402

HN_THRESHOLD_GAP = _tuning_common.HN_THRESHOLD_GAP
ADAPTIVE_HN_DELTA = _tuning_common.ADAPTIVE_HN_DELTA
_derive_adaptive_hn_max = _tuning_common.derive_adaptive_hn_max
_derive_classifier_diversity_lambda = _tuning_common.derive_classifier_diversity_lambda
_derive_hn_threshold_old = _tuning_common.derive_hn_threshold_old
patched_rerank_params = _tuning_common.patched_rerank_params
_score_metrics = _tuning_common.score_metrics
_average_seed_metrics = _tuning_common.average_seed_metrics
_validate_candidate_metrics = _tuning_common.validate_candidate_metrics

# Configure logging to suppress verbose output during optimization
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.INFO)


def _diag_float(summary: dict[str, object], key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _diag_int(summary: dict[str, object], key: str, default: int = 0) -> int:
    value = summary.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


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
                with suppress(ValueError):
                    best_params[key.strip()] = float(val.strip())

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
            with suppress(TypeError, ValueError):
                parsed[str(key)] = float(val)
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
        "knn_k": (1, 4),
        "adaptive_hn_min": (0.0, 0.07),
        "adaptive_hn_max": (0.02, 0.12),
        "freshness_boost": (0.04, 0.15),
        "freshness_half_life": (45.0, 100.0),
        "hn_threshold_young": (4.0, 16.0),
        "hn_score_cap": (120.0, 1600.0),
        "classifier_k_feat": (1, 9),
        "classifier_neg_sample_weight": (0.7, 2.0),
    }
    spaces: dict[str, set[str]] = {
        "full": set(full_defaults.keys()),
        "core": set(full_defaults.keys()) - {"freshness_boost"},
        "cat_relevance": {
            "knn_k",
            "classifier_k_feat",
            "classifier_neg_sample_weight",
        },
        "cat_freshness": {"freshness_boost", "freshness_half_life"},
        "cat_semantic": {"knn_k"},
        "cat_hn": {"adaptive_hn_min", "hn_threshold_young", "hn_score_cap"},
        "cat_hn_decoupled": {
            "adaptive_hn_min",
            "adaptive_hn_max",
            "hn_threshold_young",
            "hn_score_cap",
        },
    }
    tuned_keys = spaces.get(space)
    if tuned_keys is None:
        raise ValueError(f"Unknown space: {space}")
    defaults = {k: v for k, v in full_defaults.items() if k in tuned_keys}
    integer_keys = {"knn_k", "classifier_k_feat"}

    if prev is None:
        return defaults

    # Narrow ranges to ±20% around previous best, clamped to defaults.
    narrowed: dict[str, tuple[float, float]] = {}
    for key, (lo, hi) in defaults.items():
        if key in prev:
            v = prev[key]
            if v < lo or v > hi:
                narrowed[key] = (lo, hi)
                continue
            if key in integer_keys:
                center = int(round(v))
                radius = max(1, int((hi - lo) * 0.2))
                narrowed[key] = (
                    float(max(lo, center - radius)),
                    float(min(hi, center + radius)),
                )
                continue
            span = (hi - lo) * 0.2
            narrowed[key] = (max(lo, v - span), min(hi, v + span))
        else:
            narrowed[key] = (lo, hi)
    return narrowed
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


async def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters")
    parser.add_argument("username", help="HN username")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Frozen benchmark snapshot to optimize against instead of loading live data.",
    )
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
    parser.add_argument(
        "--std-penalty",
        type=float,
        default=0.5,
        help="Penalty multiplier for weighted metric std in objective",
    )
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
        choices=[
            "core",
            "full",
            "cat_relevance",
            "cat_freshness",
            "cat_semantic",
            "cat_hn",
            "cat_hn_decoupled",
        ],
        default="core",
        help="Search-space size: core tunes fewer high-impact knobs; full tunes all non-cluster knobs.",
    )
    parser.add_argument(
        "--top-trials-json-limit",
        type=int,
        default=50,
        help="How many top completed trials to include in output JSON",
    )
    parser.add_argument(
        "--final-list",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize the final displayed list instead of raw rank_stories output.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="Displayed story count when final-list mode is enabled.",
    )
    parser.add_argument(
        "--validation-seeds",
        default="0,1,2",
        help="Comma-separated seeds for incumbent-vs-candidate validation.",
    )
    parser.add_argument(
        "--validation-guard-tolerance",
        type=float,
        default=0.0,
        help="Allowed drop on validation guard metrics versus current config.",
    )
    parser.add_argument(
        "--validation-score-tolerance",
        type=float,
        default=0.0,
        help="Minimum score delta required for a candidate to be promotable.",
    )
    args = parser.parse_args()
    if args.top_trials_json_limit < 1:
        raise SystemExit("--top-trials-json-limit must be >= 1")
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    validation_seeds = _parse_seed_list(args.validation_seeds)

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
        print("Failed to load data.")
        sys.exit(1)

    print(f"Data loaded. Starting {args.trials}-trial optimization "
          f"({args.cv_folds}-fold CV, {args.n_jobs} workers)...")

    baseline = _load_baseline(args.baseline)

    # 2. Define Objective Function
    def objective(trial: Trial) -> float:
        r = ranges  # alias

        # k-NN
        if "knn_k" in r:
            knn_lo, knn_hi = int(r["knn_k"][0]), int(r["knn_k"][1])
            knn_k = trial.suggest_int("knn_k", knn_lo, knn_hi)
        else:
            knn_k = KNN_NEIGHBORS

        adaptive_hn_min = (
            trial.suggest_float("adaptive_hn_min", *r["adaptive_hn_min"])
            if "adaptive_hn_min" in r
            else ADAPTIVE_HN_WEIGHT_MIN
        )
        adaptive_hn_max = (
            trial.suggest_float(
                "adaptive_hn_max",
                max(adaptive_hn_min, r["adaptive_hn_max"][0]),
                r["adaptive_hn_max"][1],
            )
            if "adaptive_hn_max" in r
            else _derive_adaptive_hn_max(adaptive_hn_min)
        )
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

        hn_threshold_young = (
            trial.suggest_float("hn_threshold_young", *r["hn_threshold_young"])
            if "hn_threshold_young" in r
            else ADAPTIVE_HN_THRESHOLD_YOUNG
        )
        hn_score_cap = (
            trial.suggest_float("hn_score_cap", *r["hn_score_cap"])
            if "hn_score_cap" in r
            else HN_SCORE_NORMALIZATION_CAP
        )

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

        trial_params = {
            "knn_k": knn_k,
            "adaptive_hn_min": adaptive_hn_min,
            "adaptive_hn_max": adaptive_hn_max,
            "freshness_boost": freshness_boost,
            "freshness_half_life": freshness_half_life,
            "hn_threshold_young": hn_threshold_young,
            "hn_score_cap": hn_score_cap,
            "classifier_k_feat": classifier_k_feat,
            "classifier_neg_sample_weight": classifier_neg_sample_weight,
        }

        with patched_rerank_params(trial_params):
            resolved = _tuning_common.resolve_params(trial_params)
            ranking = resolved["ranking"]
            semantic = resolved["semantic"]
            def report_callback(step: int, interim_metrics: dict[str, float]) -> None:
                score = _score_metrics(interim_metrics, std_penalty=args.std_penalty)
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            metrics = evaluator.evaluate_cv(
                n_folds=args.cv_folds,
                diversity=float(ranking["diversity_lambda"]),
                knn=int(semantic["knn_neighbors"]),
                neg_weight=float(ranking["negative_weight"]),
                use_classifier=True,
                k_metrics=[10, 20, 30, 40],
                report_each=False,
                report_callback=report_callback,
                final_list_count=args.count if args.final_list else None,
            )
            trial.set_user_attr(
                "cv_metrics",
                {
                    str(key): float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
            )

        line = (
            f"Trial {trial.number}: mrr={metrics['mrr']:.3f} "
            f"ndcg@10={metrics['ndcg@10']:.3f} ndcg@20={metrics['ndcg@20']:.3f} "
            f"precision@20={metrics.get('precision@20', 0):.3f}"
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

        return _score_metrics(metrics, std_penalty=args.std_penalty)

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
            lo, hi = ranges[k]
            if k in int_params:
                enqueue_params[k] = max(int(lo), min(int(hi), int(round(v))))
            else:
                enqueue_params[k] = max(lo, min(hi, float(v)))
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

    def _evaluate_seeded_metrics(
        params: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, object]]:
        per_seed: list[dict[str, float]] = []
        per_seed_diagnostics: list[dict[str, object]] = []
        with patched_rerank_params(params) as resolved:
            ranking = resolved["ranking"]
            semantic = resolved["semantic"]
            for validation_seed in validation_seeds:
                np.random.seed(validation_seed)
                diagnostics_summary: dict[str, object] = {}
                per_seed.append(
                    evaluator.evaluate_cv(
                        n_folds=args.cv_folds,
                        diversity=float(ranking["diversity_lambda"]),
                        knn=int(semantic["knn_neighbors"]),
                        neg_weight=float(ranking["negative_weight"]),
                        use_classifier=True,
                        k_metrics=[10, 20, 30, 40],
                        report_each=False,
                        parallel=False,
                        final_list_count=args.count if args.final_list else None,
                        diagnostics_summary=diagnostics_summary,
                    )
                )
                per_seed_diagnostics.append(diagnostics_summary)
        return (
            _average_seed_metrics(per_seed),
            _merge_rank_diagnostic_summaries(per_seed_diagnostics),
        )

    best = study.best_params
    candidate_metrics, candidate_diagnostics = _evaluate_seeded_metrics(best)
    current_metrics, current_diagnostics = _evaluate_seeded_metrics({})
    validation = _validate_candidate_metrics(
        candidate_metrics,
        current_metrics,
        std_penalty=args.std_penalty,
        score_tolerance=args.validation_score_tolerance,
        guard_tolerance=args.validation_guard_tolerance,
    )

    rprint("\nCandidate Validation:")
    rprint(f"  Promotable: {validation['promotable']}")
    rprint(
        f"  Score Delta: {validation['score_delta']:.4f} "
        f"(candidate={validation['candidate_score']:.4f}, current={validation['incumbent_score']:.4f})"
    )
    if validation["primary_failures"]:
        rprint(f"  Primary metric regressions: {', '.join(validation['primary_failures'])}")
    if validation["guard_failures"]:
        rprint(f"  Guard metric regressions: {', '.join(validation['guard_failures'])}")

    rprint("\nCandidate Metrics:")
    rprint(f"  MRR: {candidate_metrics.get('mrr', 0):.3f}")
    for k in [10, 20, 30, 40]:
        rprint(
            f"  @{k}: NDCG={candidate_metrics.get(f'ndcg@{k}', 0):.3f}, "
            f"MAP={candidate_metrics.get(f'map@{k}', 0):.3f}, "
            f"P={candidate_metrics.get(f'precision@{k}', 0):.1%}, "
            f"R={candidate_metrics.get(f'recall@{k}', 0):.1%}"
        )
    if candidate_diagnostics:
        candidate_classifier_used_rate = _diag_float(
            candidate_diagnostics, "classifier_used_rate"
        )
        candidate_classifier_fallback_count = _diag_int(
            candidate_diagnostics, "classifier_fallback_count"
        )
        candidate_avg_derived_dim = _diag_float(
            candidate_diagnostics, "avg_derived_feature_dim"
        )
        candidate_rank_calls = max(_diag_int(candidate_diagnostics, "rank_calls", 1), 1)
        candidate_penalty_applied_count = _diag_float(
            candidate_diagnostics, "local_hidden_penalty_applied_count"
        )
        candidate_local_hidden_penalty_rate = (
            candidate_penalty_applied_count / candidate_rank_calls
        )
        rprint(
            "  Diagnostics: "
            f"classifier_used_rate={candidate_classifier_used_rate:.1%}, "
            f"fallbacks={candidate_classifier_fallback_count}, "
            f"avg_derived_dim={candidate_avg_derived_dim:.1f}, "
            f"local_hidden_penalty_rate={candidate_local_hidden_penalty_rate:.1%}"
        )

    rprint("\nCurrent Metrics:")
    rprint(f"  MRR: {current_metrics.get('mrr', 0):.3f}")
    for k in [10, 20, 30, 40]:
        rprint(
            f"  @{k}: NDCG={current_metrics.get(f'ndcg@{k}', 0):.3f}, "
            f"MAP={current_metrics.get(f'map@{k}', 0):.3f}, "
            f"P={current_metrics.get(f'precision@{k}', 0):.1%}, "
            f"R={current_metrics.get(f'recall@{k}', 0):.1%}"
        )
    if current_diagnostics:
        current_classifier_used_rate = _diag_float(
            current_diagnostics, "classifier_used_rate"
        )
        current_classifier_fallback_count = _diag_int(
            current_diagnostics, "classifier_fallback_count"
        )
        current_avg_derived_dim = _diag_float(
            current_diagnostics, "avg_derived_feature_dim"
        )
        current_rank_calls = max(_diag_int(current_diagnostics, "rank_calls", 1), 1)
        current_penalty_applied_count = _diag_float(
            current_diagnostics, "local_hidden_penalty_applied_count"
        )
        current_local_hidden_penalty_rate = (
            current_penalty_applied_count / current_rank_calls
        )
        rprint(
            "  Diagnostics: "
            f"classifier_used_rate={current_classifier_used_rate:.1%}, "
            f"fallbacks={current_classifier_fallback_count}, "
            f"avg_derived_dim={current_avg_derived_dim:.1f}, "
            f"local_hidden_penalty_rate={current_local_hidden_penalty_rate:.1%}"
        )

    # Write log with timestamp
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nLog saved: {log_path}")

    # Write best params to file for easy reference
    with open("optimized_params.txt", "w") as f:
        f.write(f"Best Combined Score: {study.best_value:.4f}\n")
        f.write(f"Promotable: {validation['promotable']}\n")
        f.write(f"Validation Score Delta: {validation['score_delta']:.4f}\n")
        f.write(f"MRR: {candidate_metrics.get('mrr', 0):.3f}\n")
        f.write(f"NDCG@10: {candidate_metrics.get('ndcg@10', 0):.3f}\n")
        f.write(f"NDCG@20: {candidate_metrics.get('ndcg@20', 0):.3f}\n")
        f.write(f"Precision@20: {candidate_metrics.get('precision@20', 0):.3f}\n\n")
        for key, value in study.best_params.items():
            f.write(f"{key} = {value}\n")

    # Also write structured JSON for programmatic consumption
    json_path = log_path.with_suffix(".json")
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    completed_trials.sort(key=lambda trial: float(trial.value), reverse=True)
    top_trials = []
    for trial in completed_trials[: args.top_trials_json_limit]:
        entry = {
            "number": int(trial.number),
            "value": float(trial.value),
            "params": trial.params,
        }
        cv_metrics = trial.user_attrs.get("cv_metrics")
        if isinstance(cv_metrics, dict):
            entry["cv_metrics"] = cv_metrics
        top_trials.append(entry)

    json_path.write_text(json.dumps({
        "best_score": study.best_value,
        "best_params": study.best_params,
        "candidate_metrics": candidate_metrics,
        "candidate_diagnostics": candidate_diagnostics,
        "current_metrics": current_metrics,
        "current_diagnostics": current_diagnostics,
        "validation": validation,
        "n_trials": args.trials,
        "cv_folds": args.cv_folds,
        "std_penalty": args.std_penalty,
        "snapshot": None if args.snapshot is None else str(args.snapshot),
        "final_list": args.final_list,
        "count": args.count,
        "validation_seeds": validation_seeds,
        "completed_trial_count": len(completed_trials),
        "top_trials_limit": args.top_trials_json_limit,
        "top_trials": top_trials,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
