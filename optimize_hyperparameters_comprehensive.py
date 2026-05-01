#!/usr/bin/env -S uv run
"""
Comprehensive HN Rerank hyperparameter optimization using Optuna.
Optimizes both first-stage (Bi-Encoder) and second-stage (Cross-Encoder) parameters.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.trial import Trial, TrialState
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from evaluate_quality import RankingEvaluator, _print_metrics_report
from api.config import AppConfig
from api import rerank
import tuning_common as _tuning_common

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.INFO)

async def main():
    parser = argparse.ArgumentParser(description="Comprehensive Optuna Sweep")
    parser.add_argument("username", help="HN username")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--candidates", type=int, default=1000, help="Candidate pool size")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs (keep low for model memory)")
    parser.add_argument("--pure-semantic", action="store_true", help="Optimize for pure content quality (fair)")
    parser.add_argument("--age-matched", action="store_true", help="Use age-matched candidates for fairness")
    parser.add_argument("--cache-only", action="store_true", default=True, help="Use cached data only")
    args = parser.parse_args()

    rerank.init_model()
    # Cross-encoder will be lazy-initialized in rank_stories if needed

    evaluator = RankingEvaluator(args.username)
    print(f"Loading data for {args.username}...")
    success = await evaluator.load_data(
        candidate_count=args.candidates,
        use_classifier=True,
        cache_only=args.cache_only,
        allow_stale=args.cache_only,
        age_matched=args.age_matched
    )
    if not success:
        print("Failed to load data.")
        return

    def objective(trial: Trial) -> float:
        # 1. First Stage: Classifier & Ranking
        params = {
            "neg_weight": trial.suggest_float("neg_weight", 0.1, 1.0),
            "diversity_lambda": trial.suggest_float("diversity_lambda", 0.0, 0.6),
            "pairwise_c": trial.suggest_float("pairwise_c", 0.01, 10.0, log=True),
            "pairwise_negatives": trial.suggest_int("pairwise_negatives", 5, 30),
            "classifier_k_feat": trial.suggest_int("classifier_k_feat", 3, 15),
            "cluster_distance_threshold": trial.suggest_float("cluster_distance_threshold", 0.4, 1.8),
        }

        # 2. Second Stage: Cross-Encoder
        params.update({
            "ce_enabled": trial.suggest_categorical("ce_enabled", [True, False]),
            "ce_top_n": trial.suggest_int("ce_top_n", 20, 100),
            "ce_weight": trial.suggest_float("ce_weight", 0.1, 0.9),
        })

        # 3. Adaptive HN (only if not pure semantic)
        if not args.pure_semantic:
            params.update({
                "adaptive_hn_min": trial.suggest_float("adaptive_hn_min", 0.1, 0.6),
                "hn_threshold_young": trial.suggest_float("hn_threshold_young", 10.0, 120.0),
                "hn_score_cap": trial.suggest_float("hn_score_cap", 100.0, 800.0),
                "freshness_boost": trial.suggest_float("freshness_boost", 0.0, 0.3),
            })

        with _tuning_common.tuned_config(params) as (config, resolved):
            # Override for pure semantic if requested
            from dataclasses import replace
            if args.pure_semantic:
                config = replace(config, 
                    ranking=replace(config.ranking, hn_weight=0.0),
                    freshness=replace(config.freshness, enabled=False)
                )

            metrics = evaluator.evaluate_cv(
                n_folds=args.cv_folds,
                config=config,
                k_metrics=[10, 20, 30],
                report_each=False
            )
            
            # Primary optimization target: NDCG@10 + MRR
            score = (metrics["ndcg@10"] * 0.6) + (metrics["mrr"] * 0.4)
            
            trial.set_user_attr("metrics", metrics)
            return float(score)

    storage_name = "sqlite:///optuna_sweep.db"
    study = optuna.create_study(
        study_name="comprehensive_sweep",
        storage=storage_name,
        load_if_exists=True,
        direction="maximize", 
        sampler=TPESampler(seed=42)
    )
    
    # Enqueue current production params as first trial
    current_params = {
        "neg_weight": 0.6,
        "pairwise_c": 1.47,
        "pairwise_negatives": 15,
        "classifier_k_feat": 7,
        "cluster_distance_threshold": 1.328,
        "ce_enabled": True,
        "ce_top_n": 50,
        "ce_weight": 0.5,
    }
    study.enqueue_trial(current_params)

    print(f"Starting sweep: {args.trials} trials...")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    print("\nSweep Complete!")
    print(f"Best Score: {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best results
    best_metrics = study.best_trial.user_attrs["metrics"]
    output = {
        "best_params": study.best_params,
        "best_metrics": best_metrics,
        "config": {
            "pure_semantic": args.pure_semantic,
            "age_matched": args.age_matched,
            "candidates": args.candidates,
        }
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"optuna_comprehensive_{ts}.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {out_path}")

    # Generate promoted TOML
    resolved_best = _tuning_common.resolve_params(study.best_params)
    promoted_toml = _tuning_common.render_promoted_toml(resolved_best)
    print("\nPromoted TOML Configuration:")
    print(promoted_toml)

if __name__ == "__main__":
    asyncio.run(main())
