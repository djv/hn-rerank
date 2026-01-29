#!/usr/bin/env -S uv run
"""
Optimize HN Rerank hyperparameters using Optuna.
Usage: uv run optimize_hyperparameters.py <username>
"""

import argparse
import asyncio
import logging
import sys
from unittest.mock import patch

import optuna
from optuna.trial import Trial

from evaluate_quality import RankingEvaluator
import api.rerank

# Configure logging to suppress verbose output during optimization
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.INFO)

async def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters")
    parser.add_argument("username", help="HN username")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--candidates", type=int, default=200, help="Candidate pool size")
    args = parser.parse_args()

    # 1. Load Data (Once)
    evaluator = RankingEvaluator(args.username)
    print("Loading data...")
    success = await evaluator.load_data(
        holdout=0.2,
        candidate_count=args.candidates,
        use_classifier=True,
        use_recency=True
    )
    if not success:
        print("Failed to load data.")
        sys.exit(1)

    print("Data loaded. Starting optimization...")

    # 2. Define Objective Function
    def objective(trial: Trial) -> float:
        # Suggest hyperparameters

        # Ranking weights
        diversity_lambda = trial.suggest_float("diversity_lambda", 0.0, 0.6)
        neg_weight = trial.suggest_float("neg_weight", 0.2, 0.8)

        # k-NN (lower k = stricter matching)
        knn_k = trial.suggest_int("knn_k", 1, 5)

        # Adaptive HN Weighting (prefer semantic signal)
        adaptive_hn_min = trial.suggest_float("adaptive_hn_min", 0.0, 0.1)
        adaptive_hn_max = trial.suggest_float("adaptive_hn_max", adaptive_hn_min, 0.15)

        # Freshness (lower boost = less recency bias)
        freshness_boost = trial.suggest_float("freshness_boost", 0.0, 0.15)
        freshness_half_life = trial.suggest_float("freshness_half_life", 24.0, 120.0)

        # Semantic Sigmoid (key for classifier boundary)
        sigmoid_threshold = trial.suggest_float("sigmoid_threshold", 0.3, 0.7)
        sigmoid_k = trial.suggest_float("sigmoid_k", 10.0, 40.0)
        
        # Patch the constants in api.rerank
        with patch.multiple(
            api.rerank,
            ADAPTIVE_HN_WEIGHT_MIN=adaptive_hn_min,
            ADAPTIVE_HN_WEIGHT_MAX=adaptive_hn_max,
            FRESHNESS_MAX_BOOST=freshness_boost,
            FRESHNESS_HALF_LIFE_HOURS=freshness_half_life,
            SEMANTIC_SIGMOID_THRESHOLD=sigmoid_threshold,
            SEMANTIC_SIGMOID_K=sigmoid_k,
            KNN_NEIGHBORS=knn_k,
        ):
            metrics = evaluator.evaluate(
                diversity=diversity_lambda,
                knn=knn_k,
                neg_weight=neg_weight,
                use_classifier=True,
                k_metrics=[10, 30]
            )

        # Combined objective: NDCG + MAP (rewards both ranking quality and precision)
        ndcg_30 = metrics["ndcg@30"]
        map_30 = metrics["map@30"]
        ndcg_10 = metrics["ndcg@10"]

        # Weight early precision higher (NDCG@10) + overall ranking (NDCG@30) + precision (MAP@30)
        return 0.3 * ndcg_10 + 0.4 * ndcg_30 + 0.3 * map_30

    # 3. Run Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    # 4. Report Results
    print("\nOptimization Complete!")
    print(f"Best Combined Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Run final evaluation with best params to show all metrics
    best = study.best_params
    with patch.multiple(
        api.rerank,
        ADAPTIVE_HN_WEIGHT_MIN=best["adaptive_hn_min"],
        ADAPTIVE_HN_WEIGHT_MAX=best["adaptive_hn_max"],
        FRESHNESS_MAX_BOOST=best["freshness_boost"],
        FRESHNESS_HALF_LIFE_HOURS=best["freshness_half_life"],
        SEMANTIC_SIGMOID_THRESHOLD=best["sigmoid_threshold"],
        SEMANTIC_SIGMOID_K=best["sigmoid_k"],
        KNN_NEIGHBORS=best["knn_k"],
    ):
        final_metrics = evaluator.evaluate(
            diversity=best["diversity_lambda"],
            knn=best["knn_k"],
            neg_weight=best["neg_weight"],
            use_classifier=True,
            k_metrics=[10, 20, 30, 50]
        )

    print("\nFinal Metrics:")
    print(f"  MRR: {final_metrics.get('mrr', 0):.3f}")
    for k in [10, 20, 30, 50]:
        print(f"  @{k}: NDCG={final_metrics.get(f'ndcg@{k}', 0):.3f}, MAP={final_metrics.get(f'map@{k}', 0):.3f}, P={final_metrics.get(f'precision@{k}', 0):.1%}, R={final_metrics.get(f'recall@{k}', 0):.1%}")

    # Write best params to file for easy reference
    with open("optimized_params.txt", "w") as f:
        f.write(f"Best Combined Score: {study.best_value:.4f}\n")
        f.write(f"MRR: {final_metrics.get('mrr', 0):.3f}\n")
        f.write(f"NDCG@30: {final_metrics.get('ndcg@30', 0):.3f}\n")
        f.write(f"MAP@30: {final_metrics.get('map@30', 0):.3f}\n\n")
        for key, value in study.best_params.items():
            f.write(f"{key} = {value}\n")

if __name__ == "__main__":
    asyncio.run(main())
