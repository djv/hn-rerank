from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, TypeAlias
from unittest.mock import patch

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

OBJECTIVE_WEIGHTS: dict[str, float] = {
    "mrr": 0.30,
    "ndcg@10": 0.35,
    "ndcg@30": 0.20,
    "recall@50": 0.15,
}

HN_THRESHOLD_GAP = 42.0
ADAPTIVE_HN_DELTA = 0.035

ResolvedSection: TypeAlias = dict[str, float | int]
ResolvedParams: TypeAlias = dict[str, ResolvedSection]


def derive_classifier_diversity_lambda(diversity_lambda: float) -> float:
    return max(diversity_lambda, 0.30)


def derive_hn_threshold_old(hn_threshold_young: float) -> float:
    return hn_threshold_young + HN_THRESHOLD_GAP


def derive_adaptive_hn_max(adaptive_hn_base: float) -> float:
    return adaptive_hn_base + ADAPTIVE_HN_DELTA


def score_metrics(
    metrics: Mapping[str, float],
    *,
    std_penalty: float,
    weights: Mapping[str, float] = OBJECTIVE_WEIGHTS,
) -> float:
    mean = sum(weight * metrics.get(key, 0.0) for key, weight in weights.items())
    std = sum(
        weight * metrics.get(f"{key}_std", 0.0) for key, weight in weights.items()
    )
    return float(mean - std_penalty * std)


def resolve_params(params: Mapping[str, float | int]) -> ResolvedParams:
    diversity_lambda = float(params.get("diversity_lambda", RANKING_DIVERSITY_LAMBDA))
    adaptive_hn_min = float(params.get("adaptive_hn_min", ADAPTIVE_HN_WEIGHT_MIN))
    threshold_young = float(
        params.get("hn_threshold_young", ADAPTIVE_HN_THRESHOLD_YOUNG)
    )

    return {
        "ranking": {
            "negative_weight": float(
                params.get("neg_weight", RANKING_NEGATIVE_WEIGHT)
            ),
            "diversity_lambda": diversity_lambda,
            "diversity_lambda_classifier": derive_classifier_diversity_lambda(
                diversity_lambda
            ),
        },
        "adaptive_hn": {
            "weight_min": adaptive_hn_min,
            "weight_max": derive_adaptive_hn_max(adaptive_hn_min),
            "threshold_young": threshold_young,
            "threshold_old": derive_hn_threshold_old(threshold_young),
            "score_normalization_cap": float(
                params.get("hn_score_cap", HN_SCORE_NORMALIZATION_CAP)
            ),
        },
        "freshness": {
            "half_life_hours": float(
                params.get("freshness_half_life", FRESHNESS_HALF_LIFE_HOURS)
            ),
            "max_boost": float(params.get("freshness_boost", FRESHNESS_MAX_BOOST)),
        },
        "semantic": {
            "knn_sigmoid_k": float(params.get("knn_sigmoid_k", KNN_SIGMOID_K)),
            "knn_maxsim_weight": float(
                params.get("knn_maxsim_weight", KNN_MAXSIM_WEIGHT)
            ),
            "knn_neighbors": int(round(float(params.get("knn_k", KNN_NEIGHBORS)))),
        },
        "classifier": {
            "k_feat": int(round(float(params.get("classifier_k_feat", CLASSIFIER_K_FEAT)))),
            "neg_sample_weight": float(
                params.get(
                    "classifier_neg_sample_weight", CLASSIFIER_NEG_SAMPLE_WEIGHT
                )
            ),
        },
    }


def build_patch_kwargs(resolved: ResolvedParams) -> dict[str, Any]:
    ranking = resolved["ranking"]
    adaptive_hn = resolved["adaptive_hn"]
    freshness = resolved["freshness"]
    semantic = resolved["semantic"]
    classifier = resolved["classifier"]
    return {
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
        "CLUSTER_OUTLIER_SIMILARITY_THRESHOLD": CLUSTER_OUTLIER_SIMILARITY_THRESHOLD,
        "DEFAULT_CLUSTER_COUNT": DEFAULT_CLUSTER_COUNT,
        "CLUSTER_SPECTRAL_NEIGHBORS": CLUSTER_SPECTRAL_NEIGHBORS,
    }


@contextmanager
def patched_rerank_params(
    params: Mapping[str, float | int],
) -> Iterator[ResolvedParams]:
    resolved = resolve_params(params)
    with patch.multiple(api.rerank, **build_patch_kwargs(resolved)):
        yield resolved


def render_promoted_toml(resolved: ResolvedParams) -> str:
    ranking = resolved["ranking"]
    adaptive_hn = resolved["adaptive_hn"]
    freshness = resolved["freshness"]
    semantic = resolved["semantic"]
    classifier = resolved["classifier"]
    return (
        "# Auto-generated promoted params.\n"
        "# Merge this into hn_rerank.toml under [hn_rerank.*] sections.\n\n"
        "[hn_rerank.ranking]\n"
        f"negative_weight = {float(ranking['negative_weight']):.10f}\n"
        f"diversity_lambda = {float(ranking['diversity_lambda']):.10f}\n"
        f"diversity_lambda_classifier = "
        f"{float(ranking['diversity_lambda_classifier']):.10f}\n\n"
        "[hn_rerank.adaptive_hn]\n"
        f"weight_min = {float(adaptive_hn['weight_min']):.10f}\n"
        f"weight_max = {float(adaptive_hn['weight_max']):.10f}\n"
        f"threshold_young = {float(adaptive_hn['threshold_young']):.10f}\n"
        f"threshold_old = {float(adaptive_hn['threshold_old']):.10f}\n"
        f"score_normalization_cap = "
        f"{float(adaptive_hn['score_normalization_cap']):.10f}\n\n"
        "[hn_rerank.freshness]\n"
        f"half_life_hours = {float(freshness['half_life_hours']):.10f}\n"
        f"max_boost = {float(freshness['max_boost']):.10f}\n\n"
        "[hn_rerank.semantic]\n"
        f"knn_sigmoid_k = {float(semantic['knn_sigmoid_k']):.10f}\n"
        f"knn_maxsim_weight = {float(semantic['knn_maxsim_weight']):.10f}\n"
        f"knn_neighbors = {int(semantic['knn_neighbors'])}\n\n"
        "[hn_rerank.classifier]\n"
        f"k_feat = {int(classifier['k_feat'])}\n"
        f"neg_sample_weight = {float(classifier['neg_sample_weight']):.10f}\n"
    )
