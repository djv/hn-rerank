from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeAlias

from api.config import (
    AppConfig,
)
from api.constants import (
    ADAPTIVE_HN_THRESHOLD_YOUNG,
    ADAPTIVE_HN_WEIGHT_MIN,
    CLASSIFIER_K_FEAT,
    CLASSIFIER_USE_BALANCED_CLASS_WEIGHT,
    FRESHNESS_HALF_LIFE_HOURS,
    FRESHNESS_MAX_BOOST,
    HN_SCORE_NORMALIZATION_CAP,
    KNN_NEIGHBORS,
    RANKING_COMMENT_RATIO,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_NEGATIVE_WEIGHT,
    RANKING_NON_SEMANTIC_WEIGHT,
    CLASSIFIER_SCORING_MODE,
    CLASSIFIER_FEATURE_MODE,
    CLASSIFIER_PAIRWISE_NEGATIVES,
    CLASSIFIER_PAIRWISE_C,
    CLASSIFIER_USE_LOG_POINTS_FEATURE,
    CLASSIFIER_USE_LOG_COMMENTS_FEATURE,
    CLASSIFIER_USE_COMMENT_RATIO_FEATURE,
    CLUSTER_AGGLOMERATIVE_THRESHOLD,
    CROSS_ENCODER_ENABLED,
    CROSS_ENCODER_TOP_N,
    CROSS_ENCODER_WEIGHT,
)

OBJECTIVE_WEIGHTS: dict[str, float] = {
    "mrr": 0.40,
    "ndcg@10": 0.40,
    "ndcg@20": 0.20,
}
VALIDATION_PRIMARY_METRICS: tuple[str, ...] = ("mrr", "ndcg@10", "ndcg@20")
VALIDATION_GUARD_METRICS: tuple[str, ...] = ("ndcg@30", "precision@20", "recall@30")

HN_THRESHOLD_GAP = 42.0
ADAPTIVE_HN_DELTA = 0.035

ResolvedSection: TypeAlias = dict[str, float | int | str]
ResolvedParams: TypeAlias = dict[str, ResolvedSection]
ValidationResult: TypeAlias = dict[str, Any]


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


def average_seed_metrics(
    metrics_per_seed: Sequence[Mapping[str, float | int]],
) -> dict[str, float]:
    if not metrics_per_seed:
        return {}
    keys = metrics_per_seed[0].keys()
    return {
        str(key): float(
            sum(float(metrics[key]) for metrics in metrics_per_seed)
            / len(metrics_per_seed)
        )
        for key in keys
    }


def validate_candidate_metrics(
    candidate_metrics: Mapping[str, float],
    incumbent_metrics: Mapping[str, float],
    *,
    std_penalty: float,
    score_tolerance: float = 0.0,
    guard_tolerance: float = 0.0,
    primary_metrics: tuple[str, ...] = VALIDATION_PRIMARY_METRICS,
    guard_metrics: tuple[str, ...] = VALIDATION_GUARD_METRICS,
) -> ValidationResult:
    candidate_score = score_metrics(candidate_metrics, std_penalty=std_penalty)
    incumbent_score = score_metrics(incumbent_metrics, std_penalty=std_penalty)
    score_delta = float(candidate_score - incumbent_score)

    primary_failures: list[str] = []
    for metric in primary_metrics:
        cand = float(candidate_metrics.get(metric, 0.0))
        inc = float(incumbent_metrics.get(metric, 0.0))
        if cand < inc - score_tolerance:
            primary_failures.append(metric)

    guard_failures: list[str] = []
    for metric in guard_metrics:
        cand = float(candidate_metrics.get(metric, 0.0))
        inc = float(incumbent_metrics.get(metric, 0.0))
        if cand < inc - guard_tolerance:
            guard_failures.append(metric)

    promotable = (
        score_delta > score_tolerance and not primary_failures and not guard_failures
    )

    metric_deltas = {
        metric: float(
            candidate_metrics.get(metric, 0.0) - incumbent_metrics.get(metric, 0.0)
        )
        for metric in sorted(set(primary_metrics) | set(guard_metrics))
    }

    return {
        "promotable": promotable,
        "candidate_score": candidate_score,
        "incumbent_score": incumbent_score,
        "score_delta": score_delta,
        "primary_failures": primary_failures,
        "guard_failures": guard_failures,
        "metric_deltas": metric_deltas,
    }


def resolve_params(params: Mapping[str, float | int]) -> ResolvedParams:
    diversity_lambda = float(params.get("diversity_lambda", RANKING_DIVERSITY_LAMBDA))
    adaptive_hn_min = float(params.get("adaptive_hn_min", ADAPTIVE_HN_WEIGHT_MIN))
    adaptive_hn_max = float(
        params.get("adaptive_hn_max", derive_adaptive_hn_max(adaptive_hn_min))
    )
    adaptive_hn_max = max(adaptive_hn_min, adaptive_hn_max)
    threshold_young = float(
        params.get("hn_threshold_young", ADAPTIVE_HN_THRESHOLD_YOUNG)
    )

    return {
        "ranking": {
            "negative_weight": float(params.get("neg_weight", RANKING_NEGATIVE_WEIGHT)),
            "diversity_lambda": diversity_lambda,
            "diversity_lambda_classifier": derive_classifier_diversity_lambda(
                diversity_lambda
            ),
            "non_semantic_weight": float(
                params.get("non_semantic_weight", RANKING_NON_SEMANTIC_WEIGHT)
            ),
            "comment_ratio": float(
                params.get("comment_ratio", RANKING_COMMENT_RATIO)
            ),
        },
        "adaptive_hn": {
            "weight_min": adaptive_hn_min,
            "weight_max": adaptive_hn_max,
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
            "knn_neighbors": int(round(float(params.get("knn_k", KNN_NEIGHBORS)))),
        },
        "classifier": {
            "scoring_mode": str(params.get("scoring_mode", CLASSIFIER_SCORING_MODE)),
            "feature_mode": str(params.get("feature_mode", CLASSIFIER_FEATURE_MODE)),
            "pairwise_negatives": int(round(float(params.get("pairwise_negatives", CLASSIFIER_PAIRWISE_NEGATIVES)))),
            "pairwise_c": float(params.get("pairwise_c", CLASSIFIER_PAIRWISE_C)),
            "k_feat": int(
                round(float(params.get("classifier_k_feat", CLASSIFIER_K_FEAT)))
            ),
            "use_balanced_class_weight": bool(
                params.get(
                    "classifier_use_balanced_class_weight",
                    CLASSIFIER_USE_BALANCED_CLASS_WEIGHT,
                )
            ),
            "use_log_points_feature": bool(
                params.get("use_log_points_feature", CLASSIFIER_USE_LOG_POINTS_FEATURE)
            ),
            "use_log_comments_feature": bool(
                params.get("use_log_comments_feature", CLASSIFIER_USE_LOG_COMMENTS_FEATURE)
            ),
            "use_comment_ratio_feature": bool(
                params.get("use_comment_ratio_feature", CLASSIFIER_USE_COMMENT_RATIO_FEATURE)
            ),
        },
        "clustering": {
            "distance_threshold": float(params.get("cluster_distance_threshold", CLUSTER_AGGLOMERATIVE_THRESHOLD)),
        },
        "cross_encoder": {
            "enabled": bool(params.get("ce_enabled", CROSS_ENCODER_ENABLED)),
            "top_n": int(round(float(params.get("ce_top_n", CROSS_ENCODER_TOP_N)))),
            "weight": float(params.get("ce_weight", CROSS_ENCODER_WEIGHT)),
        },
    }


def get_tuning_config(params: Mapping[str, float | int]) -> AppConfig:
    """Create an AppConfig instance with overridden tuning parameters."""
    resolved = resolve_params(params)
    
    # Load base config from file if it exists, otherwise use defaults
    base = AppConfig.load()
    
    from dataclasses import replace
    
    # Create new config instances with overridden fields
    ranking = replace(base.ranking, **resolved["ranking"])
    adaptive_hn = replace(base.adaptive_hn, **resolved["adaptive_hn"])
    freshness = replace(base.freshness, **resolved["freshness"])
    semantic = replace(base.semantic, **resolved["semantic"])
    classifier = replace(base.classifier, **resolved["classifier"])
    clustering = replace(base.clustering, **resolved["clustering"])
    cross_encoder = replace(base.cross_encoder, **resolved["cross_encoder"])
    
    return replace(
        base,
        ranking=ranking,
        adaptive_hn=adaptive_hn,
        freshness=freshness,
        semantic=semantic,
        classifier=classifier,
        clustering=clustering,
        cross_encoder=cross_encoder,
    )


@contextmanager
def tuned_config(
    params: Mapping[str, float | int],
) -> Iterator[tuple[AppConfig, ResolvedParams]]:
    """Context manager for tuning that provides a config object."""
    resolved = resolve_params(params)
    config = get_tuning_config(params)
    yield config, resolved


def render_promoted_toml(resolved: ResolvedParams) -> str:
    ranking = resolved["ranking"]
    adaptive_hn = resolved["adaptive_hn"]
    freshness = resolved["freshness"]
    semantic = resolved["semantic"]
    classifier = resolved["classifier"]
    clustering = resolved.get("clustering", {})
    cross_encoder = resolved.get("cross_encoder", {})
    
    toml = (
        "# Auto-generated promoted params.\n"
        "# Merge this into hn_rerank.toml under [hn_rerank.*] sections.\n\n"
        "[hn_rerank.ranking]\n"
        f"negative_weight = {float(ranking['negative_weight']):.10f}\n"
        f"diversity_lambda = {float(ranking['diversity_lambda']):.10f}\n"
        f"diversity_lambda_classifier = "
        f"{float(ranking['diversity_lambda_classifier']):.10f}\n"
        f"non_semantic_weight = {float(ranking['non_semantic_weight']):.10f}\n"
        f"comment_ratio = {float(ranking['comment_ratio']):.10f}\n\n"
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
        f"knn_neighbors = {int(semantic['knn_neighbors'])}\n\n"
        "[hn_rerank.classifier]\n"
        f"scoring_mode = \"{classifier['scoring_mode']}\"\n"
        f"feature_mode = \"{classifier['feature_mode']}\"\n"
        f"pairwise_negatives = {int(classifier['pairwise_negatives'])}\n"
        f"pairwise_c = {float(classifier['pairwise_c']):.10f}\n"
        f"k_feat = {int(classifier['k_feat'])}\n"
        f"use_balanced_class_weight = {str(bool(classifier['use_balanced_class_weight'])).lower()}\n"
        f"use_log_points_feature = {str(bool(classifier['use_log_points_feature'])).lower()}\n"
        f"use_log_comments_feature = {str(bool(classifier['use_log_comments_feature'])).lower()}\n"
        f"use_comment_ratio_feature = {str(bool(classifier['use_comment_ratio_feature'])).lower()}\n"
    )
    
    if clustering:
        toml += (
            "\n[hn_rerank.clustering]\n"
            f"distance_threshold = {float(clustering['distance_threshold']):.10f}\n"
        )

    if cross_encoder:
        toml += (
            "\n[hn_rerank.cross_encoder]\n"
            f"enabled = {str(bool(cross_encoder['enabled'])).lower()}\n"
            f"top_n = {int(cross_encoder['top_n'])}\n"
            f"weight = {float(cross_encoder['weight']):.10f}\n"
        )
        
    return toml
