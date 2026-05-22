"""Learned final-rank calibration for dashboard stories."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.config import AppConfig, LearnedRankerConfig
from api.feedback import FeedbackAction, FeedbackRecord
from api.models import RankResult, Story

logger = logging.getLogger(__name__)

LearnedRankerMode = Literal["trained", "loaded", "disabled", "insufficient_labels", "failed"]

FEATURE_NAMES: tuple[str, ...] = (
    "semantic_score",
    "hybrid_score",
    "hn_score",
    "freshness_boost",
    "max_cluster_score",
    "knn_score",
    "max_sim_score",
    "cross_encoder_score",
    "age_hours",
    "log_points",
    "log_comments",
    "is_hn",
    "is_external",
    "is_github_trending",
    "is_reddit",
    "is_curated_external",
)
MODEL_VERSION = 4
TRAINING_SOURCE = "dashboard_feedback"
MODEL_KIND = "ordinal_threshold_v1"
EXTERNAL_DEFAULT_SCORE = 40.0
EXTERNAL_DEFAULT_COMMENT_COUNT = 20.0
DOWNVOTE_LABEL = 0
NEUTRAL_LABEL = 1
UPVOTE_LABEL = 2
ACTION_TO_ORDINAL: dict[FeedbackAction, int] = {
    "down": DOWNVOTE_LABEL,
    "neutral": NEUTRAL_LABEL,
    "up": UPVOTE_LABEL,
}
ORDINAL_TO_ACTION: dict[int, FeedbackAction] = {
    DOWNVOTE_LABEL: "down",
    NEUTRAL_LABEL: "neutral",
    UPVOTE_LABEL: "up",
}
DASHBOARD_FEEDBACK_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "pairwise_accuracy": 0.50,
    "precision_at_10": 0.30,
    "precision_at_5": 0.20,
}
DASHBOARD_FEEDBACK_PRIMARY_METRICS: tuple[str, ...] = (
    "pairwise_accuracy",
    "precision_at_10",
)
DASHBOARD_FEEDBACK_GUARD_METRICS: tuple[str, ...] = (
    "precision_at_5",
    "roc_auc",
)


@dataclass(frozen=True)
class LabeledStory:
    """Explicit dashboard feedback label with captured rank diagnostics."""

    story: Story
    label: int
    rank_result: RankResult
    feedback_updated_at: float = 0.0
    has_raw_story_score: bool = True
    has_raw_comment_count: bool = True

    @property
    def feedback_action(self) -> FeedbackAction:
        return ORDINAL_TO_ACTION.get(self.label, "down")

    @property
    def legacy_binary_label(self) -> int:
        return 1 if self.label == UPVOTE_LABEL else 0


@dataclass(frozen=True)
class OrdinalThresholdModel:
    at_least_neutral: Pipeline
    upvote: Pipeline


@dataclass(frozen=True)
class LearnedRankerResult:
    """Result of training/loading and scoring the learned ranker."""

    mode: LearnedRankerMode
    scores: dict[int, float]
    positive_labels: int
    negative_labels: int
    neutral_labels: int = 0
    reason: str | None = None

    @property
    def has_scores(self) -> bool:
        return bool(self.scores)


@dataclass(frozen=True)
class LearnedRankerEvaluation:
    """Offline comparison of learned scores against stored dashboard labels."""

    label_count: int
    positive_labels: int
    neutral_labels: int
    negative_labels: int
    folds: int
    learned_pairwise_accuracy: float
    hybrid_pairwise_accuracy: float
    learned_precision_at_5: float
    hybrid_precision_at_5: float
    learned_precision_at_10: float
    hybrid_precision_at_10: float
    learned_neutral_rate_at_10: float
    hybrid_neutral_rate_at_10: float
    learned_downvote_rate_at_10: float
    hybrid_downvote_rate_at_10: float
    learned_roc_auc: float | None
    hybrid_roc_auc: float | None
    learned_top_sources: dict[str, int]
    hybrid_top_sources: dict[str, int]


@dataclass(frozen=True)
class DashboardFeedbackMetricSet:
    pairwise_accuracy: float
    precision_at_5: float
    precision_at_10: float
    neutral_rate_at_10: float
    downvote_rate_at_10: float
    roc_auc: float | None
    top_sources: dict[str, int]


@dataclass(frozen=True)
class DashboardFeedbackSummary:
    label_count: int
    usable_label_count: int
    skipped_missing_story_metadata: int
    train_label_count: int
    train_positive_labels: int
    train_neutral_labels: int
    train_negative_labels: int
    holdout_label_count: int
    holdout_positive_labels: int
    holdout_neutral_labels: int
    holdout_negative_labels: int
    holdout_fraction: float
    holdout_start_timestamp: float


@dataclass(frozen=True)
class DashboardFeedbackComparison:
    summary: DashboardFeedbackSummary
    incumbent: DashboardFeedbackMetricSet
    candidate: DashboardFeedbackMetricSet
    incumbent_score: float
    candidate_score: float
    score_delta: float
    primary_failures: list[str]
    guard_failures: list[str]
    passed: bool


def build_labels_from_feedback(
    records: dict[str, FeedbackRecord],
) -> list[LabeledStory]:
    labels: list[LabeledStory] = []
    for record in records.values():
        rank_result = record.to_rank_result()
        if rank_result is None:
            continue
        labels.append(
            LabeledStory(
                story=record.to_story(),
                label=ACTION_TO_ORDINAL.get(record.action, DOWNVOTE_LABEL),
                rank_result=rank_result,
                feedback_updated_at=record.updated_at,
                has_raw_story_score=record.score is not None,
                has_raw_comment_count=record.comment_count is not None,
            )
        )
    return labels


def _safe_float(value: float | int | None) -> float:
    if value is None:
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        return 0.0
    return result


def build_features(
    story: Story,
    result: RankResult | None = None,
    *,
    now: float | None = None,
    source_feature_weight: float = 1.0,
) -> list[float]:
    """Build the stable numeric feature vector used by the learned ranker."""
    now_ts = time.time() if now is None else now
    age_hours = max((now_ts - float(story.time or 0)) / 3600.0, 0.0)
    score = max(float(story.score or 0), 0.0)
    comments = max(float(story.comment_count or 0), 0.0)
    source = story.source
    curated_external = source in {
        "lobsters",
        "tildes",
        "lesswrong",
        "slashdot",
        "github_trending",
        "digg",
    }

    if result is None:
        rank_features = {
            "semantic_score": 0.0,
            "hybrid_score": 0.0,
            "hn_score": 0.0,
            "freshness_boost": 0.0,
            "max_cluster_score": 0.0,
            "knn_score": 0.0,
            "max_sim_score": 0.0,
            "cross_encoder_score": 0.0,
        }
    else:
        rank_features = {
            "semantic_score": result.semantic_score,
            "hybrid_score": result.hybrid_score,
            "hn_score": result.hn_score,
            "freshness_boost": result.freshness_boost,
            "max_cluster_score": result.max_cluster_score,
            "knn_score": result.knn_score,
            "max_sim_score": result.max_sim_score,
            "cross_encoder_score": result.cross_encoder_score,
        }

    return [
        _safe_float(rank_features["semantic_score"]),
        _safe_float(rank_features["hybrid_score"]),
        _safe_float(rank_features["hn_score"]),
        _safe_float(rank_features["freshness_boost"]),
        _safe_float(rank_features["max_cluster_score"]),
        _safe_float(rank_features["knn_score"]),
        _safe_float(rank_features["max_sim_score"]),
        _safe_float(rank_features["cross_encoder_score"]),
        _safe_float(age_hours),
        math.log1p(score),
        math.log1p(comments),
        source_feature_weight if story.is_hn else 0.0,
        source_feature_weight if story.is_external else 0.0,
        source_feature_weight if source == "github_trending" else 0.0,
        source_feature_weight if source.startswith("reddit") else 0.0,
        source_feature_weight if curated_external else 0.0,
    ]


def _label_timestamp(label: LabeledStory) -> float:
    if label.feedback_updated_at > 0:
        return float(label.feedback_updated_at)
    return float(label.story.time or 0)


def _supports_dashboard_hybrid_eval(label: LabeledStory) -> bool:
    if not label.story.is_hn:
        return True
    return label.has_raw_story_score and label.has_raw_comment_count


def _count_labels(labels: list[LabeledStory]) -> tuple[int, int, int]:
    positive = sum(item.label == UPVOTE_LABEL for item in labels)
    neutral = sum(item.label == NEUTRAL_LABEL for item in labels)
    negative = sum(item.label == DOWNVOTE_LABEL for item in labels)
    return positive, neutral, negative


def split_temporal_holdout(
    labels: list[LabeledStory],
    *,
    holdout_fraction: float = 0.25,
    min_holdout_count: int = 20,
    min_class_count: int = 2,
) -> tuple[list[LabeledStory], list[LabeledStory]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if min_holdout_count < 1:
        raise ValueError("min_holdout_count must be at least 1")
    if min_class_count < 1:
        raise ValueError("min_class_count must be at least 1")
    if len(labels) < min_class_count * 2:
        raise ValueError("not enough labels for temporal holdout")

    ordered = sorted(labels, key=_label_timestamp)
    holdout_count = max(min_holdout_count, int(math.ceil(len(ordered) * holdout_fraction)))
    holdout_count = min(len(ordered) - 1, holdout_count)
    start_index = len(ordered) - holdout_count
    while start_index > 0:
        holdout = ordered[start_index:]
        positive_count = sum(item.label == UPVOTE_LABEL for item in holdout)
        negative_count = sum(item.label == DOWNVOTE_LABEL for item in holdout)
        if positive_count >= min_class_count and negative_count >= min_class_count:
            return ordered[:start_index], holdout
        start_index -= 1

    raise ValueError(
        "unable to construct temporal holdout with enough upvote and downvote labels"
    )


def build_training_matrix(
    labels: list[LabeledStory],
    *,
    now: float | None = None,
    source_feature_weight: float = 1.0,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    rows: list[list[float]] = []
    y: list[int] = []
    for item in labels:
        rows.append(
            build_features(
                item.story,
                item.rank_result,
                now=item.feedback_updated_at or now,
                source_feature_weight=source_feature_weight,
            )
        )
        y.append(item.label)
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _binary_targets_from_ordinal(
    y: NDArray[np.int64],
    threshold: int,
) -> NDArray[np.int64]:
    return (y >= threshold).astype(np.int64)


def _balance_binary_matrix(
    x: NDArray[np.float32],
    y: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    keep = min(len(positive_indices), len(negative_indices))
    if keep == 0 or len(positive_indices) == len(negative_indices):
        return x, y
    interleaved = np.empty(keep * 2, dtype=np.int64)
    interleaved[0::2] = positive_indices[:keep]
    interleaved[1::2] = negative_indices[:keep]
    return x[interleaved], y[interleaved]


def _threshold_binary_counts(y: NDArray[np.int64]) -> dict[str, tuple[int, int]]:
    at_least_neutral = _binary_targets_from_ordinal(y, NEUTRAL_LABEL)
    upvote = _binary_targets_from_ordinal(y, UPVOTE_LABEL)
    return {
        "at_least_neutral": (
            int(np.sum(at_least_neutral == 1)),
            int(np.sum(at_least_neutral == 0)),
        ),
        "upvote": (
            int(np.sum(upvote == 1)),
            int(np.sum(upvote == 0)),
        ),
    }


def _has_sufficient_threshold_labels(
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
) -> bool:
    if not labels:
        return False
    _, y = build_training_matrix(labels, source_feature_weight=config.source_feature_weight)
    counts = _threshold_binary_counts(y)
    return all(
        positive >= config.min_positive_labels and negative >= config.min_negative_labels
        for positive, negative in counts.values()
    )


def _insufficient_label_message(
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
) -> str:
    positive_count, neutral_count, negative_count = _count_labels(labels)
    return (
        f"need at least {config.min_positive_labels} upvote and "
        f"{config.min_negative_labels} downvote labels; got "
        f"{positive_count} upvote, {neutral_count} neutral, {negative_count} downvote"
    )


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=0,
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_model(
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
    *,
    now: float | None = None,
) -> OrdinalThresholdModel:
    x_train, y_train = build_training_matrix(
        labels,
        now=now,
        source_feature_weight=config.source_feature_weight,
    )
    counts = _threshold_binary_counts(y_train)
    if any(
        positive < config.min_positive_labels or negative < config.min_negative_labels
        for positive, negative in counts.values()
    ):
        raise ValueError(f"insufficient labels: {_insufficient_label_message(labels, config)}")

    neutral_target = _binary_targets_from_ordinal(y_train, NEUTRAL_LABEL)
    upvote_target = _binary_targets_from_ordinal(y_train, UPVOTE_LABEL)

    neutral_x, neutral_y = x_train, neutral_target
    upvote_x, upvote_y = x_train, upvote_target
    if config.balance_training_labels:
        neutral_x, neutral_y = _balance_binary_matrix(neutral_x, neutral_y)
        upvote_x, upvote_y = _balance_binary_matrix(upvote_x, upvote_y)

    neutral_model = _make_pipeline()
    neutral_model.fit(neutral_x, neutral_y)

    upvote_model = _make_pipeline()
    upvote_model.fit(upvote_x, upvote_y)

    return OrdinalThresholdModel(
        at_least_neutral=neutral_model,
        upvote=upvote_model,
    )


def save_model(model: OrdinalThresholdModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": MODEL_VERSION,
            "model_kind": MODEL_KIND,
            "training_source": TRAINING_SOURCE,
            "feature_names": FEATURE_NAMES,
            "model": {
                "at_least_neutral": model.at_least_neutral,
                "upvote": model.upvote,
            },
        },
        path,
    )


def load_model(path: Path) -> OrdinalThresholdModel:
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError("learned ranker payload is not a dict")
    if raw.get("version") != MODEL_VERSION:
        raise ValueError("learned ranker model version does not match")
    if raw.get("model_kind") != MODEL_KIND:
        raise ValueError("learned ranker model kind does not match")
    if raw.get("training_source") != TRAINING_SOURCE:
        raise ValueError("learned ranker training source does not match")
    if tuple(raw.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("learned ranker feature names do not match")
    model_payload = raw.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("learned ranker model payload is not a dict")
    neutral_model = model_payload.get("at_least_neutral")
    upvote_model = model_payload.get("upvote")
    if not isinstance(neutral_model, Pipeline) or not isinstance(upvote_model, Pipeline):
        raise ValueError("learned ranker threshold models are not sklearn Pipelines")
    return OrdinalThresholdModel(
        at_least_neutral=neutral_model,
        upvote=upvote_model,
    )


def _predict_ordinal_outputs(
    model: OrdinalThresholdModel,
    x_score: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    at_least_neutral = model.at_least_neutral.predict_proba(x_score)[:, 1].astype(np.float32)
    upvote = model.upvote.predict_proba(x_score)[:, 1].astype(np.float32)
    at_least_neutral = np.clip(at_least_neutral, 0.0, 1.0)
    upvote = np.clip(upvote, 0.0, 1.0)
    upvote = np.minimum(upvote, at_least_neutral)
    downvote = np.clip(1.0 - at_least_neutral, 0.0, 1.0)
    neutral = np.clip(at_least_neutral - upvote, 0.0, 1.0)
    utility = np.clip((at_least_neutral + upvote) / 2.0, 0.0, 1.0)
    return utility, downvote, neutral, upvote


def score_ranked_results(
    ranked: list[RankResult],
    stories: list[Story],
    model: OrdinalThresholdModel,
    config: LearnedRankerConfig | None = None,
    *,
    now: float | None = None,
) -> dict[int, float]:
    if not ranked:
        return {}
    rows = [
        build_features(
            stories[result.index],
            result,
            now=now,
            source_feature_weight=(
                config.source_feature_weight if config is not None else 1.0
            ),
        )
        for result in ranked
    ]
    x_score = np.asarray(rows, dtype=np.float32)
    utility, _, _, _ = _predict_ordinal_outputs(model, x_score)
    return {
        result.index: float(utility[position])
        for position, result in enumerate(ranked)
    }


def _pairwise_accuracy(labels: list[int], scores: list[float]) -> float:
    wins = 0.0
    total_weight = 0.0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if labels[left] == labels[right]:
                continue
            weight = float(abs(labels[left] - labels[right]))
            total_weight += weight
            if labels[left] > labels[right]:
                higher, lower = left, right
            else:
                higher, lower = right, left
            if scores[higher] > scores[lower]:
                wins += weight
            elif scores[higher] == scores[lower]:
                wins += 0.5 * weight
    return wins / total_weight if total_weight else 0.0


def _precision_at(labels: list[int], scores: list[float], k: int) -> float:
    if not labels:
        return 0.0
    limit = min(k, len(labels))
    ordered = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    return sum(1 for i in ordered[:limit] if labels[i] == UPVOTE_LABEL) / limit


def _rate_at(labels: list[int], scores: list[float], k: int, target_label: int) -> float:
    if not labels:
        return 0.0
    limit = min(k, len(labels))
    ordered = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    return sum(1 for i in ordered[:limit] if labels[i] == target_label) / limit


def _top_sources(labels: list[LabeledStory], scores: list[float], k: int) -> dict[str, int]:
    ordered = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    counts: dict[str, int] = {}
    for index in ordered[: min(k, len(labels))]:
        source = labels[index].story.source
        counts[source] = counts.get(source, 0) + 1
    return counts


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    binary = [1 if label == UPVOTE_LABEL else 0 for label in labels]
    if len(set(binary)) < 2:
        return None
    return float(roc_auc_score(binary, scores))


def evaluate_labeled_order(
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
    *,
    now: float | None = None,
    max_folds: int = 5,
) -> LearnedRankerEvaluation:
    """Cross-validated learned-vs-hybrid comparison on stored feedback labels."""
    positive_count, neutral_count, negative_count = _count_labels(labels)
    if positive_count < 2 or negative_count < 2:
        raise ValueError(
            f"need at least 2 upvote and 2 downvote labels; got "
            f"{positive_count} upvote, {neutral_count} neutral, {negative_count} downvote"
        )

    y = [item.label for item in labels]
    y_binary = [item.legacy_binary_label for item in labels]
    hybrid_scores = [float(item.rank_result.hybrid_score) for item in labels]
    learned_scores = [0.0 for _ in labels]
    fold_count = min(max_folds, positive_count, negative_count)
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=0)

    label_array = np.asarray(y_binary, dtype=np.int64)
    for train_indices, test_indices in splitter.split(np.zeros(len(y_binary)), label_array):
        fold_labels = [labels[int(i)] for i in train_indices]
        model = train_model(fold_labels, config, now=now)
        rows = [
            build_features(
                labels[int(i)].story,
                labels[int(i)].rank_result,
                now=labels[int(i)].feedback_updated_at or now,
                source_feature_weight=config.source_feature_weight,
            )
            for i in test_indices
        ]
        utility, _, _, _ = _predict_ordinal_outputs(
            model,
            np.asarray(rows, dtype=np.float32),
        )
        for offset, label_index in enumerate(test_indices):
            learned_scores[int(label_index)] = float(utility[offset])

    return LearnedRankerEvaluation(
        label_count=len(labels),
        positive_labels=positive_count,
        neutral_labels=neutral_count,
        negative_labels=negative_count,
        folds=fold_count,
        learned_pairwise_accuracy=_pairwise_accuracy(y, learned_scores),
        hybrid_pairwise_accuracy=_pairwise_accuracy(y, hybrid_scores),
        learned_precision_at_5=_precision_at(y, learned_scores, 5),
        hybrid_precision_at_5=_precision_at(y, hybrid_scores, 5),
        learned_precision_at_10=_precision_at(y, learned_scores, 10),
        hybrid_precision_at_10=_precision_at(y, hybrid_scores, 10),
        learned_neutral_rate_at_10=_rate_at(y, learned_scores, 10, NEUTRAL_LABEL),
        hybrid_neutral_rate_at_10=_rate_at(y, hybrid_scores, 10, NEUTRAL_LABEL),
        learned_downvote_rate_at_10=_rate_at(y, learned_scores, 10, DOWNVOTE_LABEL),
        hybrid_downvote_rate_at_10=_rate_at(y, hybrid_scores, 10, DOWNVOTE_LABEL),
        learned_roc_auc=_safe_auc(y, learned_scores),
        hybrid_roc_auc=_safe_auc(y, hybrid_scores),
        learned_top_sources=_top_sources(labels, learned_scores, 10),
        hybrid_top_sources=_top_sources(labels, hybrid_scores, 10),
    )


def score_labels_with_hybrid_config(
    labels: list[LabeledStory],
    config: AppConfig,
) -> list[float]:
    if not labels:
        return []

    semantic_scores = np.asarray(
        [_safe_float(item.rank_result.semantic_score) for item in labels],
        dtype=np.float32,
    )
    points = np.asarray(
        [
            float(max(item.story.score, 0))
            if item.story.source == "hn"
            else float(max(item.story.score, EXTERNAL_DEFAULT_SCORE))
            for item in labels
        ],
        dtype=np.float32,
    )
    comment_counts = np.asarray(
        [
            float(max(item.story.comment_count or 0, 0))
            if item.story.source == "hn"
            else float(max(item.story.comment_count or 0, EXTERNAL_DEFAULT_COMMENT_COUNT))
            for item in labels
        ],
        dtype=np.float32,
    )

    score_cap = max(float(np.max(points)), config.adaptive_hn.score_normalization_cap)
    hn_scores = np.log1p(points) / np.log1p(score_cap)

    comment_cap = max(
        float(np.max(comment_counts)),
        config.adaptive_hn.score_normalization_cap,
    )
    comment_scores = np.log1p(comment_counts) / np.log1p(comment_cap)

    event_times = np.asarray([_label_timestamp(item) for item in labels], dtype=np.float64)
    story_times = np.asarray([float(item.story.time or 0) for item in labels], dtype=np.float64)
    ages_hours = np.maximum((event_times - story_times) / 3600.0, 0.0)

    non_semantic_weight = float(np.clip(config.ranking.non_semantic_weight, 0.0, 1.0))
    comment_ratio = float(np.clip(config.ranking.comment_ratio, 0.0, 1.0))

    if non_semantic_weight <= 0.0:
        return semantic_scores.astype(np.float32).tolist()

    young_hn_weight = min(config.adaptive_hn.weight_min, config.adaptive_hn.weight_max)
    old_hn_weight = max(config.adaptive_hn.weight_min, config.adaptive_hn.weight_max)
    threshold_span = config.adaptive_hn.threshold_old - config.adaptive_hn.threshold_young
    if threshold_span <= 0:
        adaptive_t = (ages_hours >= config.adaptive_hn.threshold_old).astype(np.float64)
    else:
        adaptive_t = np.clip(
            (ages_hours - config.adaptive_hn.threshold_young) / threshold_span,
            0.0,
            1.0,
        )
    hn_weights = young_hn_weight + adaptive_t * (old_hn_weight - young_hn_weight)

    non_semantic_scores = (
        (1.0 - comment_ratio) * hn_scores + comment_ratio * comment_scores
    )
    effective_non_semantic_weights = hn_weights * non_semantic_weight
    hybrid_scores = (
        (1.0 - effective_non_semantic_weights) * semantic_scores
        + effective_non_semantic_weights * non_semantic_scores
    ).astype(np.float32)

    if config.freshness.enabled and config.freshness.max_boost > 0:
        freshness = np.power(2.0, -ages_hours / config.freshness.half_life_hours)
        freshness = np.clip(freshness, 0.0, 1.0)
        freshness_boost = (config.freshness.max_boost * freshness).astype(np.float32)
        hybrid_scores = hybrid_scores + freshness_boost

    return hybrid_scores.astype(np.float32).tolist()


def _dashboard_metric_set(
    labels: list[LabeledStory],
    scores: list[float],
) -> DashboardFeedbackMetricSet:
    y = [item.label for item in labels]
    return DashboardFeedbackMetricSet(
        pairwise_accuracy=_pairwise_accuracy(y, scores),
        precision_at_5=_precision_at(y, scores, 5),
        precision_at_10=_precision_at(y, scores, 10),
        neutral_rate_at_10=_rate_at(y, scores, 10, NEUTRAL_LABEL),
        downvote_rate_at_10=_rate_at(y, scores, 10, DOWNVOTE_LABEL),
        roc_auc=_safe_auc(y, scores),
        top_sources=_top_sources(labels, scores, 10),
    )


def _dashboard_metric_value(
    metrics: DashboardFeedbackMetricSet,
    name: str,
) -> float | None:
    value = getattr(metrics, name)
    if value is None:
        return None
    return float(value)


def score_dashboard_feedback_metrics(
    metrics: DashboardFeedbackMetricSet,
    *,
    weights: dict[str, float] = DASHBOARD_FEEDBACK_OBJECTIVE_WEIGHTS,
) -> float:
    total = 0.0
    for metric_name, metric_weight in weights.items():
        metric_value = _dashboard_metric_value(metrics, metric_name)
        if metric_value is None:
            continue
        total += metric_weight * metric_value
    return float(total)


def compare_dashboard_feedback_configs(
    labels: list[LabeledStory],
    incumbent_config: AppConfig,
    candidate_config: AppConfig,
    *,
    holdout_fraction: float = 0.25,
    min_holdout_count: int = 20,
    min_class_count: int = 2,
    score_tolerance: float = 0.0,
    primary_metrics: tuple[str, ...] = DASHBOARD_FEEDBACK_PRIMARY_METRICS,
    guard_metrics: tuple[str, ...] = DASHBOARD_FEEDBACK_GUARD_METRICS,
) -> DashboardFeedbackComparison:
    usable_labels = [item for item in labels if _supports_dashboard_hybrid_eval(item)]
    skipped = len(labels) - len(usable_labels)
    train_labels, holdout_labels = split_temporal_holdout(
        usable_labels,
        holdout_fraction=holdout_fraction,
        min_holdout_count=min_holdout_count,
        min_class_count=min_class_count,
    )

    incumbent_scores = score_labels_with_hybrid_config(holdout_labels, incumbent_config)
    candidate_scores = score_labels_with_hybrid_config(holdout_labels, candidate_config)
    incumbent_metrics = _dashboard_metric_set(holdout_labels, incumbent_scores)
    candidate_metrics = _dashboard_metric_set(holdout_labels, candidate_scores)
    incumbent_score = score_dashboard_feedback_metrics(incumbent_metrics)
    candidate_score = score_dashboard_feedback_metrics(candidate_metrics)

    primary_failures: list[str] = []
    for metric_name in primary_metrics:
        incumbent_value = _dashboard_metric_value(incumbent_metrics, metric_name)
        candidate_value = _dashboard_metric_value(candidate_metrics, metric_name)
        if (
            incumbent_value is not None
            and candidate_value is not None
            and candidate_value < incumbent_value - score_tolerance
        ):
            primary_failures.append(metric_name)

    guard_failures: list[str] = []
    for metric_name in guard_metrics:
        incumbent_value = _dashboard_metric_value(incumbent_metrics, metric_name)
        candidate_value = _dashboard_metric_value(candidate_metrics, metric_name)
        if incumbent_value is None or candidate_value is None:
            continue
        if candidate_value < incumbent_value - score_tolerance:
            guard_failures.append(metric_name)

    train_positive, train_neutral, train_negative = _count_labels(train_labels)
    holdout_positive, holdout_neutral, holdout_negative = _count_labels(holdout_labels)
    summary = DashboardFeedbackSummary(
        label_count=len(labels),
        usable_label_count=len(usable_labels),
        skipped_missing_story_metadata=skipped,
        train_label_count=len(train_labels),
        train_positive_labels=train_positive,
        train_neutral_labels=train_neutral,
        train_negative_labels=train_negative,
        holdout_label_count=len(holdout_labels),
        holdout_positive_labels=holdout_positive,
        holdout_neutral_labels=holdout_neutral,
        holdout_negative_labels=holdout_negative,
        holdout_fraction=holdout_fraction,
        holdout_start_timestamp=_label_timestamp(holdout_labels[0]) if holdout_labels else 0.0,
    )
    score_delta = float(candidate_score - incumbent_score)
    return DashboardFeedbackComparison(
        summary=summary,
        incumbent=incumbent_metrics,
        candidate=candidate_metrics,
        incumbent_score=incumbent_score,
        candidate_score=candidate_score,
        score_delta=score_delta,
        primary_failures=primary_failures,
        guard_failures=guard_failures,
        passed=score_delta > score_tolerance
        and not primary_failures
        and not guard_failures,
    )


def train_or_load_and_score(
    ranked: list[RankResult],
    stories: list[Story],
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
    *,
    now: float | None = None,
) -> LearnedRankerResult:
    positive_count, neutral_count, negative_count = _count_labels(labels)
    if not (config.shadow_enabled or config.active_enabled):
        return LearnedRankerResult(
            mode="disabled",
            scores={},
            positive_labels=positive_count,
            negative_labels=negative_count,
            neutral_labels=neutral_count,
        )

    model_path = config.model_path
    mode: LearnedRankerMode
    try:
        if _has_sufficient_threshold_labels(labels, config):
            model = train_model(labels, config, now=now)
            save_model(model, model_path)
            mode = "trained"
        else:
            if not model_path.exists():
                return LearnedRankerResult(
                    mode="insufficient_labels",
                    scores={},
                    positive_labels=positive_count,
                    negative_labels=negative_count,
                    neutral_labels=neutral_count,
                    reason=_insufficient_label_message(labels, config),
                )
            try:
                model = load_model(model_path)
                mode = "loaded"
            except Exception as exc:
                return LearnedRankerResult(
                    mode="insufficient_labels",
                    scores={},
                    positive_labels=positive_count,
                    negative_labels=negative_count,
                    neutral_labels=neutral_count,
                    reason=(
                        f"{_insufficient_label_message(labels, config)}; "
                        f"existing model is not reusable: {type(exc).__name__}: {exc}"
                    ),
                )

        scores = score_ranked_results(ranked, stories, model, config, now=now)
    except Exception as exc:
        logger.exception("Learned final ranker failed")
        return LearnedRankerResult(
            mode="failed",
            scores={},
            positive_labels=positive_count,
            negative_labels=negative_count,
            neutral_labels=neutral_count,
            reason=f"{type(exc).__name__}: {exc}",
        )

    return LearnedRankerResult(
        mode=mode,
        scores=scores,
        positive_labels=positive_count,
        negative_labels=negative_count,
        neutral_labels=neutral_count,
    )
