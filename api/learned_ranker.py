"""Ordinal threshold utilities for feedback-trained ranking models."""

from __future__ import annotations

import logging
import math
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

from api.config import SingleModelConfig
from api.feedback import FeedbackAction, FeedbackRecord
from api.models import RankResult, Story

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "semantic_score",
    "hybrid_score",
    "max_cluster_score",
    "knn_score",
    "max_sim_score",
    "cross_encoder_score",
    "log_points",
    "log_comments",
)
MODEL_VERSION = 5
TRAINING_SOURCE = "dashboard_feedback"
MODEL_KIND = "ordinal_threshold_v1"
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
class ScoreMetrics:
    pairwise_accuracy: float
    precision_at_5: float
    precision_at_10: float
    neutral_rate_at_10: float
    downvote_rate_at_10: float
    roc_auc: float | None
    top_sources: dict[str, int]


@dataclass(frozen=True)
class ScoreComparison:
    label: str
    metrics: ScoreMetrics


def build_labels_from_feedback(
    records: dict[str, FeedbackRecord],
) -> list[LabeledStory]:
    labels: list[LabeledStory] = []
    for record in records.values():
        rank_result = RankResult(
            index=-1,
            hybrid_score=0.0,
            best_fav_index=-1,
            max_sim_score=0.0,
            knn_score=0.0,
            max_cluster_score=0.0,
            semantic_score=0.0,
        )
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
) -> list[float]:
    """Build the stable numeric feature vector used by the learned ranker."""
    score = max(float(story.score or 0), 0.0)
    comments = max(float(story.comment_count or 0), 0.0)

    if result is None:
        rank_features = {
            "semantic_score": 0.0,
            "hybrid_score": 0.0,
            "max_cluster_score": 0.0,
            "knn_score": 0.0,
            "max_sim_score": 0.0,
            "cross_encoder_score": 0.0,
        }
    else:
        rank_features = {
            "semantic_score": result.semantic_score,
            "hybrid_score": result.hybrid_score,
            "max_cluster_score": result.max_cluster_score,
            "knn_score": result.knn_score,
            "max_sim_score": result.max_sim_score,
            "cross_encoder_score": result.cross_encoder_score,
        }

    return [
        _safe_float(rank_features["semantic_score"]),
        _safe_float(rank_features["hybrid_score"]),
        _safe_float(rank_features["max_cluster_score"]),
        _safe_float(rank_features["knn_score"]),
        _safe_float(rank_features["max_sim_score"]),
        _safe_float(rank_features["cross_encoder_score"]),
        math.log1p(score),
        math.log1p(comments),
    ]


def _label_timestamp(label: LabeledStory) -> float:
    if label.feedback_updated_at > 0:
        return float(label.feedback_updated_at)
    return float(label.story.time or 0)


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
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    rows: list[list[float]] = []
    y: list[int] = []
    for item in labels:
        rows.append(
            build_features(
                item.story,
                item.rank_result,
                now=item.feedback_updated_at or now,
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
    config: SingleModelConfig,
) -> bool:
    if not labels:
        return False
    _, y = build_training_matrix(labels)
    counts = _threshold_binary_counts(y)
    return all(
        positive >= config.min_positive_labels and negative >= config.min_negative_labels
        for positive, negative in counts.values()
    )


def _insufficient_label_message(
    labels: list[LabeledStory],
    config: SingleModelConfig,
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
    config: SingleModelConfig,
    *,
    now: float | None = None,
) -> OrdinalThresholdModel:
    x_train, y_train = build_training_matrix(
        labels,
        now=now,
    )
    return train_model_from_matrix(x_train, y_train, config, labels=labels)


def train_model_from_matrix(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    config: SingleModelConfig,
    *,
    labels: list[LabeledStory] | None = None,
) -> OrdinalThresholdModel:
    counts = _threshold_binary_counts(y_train)
    if any(
        positive < config.min_positive_labels or negative < config.min_negative_labels
        for positive, negative in counts.values()
    ):
        if labels is None:
            positive_count = int(np.sum(y_train == UPVOTE_LABEL))
            neutral_count = int(np.sum(y_train == NEUTRAL_LABEL))
            negative_count = int(np.sum(y_train == DOWNVOTE_LABEL))
            detail = (
                f"need at least {config.min_positive_labels} upvote and "
                f"{config.min_negative_labels} downvote labels; got "
                f"{positive_count} upvote, {neutral_count} neutral, {negative_count} downvote"
            )
        else:
            detail = _insufficient_label_message(labels, config)
        raise ValueError(f"insufficient labels: {detail}")

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
    config: SingleModelConfig | None = None,
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


def score_metrics_for_labels(
    labels: list[LabeledStory],
    scores: list[float],
) -> ScoreMetrics:
    y = [item.label for item in labels]
    return ScoreMetrics(
        pairwise_accuracy=_pairwise_accuracy(y, scores),
        precision_at_5=_precision_at(y, scores, 5),
        precision_at_10=_precision_at(y, scores, 10),
        neutral_rate_at_10=_rate_at(y, scores, 10, NEUTRAL_LABEL),
        downvote_rate_at_10=_rate_at(y, scores, 10, DOWNVOTE_LABEL),
        roc_auc=_safe_auc(y, scores),
        top_sources=_top_sources(labels, scores, 10),
    )


def evaluate_labeled_score_sources(
    labels: list[LabeledStory],
    score_sources: dict[str, list[float]],
) -> list[ScoreComparison]:
    comparisons: list[ScoreComparison] = []
    for label, scores in score_sources.items():
        if len(scores) != len(labels):
            raise ValueError(
                f"score source '{label}' length {len(scores)} does not match labels {len(labels)}"
            )
        comparisons.append(
            ScoreComparison(
                label=label,
                metrics=score_metrics_for_labels(labels, scores),
            )
        )
    return comparisons


def evaluate_labeled_order(
    labels: list[LabeledStory],
    config: SingleModelConfig,
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
            )
            for i in test_indices
        ]
        utility, _, _, _ = _predict_ordinal_outputs(
            model,
            np.asarray(rows, dtype=np.float32),
        )
        for offset, label_index in enumerate(test_indices):
            learned_scores[int(label_index)] = float(utility[offset])

    learned_metrics, hybrid_metrics = evaluate_labeled_score_sources(
        labels,
        {
            "learned": learned_scores,
            "hybrid": hybrid_scores,
        },
    )

    return LearnedRankerEvaluation(
        label_count=len(labels),
        positive_labels=positive_count,
        neutral_labels=neutral_count,
        negative_labels=negative_count,
        folds=fold_count,
        learned_pairwise_accuracy=learned_metrics.metrics.pairwise_accuracy,
        hybrid_pairwise_accuracy=hybrid_metrics.metrics.pairwise_accuracy,
        learned_precision_at_5=learned_metrics.metrics.precision_at_5,
        hybrid_precision_at_5=hybrid_metrics.metrics.precision_at_5,
        learned_precision_at_10=learned_metrics.metrics.precision_at_10,
        hybrid_precision_at_10=hybrid_metrics.metrics.precision_at_10,
        learned_neutral_rate_at_10=learned_metrics.metrics.neutral_rate_at_10,
        hybrid_neutral_rate_at_10=hybrid_metrics.metrics.neutral_rate_at_10,
        learned_downvote_rate_at_10=learned_metrics.metrics.downvote_rate_at_10,
        hybrid_downvote_rate_at_10=hybrid_metrics.metrics.downvote_rate_at_10,
        learned_roc_auc=learned_metrics.metrics.roc_auc,
        hybrid_roc_auc=hybrid_metrics.metrics.roc_auc,
        learned_top_sources=learned_metrics.metrics.top_sources,
        hybrid_top_sources=hybrid_metrics.metrics.top_sources,
    )
