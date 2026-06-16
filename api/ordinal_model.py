"""Ordinal threshold utilities for feedback-trained ranking models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.config import SingleModelConfig
from api.feedback import FeedbackAction, FeedbackRecord
from api.models import RankResult, Story

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "model_score",
    "max_cluster_score",
    "knn_score",
    "max_sim_score",
    "log_points",
    "log_comments",
)
MODEL_VERSION = 6
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


def build_labels_from_feedback(
    records: dict[str, FeedbackRecord],
) -> list[LabeledStory]:
    labels: list[LabeledStory] = []
    for record in records.values():
        rank_result = RankResult(
            index=-1,
            model_score=0.0,
            best_fav_index=-1,
            max_sim_score=0.0,
            knn_score=0.0,
            max_cluster_score=0.0,
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
            "model_score": 0.0,
            "max_cluster_score": 0.0,
            "knn_score": 0.0,
            "max_sim_score": 0.0,
        }
    else:
        rank_features = {
            "model_score": result.model_score,
            "max_cluster_score": result.max_cluster_score,
            "knn_score": result.knn_score,
            "max_sim_score": result.max_sim_score,
        }

    return [
        _safe_float(rank_features["model_score"]),
        _safe_float(rank_features["max_cluster_score"]),
        _safe_float(rank_features["knn_score"]),
        _safe_float(rank_features["max_sim_score"]),
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
        positive >= config.min_positive_labels
        and negative >= config.min_negative_labels
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


def _make_pipeline(config: SingleModelConfig) -> Pipeline:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    mtype = config.model_type.lower().strip()
    if mtype == "random_forest":
        max_depth = config.rf_max_depth if config.rf_max_depth != 0 else None
        _rf_mf = config.rf_max_features
        try:
            max_features = float(_rf_mf)
        except ValueError:
            max_features = _rf_mf
        clf = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            min_samples_split=config.rf_min_samples_split,
            max_features=max_features,
            class_weight="balanced",
            random_state=0,
        )
    elif mtype == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=5,
            class_weight="balanced",
            random_state=0,
        )
    elif mtype == "svm":
        kernel = config.svm_kernel.lower().strip()
        gamma = config.svm_gamma
        clf = SVC(
            C=float(config.svm_c),
            kernel=kernel,
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            random_state=0,
        )
    elif mtype == "mlp":
        hl = tuple(int(x.strip()) for x in config.mlp_hidden_layers.split(","))
        clf = MLPClassifier(
            hidden_layer_sizes=hl,
            max_iter=1000,
            activation=config.mlp_activation,
            alpha=config.mlp_alpha,
            solver=config.mlp_solver,
            learning_rate_init=config.mlp_learning_rate_init,
            random_state=0,
        )
    else:
        # Default: logistic regression
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
            solver="liblinear",
        )

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", clf),
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

    neutral_model = _make_pipeline(config)
    neutral_model.fit(neutral_x, neutral_y)

    upvote_model = _make_pipeline(config)
    upvote_model.fit(upvote_x, upvote_y)

    return OrdinalThresholdModel(
        at_least_neutral=neutral_model,
        upvote=upvote_model,
    )


def _predict_ordinal_outputs(
    model: OrdinalThresholdModel,
    x_score: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    at_least_neutral = model.at_least_neutral.predict_proba(x_score)[:, 1].astype(
        np.float32
    )
    upvote = model.upvote.predict_proba(x_score)[:, 1].astype(np.float32)
    at_least_neutral = np.clip(at_least_neutral, 0.0, 1.0)
    upvote = np.clip(upvote, 0.0, 1.0)
    upvote = np.minimum(upvote, at_least_neutral)
    downvote = np.clip(1.0 - at_least_neutral, 0.0, 1.0)
    neutral = np.clip(at_least_neutral - upvote, 0.0, 1.0)
    utility = np.clip((at_least_neutral + upvote) / 2.0, 0.0, 1.0)
    return utility, downvote, neutral, upvote
