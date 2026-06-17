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
class OrdinalThresholdModel:
    at_least_neutral: Pipeline
    upvote: Pipeline


def _binary_targets_from_ordinal(
    y: NDArray[np.int64],
    threshold: int,
) -> NDArray[np.int64]:
    return (y >= threshold).astype(np.int64)


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


def _make_pipeline(config: SingleModelConfig, cv: int = 3) -> Pipeline:
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

    if cv < 2:
        model_step = clf
    else:
        from sklearn.calibration import CalibratedClassifierCV

        model_step = CalibratedClassifierCV(clf, method="isotonic", cv=cv)

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model_step),
        ]
    )


def train_model_from_matrix(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    config: SingleModelConfig,
) -> OrdinalThresholdModel:
    counts = _threshold_binary_counts(y_train)
    if any(
        positive < config.min_positive_labels or negative < config.min_negative_labels
        for positive, negative in counts.values()
    ):
        positive_count = int(np.sum(y_train == UPVOTE_LABEL))
        neutral_count = int(np.sum(y_train == NEUTRAL_LABEL))
        negative_count = int(np.sum(y_train == DOWNVOTE_LABEL))
        detail = (
            f"need at least {config.min_positive_labels} upvote and "
            f"{config.min_negative_labels} downvote labels; got "
            f"{positive_count} upvote, {neutral_count} neutral, {negative_count} downvote"
        )
        raise ValueError(f"insufficient labels: {detail}")

    neutral_target = _binary_targets_from_ordinal(y_train, NEUTRAL_LABEL)
    upvote_target = _binary_targets_from_ordinal(y_train, UPVOTE_LABEL)

    neutral_x, neutral_y = x_train, neutral_target
    upvote_x, upvote_y = x_train, upvote_target

    def get_cv(y: NDArray[np.int64]) -> int:
        min_class_count = min(np.sum(y == 0), np.sum(y == 1))
        return int(min(3, min_class_count))

    neutral_cv = get_cv(neutral_y)
    upvote_cv = get_cv(upvote_y)
    logging.warning(
        "[TRAIN DEBUG] x_train.shape=%s n_pos=%d n_neutral=%d n_neg=%d "
        "neutral_cv=%d upvote_cv=%d",
        x_train.shape,
        int(np.sum(y_train == UPVOTE_LABEL)),
        int(np.sum(y_train == NEUTRAL_LABEL)),
        int(np.sum(y_train == DOWNVOTE_LABEL)),
        neutral_cv,
        upvote_cv,
    )

    neutral_model = _make_pipeline(config, cv=neutral_cv)
    neutral_model.fit(neutral_x, neutral_y)

    upvote_model = _make_pipeline(config, cv=upvote_cv)
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
    logging.warning(
        "[SCORE DEBUG] at_least_neutral: mean=%.4f std=%.4f min=%.4f max=%.4f",
        at_least_neutral.mean(),
        at_least_neutral.std(),
        at_least_neutral.min(),
        at_least_neutral.max(),
    )
    logging.warning(
        "[SCORE DEBUG] upvote: mean=%.4f std=%.4f min=%.4f max=%.4f",
        upvote.mean(),
        upvote.std(),
        upvote.min(),
        upvote.max(),
    )
    # Compute utility on raw probabilities to preserve variance (prevents 0.5 ties)
    expected_score = at_least_neutral + upvote
    utility = np.clip(expected_score / 2.0, 0.0, 1.0)
    logging.warning(
        "[SCORE DEBUG] utility: mean=%.4f std=%.4f unique=%d",
        utility.mean(),
        utility.std(),
        len(np.unique(utility)),
    )

    # Compute valid constrained probabilities for the breakdown
    at_least_neutral = np.clip(at_least_neutral, 0.0, 1.0)
    upvote = np.clip(upvote, 0.0, 1.0)
    upvote = np.minimum(upvote, at_least_neutral)
    downvote = np.clip(1.0 - at_least_neutral, 0.0, 1.0)
    neutral = np.clip(at_least_neutral - upvote, 0.0, 1.0)
    return utility, downvote, neutral, upvote
