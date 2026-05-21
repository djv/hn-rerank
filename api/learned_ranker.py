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

from api.config import LearnedRankerConfig
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
MODEL_VERSION = 3
TRAINING_SOURCE = "dashboard_feedback"


@dataclass(frozen=True)
class LabeledStory:
    """Explicit dashboard feedback label with captured rank diagnostics."""

    story: Story
    label: int
    rank_result: RankResult


@dataclass(frozen=True)
class LearnedRankerResult:
    """Result of training/loading and scoring the learned ranker."""

    mode: LearnedRankerMode
    scores: dict[int, float]
    positive_labels: int
    negative_labels: int
    reason: str | None = None

    @property
    def has_scores(self) -> bool:
        return bool(self.scores)


@dataclass(frozen=True)
class LearnedRankerEvaluation:
    """Offline comparison of learned scores against stored dashboard labels."""

    label_count: int
    positive_labels: int
    negative_labels: int
    folds: int
    learned_pairwise_accuracy: float
    hybrid_pairwise_accuracy: float
    learned_precision_at_5: float
    hybrid_precision_at_5: float
    learned_precision_at_10: float
    hybrid_precision_at_10: float
    learned_roc_auc: float | None
    hybrid_roc_auc: float | None
    learned_top_sources: dict[str, int]
    hybrid_top_sources: dict[str, int]


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

    values = [
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
    return values


def balance_labels(labels: list[LabeledStory]) -> list[LabeledStory]:
    """Deterministically downsample the majority class for training."""
    positives = [item for item in labels if item.label == 1]
    negatives = [item for item in labels if item.label == 0]
    keep = min(len(positives), len(negatives))
    if keep == 0 or len(positives) == len(negatives):
        return labels

    balanced: list[LabeledStory] = []
    pos_iter = iter(positives[:keep])
    neg_iter = iter(negatives[:keep])
    for _ in range(keep):
        balanced.append(next(pos_iter))
        balanced.append(next(neg_iter))
    return balanced


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
                now=now,
                source_feature_weight=source_feature_weight,
            )
        )
        y.append(item.label)
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_model(
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
    *,
    now: float | None = None,
) -> Pipeline:
    positive_count = sum(1 for item in labels if item.label == 1)
    negative_count = sum(1 for item in labels if item.label == 0)
    if positive_count < config.min_positive_labels or negative_count < config.min_negative_labels:
        raise ValueError(
            f"insufficient labels: {positive_count} positive, {negative_count} negative"
        )

    training_labels = balance_labels(labels) if config.balance_training_labels else labels
    x_train, y_train = build_training_matrix(
        training_labels,
        now=now,
        source_feature_weight=config.source_feature_weight,
    )
    model = Pipeline(
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
    model.fit(x_train, y_train)
    return model


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": MODEL_VERSION,
            "training_source": TRAINING_SOURCE,
            "feature_names": FEATURE_NAMES,
            "model": model,
        },
        path,
    )


def load_model(path: Path) -> Pipeline:
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError("learned ranker payload is not a dict")
    if raw.get("version") != MODEL_VERSION:
        raise ValueError("learned ranker model version does not match")
    if raw.get("training_source") != TRAINING_SOURCE:
        raise ValueError("learned ranker training source does not match")
    if tuple(raw.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("learned ranker feature names do not match")
    model = raw.get("model")
    if not isinstance(model, Pipeline):
        raise ValueError("learned ranker model is not a sklearn Pipeline")
    return model


def score_ranked_results(
    ranked: list[RankResult],
    stories: list[Story],
    model: Pipeline,
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
    probs = model.predict_proba(x_score)[:, 1]
    return {
        result.index: float(max(0.0, min(1.0, probs[position])))
        for position, result in enumerate(ranked)
    }


def _pairwise_accuracy(labels: list[int], scores: list[float]) -> float:
    wins = 0
    pairs = 0
    for pos_index, label in enumerate(labels):
        if label != 1:
            continue
        for neg_index, neg_label in enumerate(labels):
            if neg_label != 0:
                continue
            pairs += 1
            if scores[pos_index] > scores[neg_index]:
                wins += 1
            elif scores[pos_index] == scores[neg_index]:
                wins += 0.5
    return wins / pairs if pairs else 0.0


def _precision_at(labels: list[int], scores: list[float], k: int) -> float:
    if not labels:
        return 0.0
    limit = min(k, len(labels))
    ordered = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    return sum(labels[i] for i in ordered[:limit]) / limit


def _top_sources(labels: list[LabeledStory], scores: list[float], k: int) -> dict[str, int]:
    ordered = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    counts: dict[str, int] = {}
    for index in ordered[: min(k, len(labels))]:
        source = labels[index].story.source
        counts[source] = counts.get(source, 0) + 1
    return counts


def _safe_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def evaluate_labeled_order(
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
    *,
    now: float | None = None,
    max_folds: int = 5,
) -> LearnedRankerEvaluation:
    """Cross-validated learned-vs-hybrid comparison on stored feedback labels."""
    positive_count = sum(1 for item in labels if item.label == 1)
    negative_count = sum(1 for item in labels if item.label == 0)
    if positive_count < 2 or negative_count < 2:
        raise ValueError(
            f"need at least 2 positive and 2 negative labels; got "
            f"{positive_count} positive and {negative_count} negative"
        )

    y = [item.label for item in labels]
    hybrid_scores = [float(item.rank_result.hybrid_score) for item in labels]
    learned_scores = [0.0 for _ in labels]
    fold_count = min(max_folds, positive_count, negative_count)
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=0)

    label_array = np.asarray(y, dtype=np.int64)
    for train_indices, test_indices in splitter.split(np.zeros(len(y)), label_array):
        fold_labels = [labels[int(i)] for i in train_indices]
        model = train_model(fold_labels, config, now=now)
        rows = [
            build_features(
                labels[int(i)].story,
                labels[int(i)].rank_result,
                now=now,
                source_feature_weight=config.source_feature_weight,
            )
            for i in test_indices
        ]
        probs = model.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
        for offset, label_index in enumerate(test_indices):
            learned_scores[int(label_index)] = float(probs[offset])

    return LearnedRankerEvaluation(
        label_count=len(labels),
        positive_labels=positive_count,
        negative_labels=negative_count,
        folds=fold_count,
        learned_pairwise_accuracy=_pairwise_accuracy(y, learned_scores),
        hybrid_pairwise_accuracy=_pairwise_accuracy(y, hybrid_scores),
        learned_precision_at_5=_precision_at(y, learned_scores, 5),
        hybrid_precision_at_5=_precision_at(y, hybrid_scores, 5),
        learned_precision_at_10=_precision_at(y, learned_scores, 10),
        hybrid_precision_at_10=_precision_at(y, hybrid_scores, 10),
        learned_roc_auc=_safe_auc(y, learned_scores),
        hybrid_roc_auc=_safe_auc(y, hybrid_scores),
        learned_top_sources=_top_sources(labels, learned_scores, 10),
        hybrid_top_sources=_top_sources(labels, hybrid_scores, 10),
    )


def train_or_load_and_score(
    ranked: list[RankResult],
    stories: list[Story],
    labels: list[LabeledStory],
    config: LearnedRankerConfig,
    *,
    now: float | None = None,
) -> LearnedRankerResult:
    positive_count = sum(1 for item in labels if item.label == 1)
    negative_count = sum(1 for item in labels if item.label == 0)
    if not (config.shadow_enabled or config.active_enabled):
        return LearnedRankerResult(
            mode="disabled",
            scores={},
            positive_labels=positive_count,
            negative_labels=negative_count,
        )

    model_path = config.model_path
    mode: LearnedRankerMode
    try:
        if (
            positive_count >= config.min_positive_labels
            and negative_count >= config.min_negative_labels
        ):
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
                    reason=(
                        f"need at least {config.min_positive_labels} positive and "
                        f"{config.min_negative_labels} negative labels"
                    ),
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
                    reason=(
                        f"need at least {config.min_positive_labels} positive and "
                        f"{config.min_negative_labels} negative labels; "
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
            reason=f"{type(exc).__name__}: {exc}",
        )

    return LearnedRankerResult(
        mode=mode,
        scores=scores,
        positive_labels=positive_count,
        negative_labels=negative_count,
    )
