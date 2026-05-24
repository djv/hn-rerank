"""Feedback-trained single ranking model built from dashboard feedback labels."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

from api.config import AppConfig, SingleModelConfig
from api.feedback import FeedbackRecord
from api.learned_ranker import (
    ACTION_TO_ORDINAL,
    DOWNVOTE_LABEL,
    NEUTRAL_LABEL,
    UPVOTE_LABEL,
    OrdinalThresholdModel,
    ScoreMetrics,
    _predict_ordinal_outputs,
    train_model_from_matrix,
)
from api.models import RankResult, Story
from api.rerank import (
    _classifier_metadata_features,
    cluster_interests_with_labels,
    compute_classifier_similarity_features,
    get_embeddings,
    stack_classifier_similarity_features,
)


DERIVED_FEATURE_FLAGS: tuple[tuple[str, str], ...] = (
    ("centroid_feature", "use_centroid_feature"),
    ("pos_knn_feature", "use_pos_knn_feature"),
    ("neg_knn_feature", "use_neg_knn_feature"),
    ("closest_pos", "use_closest_pos_feature"),
    ("closest_neg", "use_closest_neg_feature"),
    ("closest_centroid", "use_closest_centroid_feature"),
    ("knn_pos_n1", "use_knn_pos_n1_feature"),
    ("knn_pos_n3", "use_knn_pos_n3_feature"),
    ("knn_pos_n5", "use_knn_pos_n5_feature"),
    ("knn_pos_n10", "use_knn_pos_n10_feature"),
    ("knn_neg_n1", "use_knn_neg_n1_feature"),
    ("knn_neg_n3", "use_knn_neg_n3_feature"),
    ("knn_neg_n5", "use_knn_neg_n5_feature"),
    ("knn_neg_n10", "use_knn_neg_n10_feature"),
)
METADATA_FEATURE_FLAGS: tuple[tuple[str, str], ...] = (
    ("log_points", "use_log_points_feature"),
    ("log_comments", "use_log_comments_feature"),
    ("comment_ratio", "use_comment_ratio_feature"),
)


@dataclass(frozen=True)
class SingleModelLabeledStory:
    key: str
    story: Story
    label: int
    feedback_updated_at: float = 0.0

    @property
    def legacy_binary_label(self) -> int:
        return 1 if self.label == UPVOTE_LABEL else 0


@dataclass(frozen=True)
class SingleModelFeatureBatch:
    rows: NDArray[np.float32]
    feature_names: tuple[str, ...]
    derived_feature_dim: int
    metadata_feature_dim: int


@dataclass(frozen=True)
class FeedbackLabelBuildResult:
    labels: list[SingleModelLabeledStory]
    skipped_count: int


def build_single_model_feedback_labels(
    records: dict[str, FeedbackRecord],
) -> FeedbackLabelBuildResult:
    labels: list[SingleModelLabeledStory] = []
    skipped_count = 0
    for key, record in records.items():
        story = record.to_story()
        if not story.text_content.strip():
            skipped_count += 1
            continue
        labels.append(
            SingleModelLabeledStory(
                key=key,
                story=story,
                label=ACTION_TO_ORDINAL.get(record.action, DOWNVOTE_LABEL),
                feedback_updated_at=record.updated_at,
            )
        )
    return FeedbackLabelBuildResult(labels=labels, skipped_count=skipped_count)


def _feature_names(config: AppConfig, embedding_dim: int) -> tuple[str, ...]:
    names = [f"embedding_{index}" for index in range(embedding_dim)]
    for feature_name, attr_name in DERIVED_FEATURE_FLAGS:
        if bool(getattr(config.classifier, attr_name, False)):
            names.append(feature_name)
    for feature_name, attr_name in METADATA_FEATURE_FLAGS:
        if bool(getattr(config.classifier, attr_name, False)):
            names.append(feature_name)
    return tuple(names)


def build_single_model_feature_batch(
    stories: list[Story],
    story_embeddings: NDArray[np.float32],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None,
    config: AppConfig,
    *,
    now: float | None = None,
) -> SingleModelFeatureBatch:
    if len(stories) != len(story_embeddings):
        raise ValueError("story count does not match embedding rows")

    embedding_dim = int(story_embeddings.shape[1]) if story_embeddings.ndim == 2 else 0
    if positive_embeddings is None:
        positive_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
    if negative_embeddings is None:
        negative_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)

    centroids: NDArray[np.float32]
    if len(positive_embeddings) > 0:
        centroids, _ = cluster_interests_with_labels(
            positive_embeddings,
            config=config.clustering,
        )
    else:
        centroids = np.zeros((0, embedding_dim), dtype=np.float32)

    derived = compute_classifier_similarity_features(
        story_embeddings,
        positive_embeddings,
        negative_embeddings,
        centroids,
        config.classifier,
    )
    derived_rows = stack_classifier_similarity_features(
        derived,
        config.classifier,
        base_embeddings=np.zeros((len(stories), 0), dtype=np.float32),
    )
    metadata_rows = _classifier_metadata_features(
        stories,
        config,
        now if now is not None else time.time(),
        len(stories),
    )

    columns = [story_embeddings.astype(np.float32)]
    if derived_rows.shape[1] > 0:
        columns.append(derived_rows.astype(np.float32))
    if metadata_rows.shape[1] > 0:
        columns.append(metadata_rows.astype(np.float32))
    rows = np.hstack(columns).astype(np.float32)
    feature_names = _feature_names(config, embedding_dim)
    if rows.shape[1] != len(feature_names):
        raise ValueError(
            f"feature width mismatch: rows={rows.shape[1]} names={len(feature_names)}"
        )

    return SingleModelFeatureBatch(
        rows=rows,
        feature_names=feature_names,
        derived_feature_dim=int(derived_rows.shape[1]),
        metadata_feature_dim=int(metadata_rows.shape[1]),
    )


def build_single_model_training_matrix(
    labels: list[SingleModelLabeledStory],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None,
    config: AppConfig,
    *,
    now: float | None = None,
) -> tuple[SingleModelFeatureBatch, NDArray[np.int64]]:
    stories = [item.story for item in labels]
    story_embeddings = get_embeddings([story.text_content for story in stories])
    batch = build_single_model_feature_batch(
        stories,
        story_embeddings,
        positive_embeddings,
        negative_embeddings,
        config,
        now=now,
    )
    y = np.asarray([item.label for item in labels], dtype=np.int64)
    return batch, y


def train_single_model(
    labels: list[SingleModelLabeledStory],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None,
    config: AppConfig,
    training_config: SingleModelConfig,
    *,
    now: float | None = None,
) -> tuple[OrdinalThresholdModel, SingleModelFeatureBatch]:
    batch, y = build_single_model_training_matrix(
        labels,
        positive_embeddings,
        negative_embeddings,
        config,
        now=now,
    )
    model = train_model_from_matrix(batch.rows, y, training_config)
    return model, batch


def train_single_model_from_embeddings(
    labels: list[SingleModelLabeledStory],
    story_embeddings: NDArray[np.float32],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None,
    config: AppConfig,
    training_config: SingleModelConfig,
    *,
    now: float | None = None,
) -> tuple[OrdinalThresholdModel, SingleModelFeatureBatch]:
    stories = [item.story for item in labels]
    batch = build_single_model_feature_batch(
        stories,
        story_embeddings,
        positive_embeddings,
        negative_embeddings,
        config,
        now=now,
    )
    y = np.asarray([item.label for item in labels], dtype=np.int64)
    model = train_model_from_matrix(batch.rows, y, training_config)
    return model, batch


def score_feature_rows(
    model: OrdinalThresholdModel,
    rows: NDArray[np.float32],
) -> NDArray[np.float32]:
    utility, _, _, _ = _predict_ordinal_outputs(model, rows)
    return utility


def score_feedback_labels_oof(
    labels: list[SingleModelLabeledStory],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None,
    config: AppConfig,
    training_config: SingleModelConfig,
    *,
    max_folds: int = 5,
) -> tuple[list[float], SingleModelFeatureBatch]:
    batch, y = build_single_model_training_matrix(
        labels,
        positive_embeddings,
        negative_embeddings,
        config,
    )
    positive_count = int(np.sum(y == UPVOTE_LABEL))
    negative_count = int(np.sum(y == DOWNVOTE_LABEL))
    fold_count = min(max_folds, positive_count, negative_count)
    if fold_count < 2:
        raise ValueError("need at least 2 upvote and 2 downvote labels for OOF scoring")

    y_binary = np.asarray([item.legacy_binary_label for item in labels], dtype=np.int64)
    scores = np.zeros(len(labels), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=0)
    for train_indices, test_indices in splitter.split(np.zeros(len(labels)), y_binary):
        model = train_model_from_matrix(
            batch.rows[train_indices],
            y[train_indices],
            training_config,
        )
        fold_scores = score_feature_rows(model, batch.rows[test_indices])
        scores[test_indices] = fold_scores
    return scores.astype(float).tolist(), batch


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


def _top_sources(
    labels: list[SingleModelLabeledStory],
    scores: list[float],
    k: int,
) -> dict[str, int]:
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
    labels: list[SingleModelLabeledStory],
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


def rank_stories_with_single_model(
    stories: list[Story],
    positive_stories: list[Story],
    negative_stories: list[Story],
    config: AppConfig,
    model: OrdinalThresholdModel,
) -> list[RankResult]:
    if not stories:
        return []

    candidate_embeddings = get_embeddings([story.text_content for story in stories])
    embedding_dim = int(candidate_embeddings.shape[1])
    positive_embeddings = (
        get_embeddings([story.text_content for story in positive_stories])
        if positive_stories
        else np.zeros((0, embedding_dim), dtype=np.float32)
    )
    negative_embeddings = (
        get_embeddings([story.text_content for story in negative_stories])
        if negative_stories
        else np.zeros((0, embedding_dim), dtype=np.float32)
    )
    batch = build_single_model_feature_batch(
        stories,
        candidate_embeddings,
        positive_embeddings,
        negative_embeddings,
        config,
    )
    utility = score_feature_rows(model, batch.rows)

    if len(positive_embeddings) > 0:
        pos_sim = cosine_similarity(candidate_embeddings, positive_embeddings)
        best_fav_indices = np.argmax(pos_sim, axis=1).astype(np.int64)
        max_sim_scores = np.max(pos_sim, axis=1).astype(np.float32)
    else:
        best_fav_indices = np.full(len(stories), -1, dtype=np.int64)
        max_sim_scores = np.zeros(len(stories), dtype=np.float32)

    if len(positive_embeddings) > 0:
        centroids, _ = cluster_interests_with_labels(
            positive_embeddings,
            config=config.clustering,
        )
    else:
        centroids = np.zeros((0, embedding_dim), dtype=np.float32)
    derived = compute_classifier_similarity_features(
        candidate_embeddings,
        positive_embeddings,
        negative_embeddings,
        centroids,
        config.classifier,
    )

    ranked_indices = np.argsort(-utility, kind="stable")
    return [
        RankResult(
            index=int(candidate_index),
            hybrid_score=float(utility[candidate_index]),
            best_fav_index=int(best_fav_indices[candidate_index]),
            max_sim_score=float(max_sim_scores[candidate_index]),
            knn_score=float(derived["pos_knn_feature"][candidate_index]),
            max_cluster_score=float(derived["centroid_feature"][candidate_index]),
            semantic_score=float(utility[candidate_index]),
        )
        for candidate_index in ranked_indices
    ]
