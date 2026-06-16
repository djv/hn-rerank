"""Feedback-trained single ranking model built from dashboard feedback labels."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from api.config import AppConfig, SingleModelConfig
from api.feedback import FeedbackRecord
from api.ordinal_model import (
    ACTION_TO_ORDINAL,
    DOWNVOTE_LABEL,
    UPVOTE_LABEL,
    OrdinalThresholdModel,
    _predict_ordinal_outputs,
    train_model_from_matrix,
)
from api.models import Story
from api.rerank import (
    _classifier_metadata_features,
    cluster_interests_with_labels,
    compute_classifier_similarity_features,
    get_embeddings,
    stack_similarity_features,
    _populate_rank_cache_metadata,
    SIMILARITY_FEATURES,
    METADATA_FEATURES,
)

# --- Feature flag tables removed ---
# Feature names are derived from the registries in api/rerank.py.


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
    names: list[str] = []
    if config.classifier.raw_embedding_features:
        names.extend(f"embedding_{index}" for index in range(embedding_dim))
    for f in config.classifier.features:
        if f in SIMILARITY_FEATURES or f in METADATA_FEATURES:
            names.append(f)
    return tuple(names)


def build_single_model_feature_batch(
    stories: list[Story],
    story_embeddings: NDArray[np.float32],
    positive_embeddings: NDArray[np.float32] | None,
    negative_embeddings: NDArray[np.float32] | None,
    config: AppConfig,
    *,
    now: float | None = None,
    exclude_self_pos: bool = False,
    exclude_self_neg: bool = False,
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
        exclude_self_pos=exclude_self_pos,
        exclude_self_neg=exclude_self_neg,
    )
    derived_rows = stack_similarity_features(derived, config.classifier)
    metadata_rows = _classifier_metadata_features(
        stories,
        config,
        now if now is not None else time.time(),
        len(stories),
    )

    columns: list[NDArray[np.float32]] = []
    if config.classifier.raw_embedding_features:
        columns.append(story_embeddings.astype(np.float32))
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
    positive_stories = [item.story for item in labels if item.label == UPVOTE_LABEL]
    negative_stories = [item.story for item in labels if item.label == DOWNVOTE_LABEL]
    eval_now = now if now is not None else time.time()
    _populate_rank_cache_metadata(positive_stories, negative_stories, eval_now)

    story_embeddings = get_embeddings([story.text_content for story in stories])
    batch = build_single_model_feature_batch(
        stories,
        story_embeddings,
        positive_embeddings,
        negative_embeddings,
        config,
        now=now,
        exclude_self_pos=True,
        exclude_self_neg=True,
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
    positive_stories = [item.story for item in labels if item.label == UPVOTE_LABEL]
    negative_stories = [item.story for item in labels if item.label == DOWNVOTE_LABEL]
    eval_now = now if now is not None else time.time()
    _populate_rank_cache_metadata(positive_stories, negative_stories, eval_now)

    batch = build_single_model_feature_batch(
        stories,
        story_embeddings,
        positive_embeddings,
        negative_embeddings,
        config,
        now=now,
        exclude_self_pos=True,
        exclude_self_neg=True,
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
