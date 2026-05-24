from __future__ import annotations

import numpy as np
import pytest

from api.config import AppConfig, ClassifierConfig, LearnedRankerConfig
from api.feedback import FeedbackRecord
from api.feedback_single_model import (
    build_single_model_feature_batch,
    build_single_model_feedback_labels,
    score_feature_rows,
    train_single_model,
)
from api.models import Story
from helpers import unit_rows


def _story(
    story_id: int,
    *,
    score: int = 10,
    comment_count: int | None = 2,
    text: str | None = None,
) -> Story:
    return Story(
        id=story_id,
        title=f"Story {story_id}",
        url=f"https://example.com/{story_id}",
        score=score,
        time=1_700_000_000 + story_id,
        text_content=text or f"Story {story_id}",
        comment_count=comment_count,
    )


def test_feature_builder_includes_embeddings_and_derived_features_without_ce() -> None:
    config = AppConfig(
        classifier=ClassifierConfig(
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=True,
            use_log_points_feature=True,
            use_log_comments_feature=True,
            use_comment_ratio_feature=True,
        )
    )
    stories = [_story(1, score=100, comment_count=5), _story(2, score=5, comment_count=1)]
    story_embeddings = unit_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    positive_embeddings = unit_rows([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]])
    negative_embeddings = unit_rows([[0.0, 1.0, 0.0]])

    batch = build_single_model_feature_batch(
        stories,
        story_embeddings,
        positive_embeddings,
        negative_embeddings,
        config,
        now=0.0,
    )

    assert batch.rows.shape == (2, 9)
    assert batch.feature_names[:3] == ("embedding_0", "embedding_1", "embedding_2")
    assert "centroid_feature" in batch.feature_names
    assert "pos_knn_feature" in batch.feature_names
    assert "neg_knn_feature" in batch.feature_names
    assert "log_points" in batch.feature_names
    assert "log_comments" in batch.feature_names
    assert "comment_ratio" in batch.feature_names
    assert "cross_encoder_score" not in batch.feature_names
    assert batch.derived_feature_dim == 3
    assert batch.metadata_feature_dim == 3
    assert np.all(np.isfinite(batch.rows))


def test_feedback_rows_do_not_require_rank_diagnostics_and_default_missing_metadata() -> None:
    records = {
        "a": FeedbackRecord(
            key="a",
            action="up",
            id=1,
            source="hn",
            title="A",
            url="https://example.com/a",
            discussion_url=None,
            text_content="Alpha",
            time=1_700_000_000,
            score=None,
            comment_count=None,
        ),
        "b": FeedbackRecord(
            key="b",
            action="down",
            id=2,
            source="hn",
            title="B",
            url="https://example.com/b",
            discussion_url=None,
            text_content="Beta",
            time=1_700_000_100,
            score=5,
            comment_count=0,
        ),
    }

    result = build_single_model_feedback_labels(records)

    assert result.skipped_count == 0
    assert [item.key for item in result.labels] == ["a", "b"]
    assert result.labels[0].story.score == 0
    assert result.labels[0].story.comment_count is None

    config = AppConfig(
        classifier=ClassifierConfig(
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=True,
            use_log_points_feature=True,
            use_log_comments_feature=True,
            use_comment_ratio_feature=True,
        )
    )
    batch = build_single_model_feature_batch(
        [item.story for item in result.labels],
        unit_rows([[1.0, 0.0], [0.0, 1.0]]),
        unit_rows([[1.0, 0.0]]),
        unit_rows([[0.0, 1.0]]),
        config,
        now=0.0,
    )

    assert batch.metadata_feature_dim == 3
    assert np.all(np.isfinite(batch.rows))


def test_single_model_scores_are_stable_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(
        classifier=ClassifierConfig(
            use_centroid_feature=True,
            use_pos_knn_feature=True,
            use_neg_knn_feature=True,
            use_log_points_feature=True,
            use_log_comments_feature=True,
            use_comment_ratio_feature=False,
        )
    )
    training_config = LearnedRankerConfig(
        min_positive_labels=2,
        min_negative_labels=2,
    )
    stories = [
        _story(1, score=100, comment_count=20, text="up-1"),
        _story(2, score=90, comment_count=18, text="up-2"),
        _story(3, score=50, comment_count=10, text="neutral"),
        _story(4, score=2, comment_count=0, text="down-1"),
        _story(5, score=1, comment_count=0, text="down-2"),
    ]
    labels_result = build_single_model_feedback_labels(
        {
            f"k{i}": FeedbackRecord(
                key=f"k{i}",
                action=action,
                id=story.id,
                source="hn",
                title=story.title,
                url=story.url,
                discussion_url=None,
                text_content=story.text_content,
                time=story.time,
                score=story.score,
                comment_count=story.comment_count,
            )
            for i, (story, action) in enumerate(
                zip(stories, ["up", "up", "neutral", "down", "down"], strict=True)
            )
        }
    )

    embedding_map = {
        "up-1": [1.0, 0.0],
        "up-2": [0.9, 0.1],
        "neutral": [0.5, 0.5],
        "down-1": [0.0, 1.0],
        "down-2": [0.1, 0.9],
        "pos-a": [1.0, 0.0],
        "pos-b": [0.95, 0.05],
        "neg-a": [0.0, 1.0],
        "neg-b": [0.05, 0.95],
    }

    def fake_get_embeddings(texts: list[str]) -> np.ndarray:
        return unit_rows([embedding_map[text] for text in texts])

    monkeypatch.setattr("api.feedback_single_model.get_embeddings", fake_get_embeddings)

    positive_stories = [_story(100, text="pos-a"), _story(101, text="pos-b")]
    negative_stories = [_story(200, text="neg-a"), _story(201, text="neg-b")]
    positive_embeddings = fake_get_embeddings([story.text_content for story in positive_stories])
    negative_embeddings = fake_get_embeddings([story.text_content for story in negative_stories])

    model, batch = train_single_model(
        labels_result.labels,
        positive_embeddings,
        negative_embeddings,
        config,
        training_config,
    )
    scores = score_feature_rows(model, batch.rows)

    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert scores[0] > scores[-1]
