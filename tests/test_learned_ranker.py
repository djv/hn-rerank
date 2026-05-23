from __future__ import annotations

import time

import numpy as np
import pytest

from api.config import AdaptiveHNConfig, AppConfig, FreshnessConfig, LearnedRankerConfig, RankingConfig
from api.feedback import FeedbackRecord
from api.learned_ranker import (
    DOWNVOTE_LABEL,
    FEATURE_NAMES,
    NEUTRAL_LABEL,
    UPVOTE_LABEL,
    LabeledStory,
    build_labels_from_feedback,
    build_features,
    compare_dashboard_feedback_configs,
    evaluate_labeled_order,
    score_ranked_results,
    split_temporal_holdout,
    train_model,
    train_or_load_and_score,
)
from api.models import RankResult, Story
from generate_html import apply_learned_ranker


def _story(
    story_id: int,
    *,
    source: str = "hn",
    score: int = 10,
    comment_count: int | None = 3,
    age_hours: float = 1.0,
) -> Story:
    now = time.time()
    return Story(
        id=story_id,
        title=f"Story {story_id}",
        url=f"https://example.com/{story_id}",
        score=score,
        time=int(now - age_hours * 3600),
        text_content=f"Story {story_id}",
        source=source,  # type: ignore[arg-type]
        comment_count=comment_count,
    )


def _rank_result(
    index: int,
    *,
    hybrid_score: float = 0.5,
    semantic_score: float = 0.5,
    learned_score: float = 0.0,
) -> RankResult:
    return RankResult(
        index=index,
        hybrid_score=hybrid_score,
        best_fav_index=-1,
        max_sim_score=semantic_score,
        knn_score=semantic_score,
        max_cluster_score=semantic_score,
        semantic_score=semantic_score,
        hn_score=0.1,
        freshness_boost=0.01,
        cross_encoder_score=0.2,
        learned_score=learned_score,
    )


def test_feature_names_and_values_are_stable_and_finite() -> None:
    story = _story(
        1,
        source="github_trending",
        score=42,
        comment_count=None,
        age_hours=2.0,
    )
    result = _rank_result(0, hybrid_score=0.8, semantic_score=0.7)

    features = build_features(story, result, now=time.time())

    assert len(features) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(features))
    assert features[FEATURE_NAMES.index("hybrid_score")] == pytest.approx(0.8)
    assert features[FEATURE_NAMES.index("semantic_score")] == pytest.approx(0.7)
    assert features[FEATURE_NAMES.index("log_comments")] == pytest.approx(0.0)


def test_train_model_requires_enough_labels() -> None:
    config = LearnedRankerConfig(min_positive_labels=2, min_negative_labels=2)

    with pytest.raises(ValueError, match="insufficient labels"):
        train_model(
            [
                LabeledStory(_story(1), UPVOTE_LABEL, _rank_result(-1)),
                LabeledStory(_story(2), DOWNVOTE_LABEL, _rank_result(-1)),
            ],
            config,
        )


def test_build_labels_from_feedback_carries_updated_at_and_metadata_flags() -> None:
    record = FeedbackRecord(
        key="https://example.com/story",
        action="up",
        id=1,
        source="hn",
        title="Story",
        url="https://example.com/story",
        discussion_url="https://news.ycombinator.com/item?id=1",
        text_content="Story",
        time=1700000000,
        score=123,
        comment_count=45,
        updated_at=1700003600.0,
        hybrid_score=0.8,
        semantic_score=0.7,
        hn_score=0.6,
        freshness_boost=0.05,
        knn_score=0.4,
        max_sim_score=0.5,
        max_cluster_score=0.55,
        cross_encoder_score=0.3,
    )
    neutral = FeedbackRecord(
        key="https://example.com/neutral",
        action="neutral",
        id=2,
        source="rss",
        title="Neutral",
        url="https://example.com/neutral",
        discussion_url=None,
        text_content="Neutral",
        time=1700000100,
        score=12,
        comment_count=1,
        updated_at=1700003700.0,
        hybrid_score=0.5,
        semantic_score=0.45,
        hn_score=0.2,
        freshness_boost=0.01,
        knn_score=0.3,
        max_sim_score=0.25,
        max_cluster_score=0.2,
        cross_encoder_score=0.15,
    )

    labels = build_labels_from_feedback({record.key: record, neutral.key: neutral})
    labels_by_id = {item.story.id: item for item in labels}

    assert len(labels) == 2
    assert labels_by_id[1].label == UPVOTE_LABEL
    assert labels_by_id[2].label == NEUTRAL_LABEL
    assert labels_by_id[1].feedback_updated_at == pytest.approx(1700003600.0)
    assert labels_by_id[1].has_raw_story_score is True
    assert labels_by_id[1].has_raw_comment_count is True
    assert labels_by_id[1].story.score == 123
    assert labels_by_id[1].story.comment_count == 45


def test_split_temporal_holdout_uses_newest_labels() -> None:
    labels = [
        LabeledStory(
            _story(i),
            UPVOTE_LABEL if i % 2 == 0 else DOWNVOTE_LABEL,
            _rank_result(-1),
            feedback_updated_at=1000.0 + i,
        )
        for i in range(8)
    ]

    train_labels, holdout_labels = split_temporal_holdout(
        labels,
        holdout_fraction=0.25,
        min_holdout_count=2,
        min_class_count=1,
    )

    assert [item.story.id for item in holdout_labels] == [6, 7]
    assert [item.story.id for item in train_labels] == [0, 1, 2, 3, 4, 5]


def test_train_score_save_and_load_round_trip(tmp_path) -> None:
    labels = [
        *(
            LabeledStory(
                _story(i, source="hn", score=100 + i, comment_count=10, age_hours=1),
                UPVOTE_LABEL,
                _rank_result(-1, hybrid_score=0.8, semantic_score=0.8),
            )
            for i in range(10)
        ),
        *(
            LabeledStory(
                _story(50 + i, source="rss", score=10, comment_count=2, age_hours=24),
                NEUTRAL_LABEL,
                _rank_result(-1, hybrid_score=0.5, semantic_score=0.5),
            )
            for i in range(4)
        ),
        *(
            LabeledStory(
                _story(100 + i, source="rss", score=1, comment_count=0, age_hours=200),
                DOWNVOTE_LABEL,
                _rank_result(-1, hybrid_score=0.2, semantic_score=0.2),
            )
            for i in range(10)
        ),
    ]
    config = LearnedRankerConfig(
        shadow_enabled=True,
        model_path=tmp_path / "ranker.joblib",
        min_positive_labels=5,
        min_negative_labels=5,
    )
    ranked = [_rank_result(0, hybrid_score=0.9), _rank_result(1, hybrid_score=0.1)]
    candidates = [_story(1000, score=90), _story(1001, source="rss", score=1)]

    first = train_or_load_and_score(
        ranked,
        candidates,
        labels,
        config,
    )
    assert first.mode == "trained"
    assert config.model_path.exists()
    assert set(first.scores) == {0, 1}
    assert all(0.0 <= score <= 1.0 for score in first.scores.values())

    loaded_config = LearnedRankerConfig(
        shadow_enabled=True,
        model_path=config.model_path,
        min_positive_labels=100,
        min_negative_labels=100,
    )
    second = train_or_load_and_score(
        ranked,
        candidates,
        [],
        loaded_config,
    )
    assert second.mode == "loaded"
    assert set(second.scores) == {0, 1}


def test_score_ranked_results_returns_candidate_index_scores() -> None:
    labels = [
        *(LabeledStory(_story(i, score=100), UPVOTE_LABEL, _rank_result(-1)) for i in range(10)),
        *(LabeledStory(_story(100 + i, source="rss", score=10), NEUTRAL_LABEL, _rank_result(-1)) for i in range(4)),
        *(
            LabeledStory(_story(200 + i, source="rss", score=1), DOWNVOTE_LABEL, _rank_result(-1))
            for i in range(10)
        ),
    ]
    config = LearnedRankerConfig(min_positive_labels=5, min_negative_labels=5)
    model = train_model(labels, config)
    ranked = [_rank_result(1), _rank_result(0)]
    candidates = [_story(300), _story(301)]

    scores = score_ranked_results(ranked, candidates, model)

    assert set(scores) == {0, 1}
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_evaluate_labeled_order_compares_learned_and_hybrid() -> None:
    labels = [
        *(
            LabeledStory(
                _story(i, source="hn", score=100 + i),
                UPVOTE_LABEL,
                _rank_result(-1, hybrid_score=0.2, semantic_score=0.9),
            )
            for i in range(6)
        ),
        *(
            LabeledStory(
                _story(50 + i, source="rss", score=10),
                NEUTRAL_LABEL,
                _rank_result(-1, hybrid_score=0.6, semantic_score=0.5),
            )
            for i in range(4)
        ),
        *(
            LabeledStory(
                _story(100 + i, source="rss", score=1),
                DOWNVOTE_LABEL,
                _rank_result(-1, hybrid_score=0.9, semantic_score=0.1),
            )
            for i in range(6)
        ),
    ]
    config = LearnedRankerConfig(
        min_positive_labels=2,
        min_negative_labels=2,
        source_feature_weight=0.0,
        balance_training_labels=True,
    )

    report = evaluate_labeled_order(labels, config, max_folds=3)

    assert report.label_count == 16
    assert report.positive_labels == 6
    assert report.neutral_labels == 4
    assert report.negative_labels == 6
    assert report.folds == 3
    assert report.learned_pairwise_accuracy > report.hybrid_pairwise_accuracy
    assert report.learned_precision_at_5 >= report.hybrid_precision_at_5
    assert report.learned_top_sources
    assert report.hybrid_top_sources


def test_compare_dashboard_feedback_configs_scores_newest_holdout() -> None:
    labels = [
        LabeledStory(
            _story(
                i,
                score=200 if i % 3 == 0 else 10,
                comment_count=60 if i % 3 == 0 else (12 if i % 3 == 1 else 2),
            ),
            UPVOTE_LABEL if i % 3 == 0 else (NEUTRAL_LABEL if i % 3 == 1 else DOWNVOTE_LABEL),
            _rank_result(-1, hybrid_score=0.5, semantic_score=0.5),
            feedback_updated_at=10_000.0 + i,
            has_raw_story_score=True,
            has_raw_comment_count=True,
        )
        for i in range(12)
    ]
    current = AppConfig(
        ranking=RankingConfig(non_semantic_weight=0.0, comment_ratio=0.0),
        freshness=FreshnessConfig(enabled=False),
        adaptive_hn=AdaptiveHNConfig(
            weight_min=1.0,
            weight_max=1.0,
            threshold_young=0.0,
            threshold_old=1.0,
            score_normalization_cap=10.0,
        ),
    )
    candidate = AppConfig(
        ranking=RankingConfig(non_semantic_weight=0.8, comment_ratio=0.0),
        freshness=FreshnessConfig(enabled=False),
        adaptive_hn=AdaptiveHNConfig(
            weight_min=1.0,
            weight_max=1.0,
            threshold_young=0.0,
            threshold_old=1.0,
            score_normalization_cap=10.0,
        ),
    )

    result = compare_dashboard_feedback_configs(
        labels,
        current,
        candidate,
        holdout_fraction=0.5,
        min_holdout_count=6,
        min_class_count=2,
    )

    assert result.summary.holdout_label_count == 6
    assert result.summary.train_label_count == 6
    assert result.summary.holdout_neutral_labels >= 1
    assert result.candidate.pairwise_accuracy > result.incumbent.pairwise_accuracy
    assert result.score_delta > 0.0
    assert result.passed is True


def test_compare_dashboard_feedback_configs_skips_hn_labels_missing_metadata() -> None:
    usable = [
        LabeledStory(
            _story(
                i,
                score=120 if i % 2 == 0 else 5,
                comment_count=30 if i % 2 == 0 else 1,
            ),
            UPVOTE_LABEL if i % 2 == 0 else DOWNVOTE_LABEL,
            _rank_result(-1, hybrid_score=0.5, semantic_score=0.5),
            feedback_updated_at=20_000.0 + i,
            has_raw_story_score=True,
            has_raw_comment_count=True,
        )
        for i in range(6)
    ]
    missing = LabeledStory(
        _story(100, score=0, comment_count=None),
        NEUTRAL_LABEL,
        _rank_result(-1, hybrid_score=0.4, semantic_score=0.4),
        feedback_updated_at=19_999.0,
        has_raw_story_score=False,
        has_raw_comment_count=False,
    )
    current = AppConfig(freshness=FreshnessConfig(enabled=False))

    result = compare_dashboard_feedback_configs(
        [missing, *usable],
        current,
        current,
        holdout_fraction=0.5,
        min_holdout_count=3,
        min_class_count=1,
    )

    assert result.summary.label_count == 7
    assert result.summary.usable_label_count == 6
    assert result.summary.skipped_missing_story_metadata == 1


def test_apply_learned_ranker_shadow_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = [_rank_result(0, hybrid_score=0.1), _rank_result(1, hybrid_score=0.9)]
    cands = [_story(1), _story(2)]
    config = AppConfig(
        learned_ranker=LearnedRankerConfig(shadow_enabled=True, active_enabled=False)
    )

    def fake_score(*_args, **_kwargs):
        from api.learned_ranker import LearnedRankerResult

        return LearnedRankerResult(
            mode="trained",
            scores={0: 0.99, 1: 0.01},
            positive_labels=10,
            neutral_labels=3,
            negative_labels=10,
        )

    monkeypatch.setattr("generate_html.train_or_load_and_score", fake_score)

    updated, mode = apply_learned_ranker(ranked, cands, [], config)

    assert mode == "trained"
    assert [result.index for result in updated] == [0, 1]
    assert updated[0].learned_score == pytest.approx(0.99)
    assert not updated[0].learned_ranker_used


def test_apply_learned_ranker_active_reorders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = [_rank_result(1, hybrid_score=0.9), _rank_result(0, hybrid_score=0.1)]
    cands = [_story(1), _story(2)]
    config = AppConfig(
        learned_ranker=LearnedRankerConfig(shadow_enabled=True, active_enabled=True)
    )

    def fake_score(*_args, **_kwargs):
        from api.learned_ranker import LearnedRankerResult

        return LearnedRankerResult(
            mode="trained",
            scores={0: 0.99, 1: 0.01},
            positive_labels=10,
            neutral_labels=3,
            negative_labels=10,
        )

    monkeypatch.setattr("generate_html.train_or_load_and_score", fake_score)

    updated, mode = apply_learned_ranker(ranked, cands, [], config)

    assert mode == "trained"
    assert [result.index for result in updated] == [0, 1]
    assert updated[0].learned_ranker_used
