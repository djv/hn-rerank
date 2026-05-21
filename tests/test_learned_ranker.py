from __future__ import annotations

import time

import numpy as np
import pytest

from api.config import AppConfig, LearnedRankerConfig
from api.learned_ranker import (
    FEATURE_NAMES,
    LabeledStory,
    balance_labels,
    build_features,
    build_training_matrix,
    evaluate_labeled_order,
    score_ranked_results,
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
    assert features[FEATURE_NAMES.index("is_hn")] == pytest.approx(0.0)
    assert features[FEATURE_NAMES.index("is_external")] == pytest.approx(1.0)
    assert features[FEATURE_NAMES.index("is_github_trending")] == pytest.approx(1.0)


def test_train_model_requires_enough_labels() -> None:
    config = LearnedRankerConfig(min_positive_labels=2, min_negative_labels=2)

    with pytest.raises(ValueError, match="insufficient labels"):
        train_model(
            [
                LabeledStory(_story(1), 1, _rank_result(-1)),
                LabeledStory(_story(2), 0, _rank_result(-1)),
            ],
            config,
        )


def test_source_features_can_be_zeroed_for_training() -> None:
    labels = [
        LabeledStory(
            _story(1, source="github_trending"),
            1,
            _rank_result(-1),
        )
    ]

    matrix, _ = build_training_matrix(labels, source_feature_weight=0.0)

    for name in (
        "is_hn",
        "is_external",
        "is_github_trending",
        "is_reddit",
        "is_curated_external",
    ):
        assert matrix[0][FEATURE_NAMES.index(name)] == pytest.approx(0.0)


def test_balance_labels_downsamples_majority_class() -> None:
    labels = [
        *(LabeledStory(_story(i), 1, _rank_result(-1)) for i in range(2)),
        *(LabeledStory(_story(100 + i), 0, _rank_result(-1)) for i in range(5)),
    ]

    balanced = balance_labels(labels)

    assert len(balanced) == 4
    assert sum(item.label == 1 for item in balanced) == 2
    assert sum(item.label == 0 for item in balanced) == 2


def test_train_score_save_and_load_round_trip(tmp_path) -> None:
    labels = [
        *(
            LabeledStory(
                _story(i, source="hn", score=100 + i, comment_count=10, age_hours=1),
                1,
                _rank_result(-1, hybrid_score=0.8, semantic_score=0.8),
            )
            for i in range(10)
        ),
        *(
            LabeledStory(
                _story(100 + i, source="rss", score=1, comment_count=0, age_hours=200),
                0,
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
        *(LabeledStory(_story(i, score=100), 1, _rank_result(-1)) for i in range(10)),
        *(
            LabeledStory(_story(100 + i, source="rss", score=1), 0, _rank_result(-1))
            for i in range(10)
        ),
    ]
    config = LearnedRankerConfig(min_positive_labels=5, min_negative_labels=5)
    model = train_model(labels, config)
    ranked = [_rank_result(1), _rank_result(0)]
    candidates = [_story(200), _story(201)]

    scores = score_ranked_results(ranked, candidates, model)

    assert set(scores) == {0, 1}
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_evaluate_labeled_order_compares_learned_and_hybrid() -> None:
    labels = [
        *(
            LabeledStory(
                _story(i, source="hn", score=100 + i),
                1,
                _rank_result(-1, hybrid_score=0.2, semantic_score=0.8),
            )
            for i in range(6)
        ),
        *(
            LabeledStory(
                _story(100 + i, source="rss", score=1),
                0,
                _rank_result(-1, hybrid_score=0.9, semantic_score=0.2),
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

    assert report.label_count == 12
    assert report.folds == 3
    assert report.learned_pairwise_accuracy > report.hybrid_pairwise_accuracy
    assert report.learned_precision_at_5 >= report.hybrid_precision_at_5
    assert report.learned_top_sources
    assert report.hybrid_top_sources


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
            negative_labels=10,
        )

    monkeypatch.setattr("generate_html.train_or_load_and_score", fake_score)

    updated, mode = apply_learned_ranker(ranked, cands, [], config)

    assert mode == "trained"
    assert [result.index for result in updated] == [0, 1]
    assert updated[0].learned_ranker_used
