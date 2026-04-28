import numpy as np
from unittest.mock import patch

from api.models import RankResult, Story
from evaluate_quality import (
    EvaluationDataset,
    RankingEvaluator,
    _finalize_ranked_results,
)


def _story(sid: int) -> Story:
    return Story(
        id=sid,
        title=f"Story {sid}",
        url=f"https://example.com/{sid}",
        score=sid,
        time=1700000000 + sid,
        comments=[f"Comment {sid}"],
        text_content=f"Story {sid} text",
    )


def test_snapshot_round_trip_recomputes_embeddings(tmp_path):
    train = [_story(1), _story(2)]
    test = [_story(3)]
    neg = [_story(4)]
    candidates = [_story(5), _story(3)]

    evaluator = RankingEvaluator("snapshot_user")
    evaluator.dataset = EvaluationDataset(
        train_stories=train,
        test_stories=test,
        neg_stories=neg,
        candidates=candidates,
        train_embeddings=np.ones((2, 3), dtype=np.float32),
        neg_embeddings=np.ones((1, 3), dtype=np.float32),
        test_ids={3},
    )
    evaluator.snapshot_metadata = {"source": "test"}

    snapshot_path = tmp_path / "benchmark.json"
    evaluator.save_snapshot(snapshot_path, metadata={"note": "unit"})

    loaded = RankingEvaluator("other_user")
    with patch(
        "evaluate_quality.get_embeddings",
        side_effect=[
            np.full((2, 3), 2.0, dtype=np.float32),
            np.full((1, 3), 3.0, dtype=np.float32),
        ],
    ) as mock_get_embeddings:
        assert loaded.load_snapshot(snapshot_path) is True

    assert mock_get_embeddings.call_count == 2
    assert loaded.username == "snapshot_user"
    assert loaded.dataset is not None
    assert [story.id for story in loaded.dataset.train_stories] == [1, 2]
    assert [story.id for story in loaded.dataset.test_stories] == [3]
    assert [story.id for story in loaded.dataset.neg_stories] == [4]
    assert [story.id for story in loaded.dataset.candidates] == [5, 3]
    assert loaded.dataset.test_ids == {3}
    assert loaded.dataset.neg_embeddings is not None
    assert np.allclose(loaded.dataset.train_embeddings, 2.0)
    assert np.allclose(loaded.dataset.neg_embeddings, 3.0)
    assert loaded.snapshot_metadata["source"] == "snapshot"
    assert loaded.snapshot_metadata["note"] == "unit"


def test_load_snapshot_rejects_missing_story_sets(tmp_path):
    snapshot_path = tmp_path / "broken.json"
    snapshot_path.write_text('{"format_version": 1, "username": "u", "train_stories": [], "test_stories": [], "neg_stories": [], "candidates": []}')

    evaluator = RankingEvaluator("u")
    assert evaluator.load_snapshot(snapshot_path) is False


def test_finalize_ranked_results_applies_render_dedup_without_refill():
    candidates = [
        Story(
            id=1,
            title="Same URL A",
            url="https://example.com/post",
            score=10,
            time=1,
            text_content="a",
        ),
        Story(
            id=2,
            title="Same URL B",
            url="https://example.com/post/",
            score=9,
            time=1,
            text_content="b",
        ),
        Story(
            id=3,
            title="Unique",
            url="https://example.com/unique",
            score=8,
            time=1,
            text_content="c",
        ),
    ]
    results = [
        RankResult(index=0, hybrid_score=0.9, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
        RankResult(index=1, hybrid_score=0.8, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
        RankResult(index=2, hybrid_score=0.7, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
    ]

    finalized = _finalize_ranked_results(results, candidates, count=3)

    assert [candidates[result.index].id for result in finalized] == [1, 3]


def test_finalize_ranked_results_keeps_distinct_query_identified_urls():
    candidates = [
        Story(
            id=1,
            title="Query Story A",
            url="https://example.com/post?id=1&utm_source=hn",
            score=10,
            time=1,
            text_content="a",
        ),
        Story(
            id=2,
            title="Query Story B",
            url="https://example.com/post?id=2",
            score=9,
            time=1,
            text_content="b",
        ),
        Story(
            id=3,
            title="Unique",
            url="https://example.com/unique",
            score=8,
            time=1,
            text_content="c",
        ),
    ]
    results = [
        RankResult(index=0, hybrid_score=0.9, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
        RankResult(index=1, hybrid_score=0.8, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
        RankResult(index=2, hybrid_score=0.7, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
    ]

    finalized = _finalize_ranked_results(results, candidates, count=3)

    assert [candidates[result.index].id for result in finalized] == [1, 2, 3]


def test_evaluate_can_score_final_displayed_list():
    train = [_story(1), _story(2)]
    test = [_story(3)]
    candidates = [_story(10), _story(11), _story(12)]

    evaluator = RankingEvaluator("snapshot_user")
    evaluator.dataset = EvaluationDataset(
        train_stories=train,
        test_stories=test,
        neg_stories=[],
        candidates=candidates,
        train_embeddings=np.ones((2, 3), dtype=np.float32),
        neg_embeddings=None,
        test_ids={12},
    )

    ranked = [
        RankResult(index=0, hybrid_score=0.9, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
        RankResult(index=1, hybrid_score=0.8, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
        RankResult(index=2, hybrid_score=0.7, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
    ]

    with (
        patch("evaluate_quality.rank_stories", return_value=ranked),
        patch("evaluate_quality._finalize_ranked_results", return_value=[ranked[2], ranked[0]]),
    ):
        metrics = evaluator.evaluate(k_metrics=[1, 2], final_list_count=2)

    assert metrics["mrr"] == 1.0
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@2"] == 1.0


def test_evaluate_cv_populates_diagnostics_summary():
    train = [_story(1), _story(2), _story(3), _story(4)]
    test = [_story(5), _story(6)]
    candidates = [_story(10), _story(11), _story(12), _story(13)]
    evaluator = RankingEvaluator("snapshot_user")
    evaluator.dataset = EvaluationDataset(
        train_stories=train,
        test_stories=test,
        neg_stories=[_story(20), _story(21)],
        candidates=candidates,
        train_embeddings=np.ones((4, 3), dtype=np.float32),
        neg_embeddings=np.ones((2, 3), dtype=np.float32),
        test_ids={12},
    )

    def _fake_rank_stories(*args, **kwargs):
        diagnostics = kwargs.get("diagnostics")
        if diagnostics is not None:
            diagnostics.update(
                {
                    "classifier_requested": True,
                    "classifier_used": True,
                    "classifier_failure_reason": None,
                    "positive_count": 4,
                    "negative_count": 2,
                    "base_feature_dim": 3,
                    "derived_feature_dim": 2,
                    "local_hidden_penalty_applied": False,
                    "local_hidden_penalty_mean": 0.0,
                    "local_hidden_penalty_max": 0.0,
                }
            )
        return [
            RankResult(index=0, hybrid_score=0.9, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
            RankResult(index=1, hybrid_score=0.8, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
            RankResult(index=2, hybrid_score=0.7, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
            RankResult(index=3, hybrid_score=0.6, best_fav_index=-1, max_sim_score=0.0, knn_score=0.0),
        ]

    diagnostics_summary: dict[str, object] = {}
    with patch("evaluate_quality.get_embeddings", return_value=np.ones((6, 3), dtype=np.float32)):
        with patch("evaluate_quality.rank_stories", side_effect=_fake_rank_stories):
            evaluator.evaluate_cv(
                n_folds=2,
                k_metrics=[10],
                report_each=False,
                parallel=False,
                diagnostics_summary=diagnostics_summary,
            )

    assert diagnostics_summary["rank_calls"] == 2
    assert diagnostics_summary["classifier_requested_count"] == 2
    assert diagnostics_summary["classifier_used_count"] == 2
    assert diagnostics_summary["classifier_used_rate"] == 1.0
    assert diagnostics_summary["avg_derived_feature_dim"] == 2.0
