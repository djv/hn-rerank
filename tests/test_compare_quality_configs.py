from __future__ import annotations

import pytest

from scripts.compare_quality_configs import (
    average_metric_runs,
    compare_metrics,
    parse_seeds,
)


def test_parse_seeds_deduplicates_blank_entries() -> None:
    assert parse_seeds("0, 1,,2") == [0, 1, 2]
    assert parse_seeds(" , ") == [0]


def test_average_metric_runs_averages_union_of_keys() -> None:
    assert average_metric_runs(
        [
            {"mrr": 1.0, "ndcg@10": 0.5},
            {"mrr": 0.5, "ndcg@10": 1.0},
        ]
    ) == {"mrr": pytest.approx(0.75), "ndcg@10": pytest.approx(0.75)}


def test_compare_metrics_requires_relative_gain_and_no_regressions() -> None:
    baseline = {
        "mrr": 0.5,
        "ndcg@10": 0.5,
        "ndcg@20": 0.5,
        "ndcg@30": 0.5,
        "precision@20": 0.5,
        "recall@30": 0.5,
    }
    candidate = {
        "mrr": 0.6,
        "ndcg@10": 0.6,
        "ndcg@20": 0.6,
        "ndcg@30": 0.6,
        "precision@20": 0.6,
        "recall@30": 0.6,
    }

    result = compare_metrics(
        baseline,
        candidate,
        min_relative_improvement=0.05,
        std_penalty=0.5,
    )

    assert result.passed is True
    assert result.relative_delta >= 0.05


def test_compare_metrics_rejects_guard_regression_even_with_score_gain() -> None:
    baseline = {
        "mrr": 0.5,
        "ndcg@10": 0.5,
        "ndcg@20": 0.5,
        "ndcg@30": 0.5,
        "precision@20": 0.5,
        "recall@30": 0.5,
    }
    candidate = {
        "mrr": 0.8,
        "ndcg@10": 0.8,
        "ndcg@20": 0.8,
        "ndcg@30": 0.4,
        "precision@20": 0.8,
        "recall@30": 0.8,
    }

    result = compare_metrics(
        baseline,
        candidate,
        min_relative_improvement=0.05,
        std_penalty=0.5,
    )

    assert result.passed is False
    assert result.guard_failures == ["ndcg@30"]
