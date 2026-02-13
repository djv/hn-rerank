"""Tests for multi-seed promotion utilities."""

from __future__ import annotations

import json

import pytest

from promote_stable_params import (
    _collect_candidates,
    _is_stable,
    _latest_optuna_json,
    _parse_seed_list,
    _render_promoted_toml,
    _resolved_params,
    _score_metrics,
    SeedRun,
)


def test_parse_seed_list_dedups_preserves_order():
    assert _parse_seed_list("42, 43,42, 44") == [42, 43, 44]


def test_parse_seed_list_raises_on_empty():
    with pytest.raises(ValueError):
        _parse_seed_list(" , , ")


def test_latest_optuna_json_picks_most_recent(tmp_path):
    older = tmp_path / "optuna_20260205_120000.json"
    newer = tmp_path / "optuna_20260205_120100.json"
    older.write_text(json.dumps({"best_score": 0.1, "best_params": {"x": 1}}))
    newer.write_text(json.dumps({"best_score": 0.2, "best_params": {"x": 2}}))
    assert _latest_optuna_json(tmp_path) == newer


def test_score_metrics_penalizes_variance():
    stable = {
        "mrr": 0.5,
        "ndcg@10": 0.4,
        "ndcg@30": 0.3,
        "recall@50": 0.2,
        "mrr_std": 0.01,
        "ndcg@10_std": 0.01,
        "ndcg@30_std": 0.01,
        "recall@50_std": 0.01,
    }
    noisy = {**stable, "mrr_std": 0.20, "ndcg@10_std": 0.20, "ndcg@30_std": 0.20, "recall@50_std": 0.20}
    assert _score_metrics(stable) > _score_metrics(noisy)


def test_collect_candidates_dedups_identical_param_sets():
    params = {"diversity_lambda": 0.3, "neg_weight": 0.5}
    runs = [
        SeedRun(seed=42, run_dir="a", json_path="a.json", best_score=0.1, best_params=params),
        SeedRun(seed=43, run_dir="b", json_path="b.json", best_score=0.2, best_params=params),
    ]
    candidates = _collect_candidates(runs)
    assert len(candidates) == 1
    assert candidates[0][1] == params


def test_is_stable_true_when_gain_consistent():
    stable, mean_delta, _, lcb, regressions = _is_stable(
        deltas=[0.02, 0.015, 0.018],
        min_improvement=0.005,
        max_seed_regressions=0,
    )
    assert stable is True
    assert mean_delta > 0.005
    assert lcb > 0.0
    assert regressions == 0


def test_is_stable_false_when_regressions_exceed_limit():
    stable, _, _, _, regressions = _is_stable(
        deltas=[0.02, -0.01, 0.01],
        min_improvement=0.005,
        max_seed_regressions=0,
    )
    assert regressions == 1
    assert stable is False


def test_render_promoted_toml_contains_expected_sections():
    resolved = _resolved_params({"diversity_lambda": 0.25, "knn_k": 2})
    text = _render_promoted_toml(resolved)
    assert "[hn_rerank.ranking]" in text
    assert "[hn_rerank.adaptive_hn]" in text
    assert "[hn_rerank.freshness]" in text
    assert "[hn_rerank.semantic]" in text
    assert "[hn_rerank.classifier]" in text
