"""Tests for optimize_hyperparameters scoring and constraint helpers."""

import json
import time

import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

from optimize_hyperparameters import (
    HN_THRESHOLD_GAP,
    ADAPTIVE_HN_DELTA,
    _build_ranges,
    _derive_classifier_diversity_lambda,
    _derive_hn_threshold_old,
    _derive_adaptive_hn_max,
    _parse_last_log,
)

# ---------------------------------------------------------------------------
# Derivation functions — property-based
# ---------------------------------------------------------------------------

CLASSIFIER_FLOOR = 0.30


class TestDeriveClassifierDiversityLambda:
    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_floor_and_passthrough(self, x: float):
        result = _derive_classifier_diversity_lambda(x)
        assert result >= CLASSIFIER_FLOOR
        assert result >= x


class TestDeriveHnThresholdOld:
    @given(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    def test_exact_gap_above_young(self, young: float):
        old = _derive_hn_threshold_old(young)
        assert old == pytest.approx(young + HN_THRESHOLD_GAP)


class TestDeriveAdaptiveHnMax:
    @given(st.floats(min_value=0.0, max_value=0.5, allow_nan=False))
    def test_exact_delta_above_base(self, base: float):
        result = _derive_adaptive_hn_max(base)
        assert result == pytest.approx(base + ADAPTIVE_HN_DELTA)


# ---------------------------------------------------------------------------
# _build_ranges
# ---------------------------------------------------------------------------

# Canonical default ranges (source of truth for tests)
_DEFAULTS_CORE = _build_ranges(None, "core")
_DEFAULTS_FULL = _build_ranges(None, "full")


class TestBuildRangesDefaults:
    def test_no_collapsed_keys(self):
        removed = {"classifier_diversity_lambda", "hn_threshold_old", "adaptive_hn_max"}
        assert removed.isdisjoint(_DEFAULTS_CORE), f"Stale keys: {removed & _DEFAULTS_CORE.keys()}"
        assert removed.isdisjoint(_DEFAULTS_FULL), f"Stale keys: {removed & _DEFAULTS_FULL.keys()}"

    def test_param_counts(self):
        assert len(_DEFAULTS_CORE) == 9, f"core got {len(_DEFAULTS_CORE)}: {sorted(_DEFAULTS_CORE)}"
        assert len(_DEFAULTS_FULL) == 12, f"full got {len(_DEFAULTS_FULL)}: {sorted(_DEFAULTS_FULL)}"

    def test_core_is_subset_of_full(self):
        assert _DEFAULTS_CORE.keys() <= _DEFAULTS_FULL.keys()

    def test_core_drops_expected_keys(self):
        dropped = {"freshness_boost", "knn_sigmoid_k", "knn_maxsim_weight"}
        assert dropped.isdisjoint(_DEFAULTS_CORE)
        assert dropped <= _DEFAULTS_FULL.keys()

    def test_all_lo_lt_hi(self):
        for key, (lo, hi) in _DEFAULTS_CORE.items():
            assert lo < hi, f"{key}: lo={lo} >= hi={hi}"
        for key, (lo, hi) in _DEFAULTS_FULL.items():
            assert lo < hi, f"{key}: lo={lo} >= hi={hi}"


# Strategy: generate a valid prev_best dict (subset of default keys, values within bounds)
_prev_best_st = st.fixed_dictionaries(
    {},
    optional={
        k: st.floats(min_value=lo, max_value=hi, allow_nan=False)
        for k, (lo, hi) in _DEFAULTS_FULL.items()
    },
)


class TestBuildRangesNarrowing:
    @given(_prev_best_st)
    def test_narrowed_ranges_within_defaults(self, prev: dict[str, float]):
        assume(len(prev) > 0)
        ranges = _build_ranges(prev, "full")
        for key, (lo, hi) in ranges.items():
            def_lo, def_hi = _DEFAULTS_FULL[key]
            assert lo >= def_lo - 1e-9, f"{key}: lo {lo} < default lo {def_lo}"
            assert hi <= def_hi + 1e-9, f"{key}: hi {hi} > default hi {def_hi}"
            assert lo < hi, f"{key}: lo={lo} >= hi={hi}"

    @given(_prev_best_st)
    def test_key_set_unchanged(self, prev: dict[str, float]):
        ranges = _build_ranges(prev, "full")
        assert ranges.keys() == _DEFAULTS_FULL.keys()

    def test_extra_keys_in_prev_ignored(self):
        prev = {"nonexistent_param": 99.0}
        ranges = _build_ranges(prev, "full")
        assert "nonexistent_param" not in ranges
        assert ranges.keys() == _DEFAULTS_FULL.keys()

    def test_none_returns_defaults(self):
        assert _build_ranges(None, "core") == _DEFAULTS_CORE
        assert _build_ranges(None, "full") == _DEFAULTS_FULL


# ---------------------------------------------------------------------------
# _parse_last_log
# ---------------------------------------------------------------------------


class TestParseLastLog:
    def test_valid_log(self, tmp_path):
        log = tmp_path / "optuna_20260205_120000.log"
        log.write_text(
            "Optimization Complete! (50 trials, 3-fold CV)\n"
            "Best Combined Score: 0.4321\n"
            "Best Parameters:\n"
            "  diversity_lambda: 0.350000\n"
            "  neg_weight: 0.550000\n"
            "  knn_k: 2\n"
        )
        result = _parse_last_log(tmp_path)
        assert result is not None
        assert result["diversity_lambda"] == pytest.approx(0.35)
        assert result["neg_weight"] == pytest.approx(0.55)
        assert result["knn_k"] == pytest.approx(2.0)

    def test_empty_dir(self, tmp_path):
        assert _parse_last_log(tmp_path) is None

    def test_log_without_best_params_section(self, tmp_path):
        log = tmp_path / "optuna_20260205_120000.log"
        log.write_text("Optimization Complete!\nNo best params here.\n")
        assert _parse_last_log(tmp_path) is None

    def test_non_float_values_skipped(self, tmp_path):
        log = tmp_path / "optuna_20260205_120000.log"
        log.write_text(
            "Best Parameters:\n"
            "  good_param: 1.5\n"
            "  bad_param: not_a_number\n"
        )
        result = _parse_last_log(tmp_path)
        assert result is not None
        assert "good_param" in result
        assert "bad_param" not in result

    def test_picks_most_recent_log(self, tmp_path):
        old = tmp_path / "optuna_20260101_000000.log"
        old.write_text("Best Parameters:\n  x: 1.0\n")
        new = tmp_path / "optuna_20260205_000000.log"
        new.write_text("Best Parameters:\n  x: 2.0\n")
        result = _parse_last_log(tmp_path)
        assert result is not None
        assert result["x"] == pytest.approx(2.0)

    def test_empty_params_section_returns_none(self, tmp_path):
        log = tmp_path / "optuna_20260205_120000.log"
        log.write_text("Best Parameters:\n\nSome other section\n")
        assert _parse_last_log(tmp_path) is None

    def test_json_fallback_when_log_incomplete(self, tmp_path):
        log = tmp_path / "optuna_20260205_120000.log"
        log.write_text("Optimization in progress...\n")
        json_path = tmp_path / "optuna_20260205_120000.json"
        json_path.write_text(json.dumps({
            "best_params": {"diversity_lambda": 0.42, "knn_k": 3},
        }))
        result = _parse_last_log(tmp_path)
        assert result is not None
        assert result["diversity_lambda"] == pytest.approx(0.42)
        assert result["knn_k"] == pytest.approx(3.0)

    def test_json_malformed_skipped(self, tmp_path):
        bad = tmp_path / "optuna_20260205_120000.json"
        bad.write_text("{not valid json")
        assert _parse_last_log(tmp_path) is None

    def test_json_missing_best_params_skipped(self, tmp_path):
        j = tmp_path / "optuna_20260205_120000.json"
        j.write_text(json.dumps({"other_key": 1}))
        assert _parse_last_log(tmp_path) is None

    def test_log_preferred_over_json(self, tmp_path):
        log = tmp_path / "optuna_20260205_120000.log"
        log.write_text("Best Parameters:\n  x: 1.0\n")
        j = tmp_path / "optuna_20260205_120000.json"
        j.write_text(json.dumps({"best_params": {"x": 2.0}}))
        result = _parse_last_log(tmp_path)
        assert result is not None
        assert result["x"] == pytest.approx(1.0), "Log should take priority over JSON"

    def test_skips_incomplete_log_uses_next(self, tmp_path):
        old = tmp_path / "optuna_20260101_000000.log"
        old.write_text("Best Parameters:\n  x: 5.0\n")
        old.touch()
        time.sleep(0.05)
        new = tmp_path / "optuna_20260205_000000.log"
        new.write_text("Optimization in progress...\n")
        new.touch()
        result = _parse_last_log(tmp_path)
        assert result is not None
        assert result["x"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# score_metrics (replicated closure — formula must match main())
# ---------------------------------------------------------------------------

_WEIGHTS = {"mrr": 0.30, "ndcg@10": 0.35, "ndcg@30": 0.20, "recall@50": 0.15}


def _score(metrics: dict[str, float]) -> float:
    mean = sum(w * metrics.get(k, 0) for k, w in _WEIGHTS.items())
    std = sum(w * metrics.get(f"{k}_std", 0) for k, w in _WEIGHTS.items())
    return mean - 0.5 * std


_metric_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_full_metrics_st = st.fixed_dictionaries({
    "mrr": _metric_st, "ndcg@10": _metric_st,
    "ndcg@30": _metric_st, "recall@50": _metric_st,
    "mrr_std": _metric_st, "ndcg@10_std": _metric_st,
    "ndcg@30_std": _metric_st, "recall@50_std": _metric_st,
})


class TestScoreMetrics:
    def test_variance_penalizes(self):
        base = {"mrr": 0.5, "ndcg@10": 0.4, "ndcg@30": 0.3, "recall@50": 0.6}
        stable = {**base, "mrr_std": 0.01, "ndcg@10_std": 0.02,
                  "ndcg@30_std": 0.01, "recall@50_std": 0.02}
        noisy = {**base, "mrr_std": 0.15, "ndcg@10_std": 0.20,
                 "ndcg@30_std": 0.15, "recall@50_std": 0.20}
        assert _score(stable) > _score(noisy)

    def test_zero_std_no_penalty(self):
        m = {"mrr": 0.5, "ndcg@10": 0.4, "ndcg@30": 0.3, "recall@50": 0.6}
        expected = 0.30 * 0.5 + 0.35 * 0.4 + 0.20 * 0.3 + 0.15 * 0.6
        assert _score(m) == pytest.approx(expected)

    @given(_full_metrics_st)
    def test_score_bounded(self, metrics: dict[str, float]):
        """Score with metrics in [0,1] must be in [-0.5, 1.0]."""
        s = _score(metrics)
        assert -0.5 <= s <= 1.0

    @given(_full_metrics_st)
    def test_higher_mean_higher_score(self, metrics: dict[str, float]):
        """Increasing all means by epsilon increases score."""
        boosted = {k: v + 0.01 if "_std" not in k else v for k, v in metrics.items()}
        assert _score(boosted) > _score(metrics)

    @given(_full_metrics_st)
    def test_higher_std_lower_score(self, metrics: dict[str, float]):
        """Increasing all stds by epsilon decreases score."""
        noisier = {k: v + 0.01 if "_std" in k else v for k, v in metrics.items()}
        assert _score(noisier) < _score(metrics)

    def test_weights_sum_to_one(self):
        assert sum(_WEIGHTS.values()) == pytest.approx(1.0)
