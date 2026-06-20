import json
from pathlib import Path

REPORT = Path(__file__).parent.parent / "eval_report.json"


def test_report_exists():
    assert REPORT.exists(), "Run `uv run python eval.py` first."


def test_report_has_expected_formulas():
    r = json.loads(REPORT.read_text())
    expected = {"current", "up_only", "hn_baseline"}
    assert expected.issubset(r["formulas"].keys())


def test_report_has_5_folds():
    r = json.loads(REPORT.read_text())
    for formula in r["formulas"].values():
        assert len(formula["per_fold"]) == 5


def test_svm_better_than_random():
    r = json.loads(REPORT.read_text())
    up_only = r["formulas"]["up_only"]["mean"]["mmr"]["ndcg_at_100"]
    hn = r["formulas"]["hn_baseline"]["mean"]["mmr"]["ndcg_at_100"]
    assert up_only > hn, f"SVM NDCG@100 ({up_only:.3f}) <= HN baseline ({hn:.3f})"


def test_report_has_map_and_brier():
    r = json.loads(REPORT.read_text())
    for formula_data in r["formulas"].values():
        for variant in ("mmr", "raw"):
            metrics = formula_data["mean"][variant]
            assert "map" in metrics, f"map missing from {variant}"
            assert "brier_up" in metrics, f"brier_up missing from {variant}"


def test_map_in_valid_range():
    r = json.loads(REPORT.read_text())
    for formula_data in r["formulas"].values():
        for variant in ("mmr", "raw"):
            map_val = formula_data["mean"][variant]["map"]
            assert 0.0 <= map_val <= 1.0, f"map {map_val} out of [0,1]"


def test_svm_map_better_than_hn_baseline():
    r = json.loads(REPORT.read_text())
    svm_map = r["formulas"]["up_only"]["mean"]["mmr"]["map"]
    hn_map = r["formulas"]["hn_baseline"]["mean"]["mmr"]["map"]
    assert svm_map > hn_map, (
        f"SVM MAP ({svm_map:.3f}) <= HN baseline MAP ({hn_map:.3f})"
    )
