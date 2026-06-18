import json
from pathlib import Path

REPORT = Path(__file__).parent.parent / "eval_report.json"


def test_report_exists():
    assert REPORT.exists(), "Run `uv run python eval.py` first."


def test_report_has_4_formulas():
    r = json.loads(REPORT.read_text())
    assert r["formulas"].keys() == {"current", "diff", "up_only", "hn_baseline"}


def test_report_has_5_folds():
    r = json.loads(REPORT.read_text())
    for formula in r["formulas"].values():
        assert len(formula["per_fold"]) == 5


def test_svm_better_than_random():
    r = json.loads(REPORT.read_text())
    ndcg = r["formulas"]["up_only"]["mean"]["ndcg_at_10"]
    assert ndcg > 0.15, f"Best SVM NDCG@10 ({ndcg:.3f}) <= random (~0.15)"
