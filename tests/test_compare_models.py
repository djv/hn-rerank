from __future__ import annotations

import argparse

import pytest

from scripts.compare_models import (
    ModelSpec,
    normalize_seeds,
    parse_args,
    parse_model_arg,
    summarize_seed_metrics,
)


def make_model_dir(tmp_path, name: str):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")
    return model_dir


def test_parse_model_arg_accepts_label_and_model_dir(tmp_path) -> None:
    model_dir = make_model_dir(tmp_path, "prod")

    spec = parse_model_arg(f"prod={model_dir}")

    assert spec == ModelSpec(label="prod", path=model_dir.resolve())


def test_parse_model_arg_rejects_missing_model_file(tmp_path) -> None:
    model_dir = tmp_path / "broken"
    model_dir.mkdir()

    with pytest.raises(argparse.ArgumentTypeError):
        parse_model_arg(f"broken={model_dir}")


def test_parse_args_supports_repeated_models_and_flags(tmp_path) -> None:
    prod_dir = make_model_dir(tmp_path, "prod")
    base_dir = make_model_dir(tmp_path, "base")

    args = parse_args(
        [
            "alice",
            "--model",
            f"prod={prod_dir}",
            "--model",
            f"base={base_dir}",
            "--no-classifier",
            "--no-recency",
            "--seed",
            "2",
            "--seed",
            "2",
            "--seed",
            "5",
            "--cv-folds",
            "7",
            "--candidates",
            "300",
            "--cache-only",
        ]
    )

    assert args.username == "alice"
    assert [spec.label for spec in args.models] == ["prod", "base"]
    assert args.classifier is False
    assert args.recency is False
    assert args.seeds == [2, 5]
    assert args.cv_folds == 7
    assert args.candidates == 300
    assert args.cache_only is True


def test_parse_args_defaults_recency_to_project_setting(tmp_path) -> None:
    prod_dir = make_model_dir(tmp_path, "prod")
    base_dir = make_model_dir(tmp_path, "base")

    args = parse_args(
        [
            "alice",
            "--model",
            f"prod={prod_dir}",
            "--model",
            f"base={base_dir}",
        ]
    )

    assert args.recency is False


def test_normalize_seeds_defaults_and_dedupes() -> None:
    assert normalize_seeds(None) == [0]
    assert normalize_seeds([3, 3, 1, 3, 1]) == [3, 1]


def test_summarize_seed_metrics_adds_seed_std() -> None:
    summary = summarize_seed_metrics(
        [
            {"mrr": 0.1, "mrr_std": 0.01, "ndcg@10": 0.2},
            {"mrr": 0.3, "mrr_std": 0.05, "ndcg@10": 0.4},
        ]
    )

    assert summary["mrr"] == pytest.approx(0.2)
    assert summary["mrr_std"] == pytest.approx(0.03)
    assert summary["mrr_seed_std"] == pytest.approx(0.1)
    assert summary["ndcg@10"] == pytest.approx(0.3)
    assert summary["ndcg@10_seed_std"] == pytest.approx(0.1)
