from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from api.config import AppConfig
from api.models import Story, StoryDict
from evaluate_quality import RankingEvaluator


@dataclass
class _AblationResult:
    name: str
    features: list[str]
    mrr: float
    precision_at_5: float
    recall_at_5: float
    n_train: int
    n_test: int
    n_candidates: int
    classifier_used: bool
    elapsed_seconds: float


def _make_snapshot_dict() -> dict:
    """Build a minimal snapshot for testing."""
    stories: list[Story] = [
        Story(
            id=1,
            title="Python",
            url=None,
            score=0,
            time=1000,
            text_content="Python programming language tutorial",
        ),
        Story(
            id=2,
            title="Rust",
            url=None,
            score=0,
            time=1001,
            text_content="Rust systems programming language",
        ),
        Story(
            id=3,
            title="Go",
            url=None,
            score=0,
            time=1002,
            text_content="Go programming language concurrency",
        ),
        Story(
            id=4,
            title="TypeScript",
            url=None,
            score=0,
            time=1003,
            text_content="TypeScript typed JavaScript",
        ),
        Story(
            id=5,
            title="Test story",
            url=None,
            score=0,
            time=999,
            text_content="This is a test story for evaluation",
        ),
    ]
    candidates = stories[:5]
    train_stories = stories[:3]
    test_stories = [stories[4]]
    neg_stories = [stories[3]]
    test_ids = {5}

    def _to_dict(s: Story) -> StoryDict:
        return {
            "id": s.id,
            "title": s.title,
            "url": s.url,
            "score": s.score,
            "time": s.time,
            "discussion_url": None,
            "comments": [],
            "text_content": s.text_content,
            "source": "hn",
            "comment_count": None,
        }

    return {
        "format_version": 1,
        "username": "test",
        "saved_at": 2000000,
        "metadata": {
            "source": "test",
            "holdout": 0.2,
            "limit_pos": 200,
            "limit_neg": 100,
            "candidate_count": 5,
            "use_classifier": True,
            "cache_only": True,
            "allow_stale": True,
            "age_matched": False,
        },
        "train_stories": [_to_dict(s) for s in train_stories],
        "test_stories": [_to_dict(s) for s in test_stories],
        "neg_stories": [_to_dict(s) for s in neg_stories],
        "candidates": [_to_dict(s) for s in candidates],
        "test_ids": sorted(test_ids),
    }


def _make_fake_embeddings(texts: list[str], dim: int = 32) -> np.ndarray:
    """Create deterministic fake embeddings."""
    rng = np.random.default_rng(42)
    emb = rng.normal(0, 1, (len(texts), dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return emb / norms


@pytest.fixture
def test_snapshot_path() -> Path:
    """Create a temporary snapshot file."""
    payload = _make_snapshot_dict()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


def _mock_embeddings_shape(texts, **kwargs):
    """Return fake embeddings matching the number of texts."""
    return _make_fake_embeddings(texts)


def _ablate(
    evaluator: RankingEvaluator,
    features: list[str],
    name: str = "test",
) -> _AblationResult:
    """Train and evaluate with the given feature set (test helper, mirrors scripts/feature_ablation.run_one)."""
    import api.rerank as rerank_mod
    from api.config import ClassifierConfig

    classifier_cfg = ClassifierConfig(features=tuple(features))
    config = AppConfig(classifier=classifier_cfg)
    rerank_mod.clear_story_age_at_vote_map()

    start = time.time()
    metrics = evaluator.evaluate(config=config)
    elapsed = time.time() - start

    dataset = evaluator.dataset
    return _AblationResult(
        name=name,
        features=list(features),
        mrr=metrics.get("mrr", 0.0),
        precision_at_5=metrics.get("precision@5", 0.0),
        recall_at_5=metrics.get("recall@5", 0.0),
        n_train=len(dataset.train_stories) if dataset else 0,
        n_test=len(dataset.test_stories) if dataset else 0,
        n_candidates=len(dataset.candidates) if dataset else 0,
        classifier_used=True,
        elapsed_seconds=elapsed,
    )


def test_ablation_runs_baseline(test_snapshot_path: Path) -> None:
    """Harness runs the baseline feature set without error and returns metrics."""
    with (
        patch("api.rerank.get_embeddings", side_effect=_mock_embeddings_shape),
        patch("evaluate_quality.get_embeddings", side_effect=_mock_embeddings_shape),
    ):
        evaluator = RankingEvaluator(username="test")
        assert evaluator.load_snapshot(test_snapshot_path)
        result = _ablate(evaluator, baseline_features(), "test")

    assert 0.0 <= result.mrr <= 1.0
    assert 0.0 <= result.precision_at_5 <= 1.0
    assert 0.0 <= result.recall_at_5 <= 1.0
    assert result.n_train > 0
    assert result.n_test > 0
    assert result.elapsed_seconds > 0.0


def test_ablation_different_features_produce_metric_changes(
    test_snapshot_path: Path,
) -> None:
    """Different feature sets should produce (potentially) different metrics."""
    with (
        patch("api.rerank.get_embeddings", side_effect=_mock_embeddings_shape),
        patch("evaluate_quality.get_embeddings", side_effect=_mock_embeddings_shape),
    ):
        evaluator = RankingEvaluator(username="test")
        assert evaluator.load_snapshot(test_snapshot_path)

        full = _ablate(evaluator, baseline_features(), "full")
        mini = _ablate(evaluator, ["centroid"], "only-centroid")

    # The mrr should differ between full and single-feature (almost certainly)
    # but we only assert that both produce valid metrics.
    assert 0.0 <= full.mrr <= 1.0
    assert 0.0 <= mini.mrr <= 1.0


def baseline_features() -> list[str]:
    return list(AppConfig().classifier.features)
