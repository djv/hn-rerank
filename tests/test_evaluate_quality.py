from __future__ import annotations

from api.config import AppConfig
from evaluate_quality import apply_evaluator_overrides, build_first_stage_ablation_config


def test_apply_evaluator_overrides_can_strip_metadata_features_for_pure_semantic() -> None:
    config = AppConfig()

    updated = apply_evaluator_overrides(
        config,
        pure_semantic=True,
    )

    assert updated.classifier.use_log_points_feature is False
    assert updated.classifier.use_log_comments_feature is False
    assert updated.classifier.use_comment_ratio_feature is False


def test_build_first_stage_ablation_config_preserves_other_settings() -> None:
    config = AppConfig(count=55, candidates=321)

    updated = build_first_stage_ablation_config(config)

    assert updated.count == 55
    assert updated.candidates == 321
    assert updated.classifier.use_log_points_feature is False
    assert updated.classifier.use_log_comments_feature is False
    assert updated.classifier.use_comment_ratio_feature is False
