from __future__ import annotations

from dataclasses import replace

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


def test_apply_evaluator_overrides_can_enable_new_metadata_features() -> None:
    config = AppConfig()

    updated = apply_evaluator_overrides(
        config,
        use_new_features=True,
    )

    assert updated.classifier.use_title_len_feature is True
    assert updated.classifier.use_text_len_feature is True
    assert updated.classifier.use_has_url_feature is True
    assert updated.classifier.use_github_feature is True
    assert updated.classifier.use_pdf_feature is True
    assert updated.classifier.use_comments_count_feature is True


def test_single_model_config_can_override_svm_kernel() -> None:
    config = AppConfig()

    updated = replace(
        config,
        single_model=replace(config.single_model, svm_kernel="linear"),
    )

    assert updated.single_model.model_type == "svm"
    assert updated.single_model.svm_kernel == "linear"


def test_single_model_config_can_override_svm_c_and_gamma() -> None:
    config = AppConfig()

    updated = replace(
        config,
        single_model=replace(
            config.single_model,
            svm_c=2.5,
            svm_gamma=0.125,
        ),
    )

    assert updated.single_model.svm_c == 2.5
    assert updated.single_model.svm_gamma == 0.125
