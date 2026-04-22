from __future__ import annotations

from api.model_metadata import (
    CURRENT_PRODUCTION_SPEC,
    E5_BASE_V2_SPEC,
    EmbeddingModelSpec,
    load_model_spec,
    write_model_spec,
)


def test_load_model_spec_defaults_to_current_production(tmp_path) -> None:
    spec = load_model_spec(tmp_path)

    assert spec == CURRENT_PRODUCTION_SPEC


def test_write_and_load_model_spec_round_trip(tmp_path) -> None:
    spec = EmbeddingModelSpec(
        model_id="test/model",
        pooling="cls",
        normalize=True,
        text_mode="plain",
    )

    write_model_spec(tmp_path, spec)

    assert load_model_spec(tmp_path) == spec


def test_prepare_text_respects_query_prefix_all_mode() -> None:
    assert E5_BASE_V2_SPEC.prepare_text("hello", is_query=True) == "query: hello"
    assert E5_BASE_V2_SPEC.prepare_text("hello", is_query=False) == "query: hello"
