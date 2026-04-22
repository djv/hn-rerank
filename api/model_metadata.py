from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MODEL_METADATA_FILENAME = "hn_embedding_model.json"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

type PoolingMode = Literal["mean", "cls"]
type TextMode = Literal["plain", "query_prefix_all", "asymmetric_query_doc"]


@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_id: str
    pooling: PoolingMode
    normalize: bool
    text_mode: TextMode
    query_prefix: str = ""
    document_prefix: str = ""
    max_tokens: int = 512
    trust_remote_code: bool = False

    @property
    def cache_key(self) -> str:
        parts = [
            self.model_id,
            self.pooling,
            "norm" if self.normalize else "raw",
            self.text_mode,
            self.query_prefix,
            self.document_prefix,
            str(self.max_tokens),
        ]
        return "|".join(parts)

    def prepare_text(self, text: str, *, is_query: bool) -> str:
        if self.text_mode == "plain":
            prefix = self.query_prefix if is_query else self.document_prefix
        elif self.text_mode == "query_prefix_all":
            prefix = self.query_prefix
        else:
            prefix = self.query_prefix if is_query else self.document_prefix
        return f"{prefix}{text}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


CURRENT_PRODUCTION_SPEC = EmbeddingModelSpec(
    model_id="bge-base-en-v1.5",
    pooling="mean",
    normalize=True,
    text_mode="asymmetric_query_doc",
    query_prefix=DEFAULT_QUERY_PREFIX,
)

BGE_BASE_OFFICIAL_SPEC = EmbeddingModelSpec(
    model_id="BAAI/bge-base-en-v1.5",
    pooling="cls",
    normalize=True,
    text_mode="plain",
)

GTE_BASE_V15_SPEC = EmbeddingModelSpec(
    model_id="Alibaba-NLP/gte-base-en-v1.5",
    pooling="cls",
    normalize=True,
    text_mode="plain",
    trust_remote_code=True,
)

E5_BASE_V2_SPEC = EmbeddingModelSpec(
    model_id="intfloat/e5-base-v2",
    pooling="mean",
    normalize=True,
    text_mode="query_prefix_all",
    query_prefix="query: ",
)

BIENCODER_BAKEOFF_SPECS: dict[str, EmbeddingModelSpec] = {
    "bge_base_official": BGE_BASE_OFFICIAL_SPEC,
    "gte_base_v15": GTE_BASE_V15_SPEC,
    "e5_base_v2": E5_BASE_V2_SPEC,
}


def load_model_spec(model_dir: str | Path) -> EmbeddingModelSpec:
    metadata_path = Path(model_dir) / MODEL_METADATA_FILENAME
    if not metadata_path.is_file():
        return CURRENT_PRODUCTION_SPEC

    raw = json.loads(metadata_path.read_text())
    return EmbeddingModelSpec(
        model_id=str(raw["model_id"]),
        pooling=raw["pooling"],
        normalize=bool(raw.get("normalize", True)),
        text_mode=raw["text_mode"],
        query_prefix=str(raw.get("query_prefix", "")),
        document_prefix=str(raw.get("document_prefix", "")),
        max_tokens=int(raw.get("max_tokens", 512)),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
    )


def write_model_spec(model_dir: str | Path, spec: EmbeddingModelSpec) -> Path:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path / MODEL_METADATA_FILENAME
    metadata_path.write_text(spec.to_json() + "\n")
    return metadata_path
