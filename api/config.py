"""Typed configuration for HN Rerank."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class RankingConfig:
    """Weights and limits for the ranking engine."""

    max_results: int = 500
    external_min_model_score: float = 0.52


@dataclass(frozen=True)
class SemanticConfig:
    """Semantic scoring and k-NN parameters."""

    knn_neighbors: int = 6
    match_threshold: float = 0.85


@dataclass(frozen=True)
class ClassifierConfig:
    """Classifier training and feature parameters."""

    k_feat: int = 7
    raw_embedding_features: bool = True
    features: tuple[str, ...] = (
        "centroid",
        "pos_knn",
        "neg_knn",
        "closest_margin",
        "title_len",
        "text_len",
        "has_url",
        "is_github",
        "is_pdf",
        "comments_count",
        "is_hn",
        "source_trust",
    )
    # Minimum labeled examples on each side required to activate the model path.
    min_positive_examples: int = 5
    min_negative_examples: int = 5


@dataclass(frozen=True)
class ClusteringConfig:
    """Multi-interest clustering parameters."""

    similarity_threshold: float = 0.55
    outlier_similarity_threshold: float = 0.0
    min_samples_per_cluster: int = 1
    max_cluster_fraction: float = 0.25
    max_cluster_size: int = 40
    refine_iters: int = 2
    min_clusters: int = 2
    max_clusters: int = 40


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider and model settings."""

    provider: str = "mistral"
    cluster_name_model_primary: str = "llama-3.3-70b-versatile"
    cluster_name_model_fallback: str = "llama-3.1-8b-instant"
    mistral_model: str = "mistral-small-latest"
    tldr_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    tldr_batch_size: int = 3
    max_total_seconds: float = 600.0


@dataclass(frozen=True)
class ExploreConfig:
    """Explore/exploit dashboard acquisition settings."""

    enabled: bool = False
    slots: int = 4
    method: str = "entropy"
    min_quality: float = 0.4
    top_reserve: int = 5


@dataclass(frozen=True)
class ArchiveConfig:
    """Historical HN archive fetching parameters."""

    open_index_enabled: bool = False
    use_cached_stories: bool = True
    open_index_candidate_limit: int = 50


@dataclass(frozen=True)
class SingleModelConfig:
    """Feedback-trained runtime ranking model parameters."""

    min_positive_labels: int = 10
    min_negative_labels: int = 10
    balance_training_labels: bool = True
    model_type: str = (
        "svm"  # Options: logistic, random_forest, gradient_boosting, svm, mlp
    )
    svm_kernel: str = "rbf"  # Options: linear, poly, rbf, sigmoid
    svm_c: float = 3.0
    svm_gamma: str | float = "scale"  # "scale", "auto", or a numeric gamma value
    rf_n_estimators: int = 200
    rf_max_depth: int = 0  # 0 means None (unlimited)
    rf_min_samples_leaf: int = 2
    rf_min_samples_split: int = 2
    rf_max_features: str = "sqrt"  # "sqrt", "log2", or a float fraction


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object."""

    username: str = field(
        default_factory=lambda: os.getlogin() if os.name != "nt" else "user"
    )
    output_path: Path = Path("public/index.html")
    days: int = 30
    count: int = 40
    candidates: int = 2000
    no_rss: bool = False
    no_tldr: bool = False
    no_naming: bool = False
    debug_scores: bool = True
    debug_clusters: bool = False

    ranking: RankingConfig = field(default_factory=RankingConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    single_model: SingleModelConfig = field(default_factory=SingleModelConfig)

    @classmethod
    def load(cls, toml_path: Path | str | None = None, **overrides: Any) -> Self:
        """Load from TOML and apply overrides."""
        data: dict[str, Any] = {}
        config_path = Path(toml_path) if toml_path else Path("hn_rerank.toml")

        if config_path.exists():
            with open(config_path, "rb") as f:
                raw = tomllib.load(f)
                data = raw.get("hn_rerank", {})

        # Helper to extract sections
        def _get_section(name: str) -> dict[str, Any]:
            return data.get(name, {})

        # Strip unknown keys from each section so old TOML files with removed
        # fields (e.g. adaptive_hn, freshness) don't cause TypeError on load.
        def _safe_section(name: str, cls_: type) -> dict[str, Any]:
            import dataclasses

            known = {f.name for f in dataclasses.fields(cls_)}
            return {k: v for k, v in _get_section(name).items() if k in known}

        # Initialize sub-configs
        ranking = RankingConfig(**_safe_section("ranking", RankingConfig))
        semantic = SemanticConfig(**_safe_section("semantic", SemanticConfig))
        classifier = ClassifierConfig(**_safe_section("classifier", ClassifierConfig))
        clustering = ClusteringConfig(**_safe_section("clustering", ClusteringConfig))
        llm = LLMConfig(**_safe_section("llm", LLMConfig))
        archive = ArchiveConfig(**_safe_section("archive", ArchiveConfig))
        explore = ExploreConfig(**_safe_section("explore", ExploreConfig))
        single_model = SingleModelConfig(
            **_safe_section("single_model", SingleModelConfig)
        )

        def _get_root(key: str, default: Any) -> Any:
            val = overrides.get(key)
            if val is not None:
                return val
            return data.get(key, default)

        # Root level fields (mapping TOML names to dataclass names if they differ)
        output_val = str(_get_root("output", "public/index.html"))

        return cls(
            username=str(_get_root("username", "user")),
            output_path=Path(output_val),
            days=int(_get_root("days", 30)),
            count=int(_get_root("count", 40)),
            candidates=int(_get_root("candidates", 2000)),
            no_rss=bool(_get_root("no_rss", False)),
            no_tldr=bool(_get_root("no_tldr", False)),
            no_naming=bool(_get_root("no_naming", False)),
            debug_scores=bool(_get_root("debug_scores", True)),
            debug_clusters=bool(_get_root("debug_clusters", False)),
            ranking=ranking,
            semantic=semantic,
            classifier=classifier,
            clustering=clustering,
            llm=llm,
            archive=archive,
            explore=explore,
            single_model=single_model,
        )
