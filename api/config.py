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
    negative_weight: float = 0.5529047831
    diversity_lambda: float = 0.2396634418
    diversity_lambda_classifier: float = 0.30
    max_results: int = 500
    non_semantic_weight: float = 0.05  # Total share for non-semantic signals
    comment_ratio: float = 0.0  # Share of non-semantic score assigned to comments

@dataclass(frozen=True)
class AdaptiveHNConfig:
    """Age-based HN weighting parameters."""
    weight_min: float = 0.3963567514
    weight_max: float = 0.4454707212
    threshold_young: float = 69.1427531319
    threshold_old: float = 720.0
    score_normalization_cap: float = 212.4038210119

@dataclass(frozen=True)
class FreshnessConfig:
    """Freshness decay parameters."""
    enabled: bool = True
    half_life_hours: float = 168.0
    max_boost: float = 0.1

@dataclass(frozen=True)
class SemanticConfig:
    """Semantic scoring and k-NN parameters."""
    maxsim_weight: float = 1.0
    meansim_weight: float = 0.0
    sigmoid_k: float = 31.2249293861
    sigmoid_threshold: float = 0.4749411784
    knn_neighbors: int = 6
    match_threshold: float = 0.85

@dataclass(frozen=True)
class ClassifierConfig:
    """Classifier training and feature parameters."""
    scoring_mode: str = "pairwise_logistic"
    feature_mode: str = "bottleneck"
    pairwise_negatives: int = 15
    pairwise_c: float = 1.4700450168
    k_feat: int = 7
    use_balanced_class_weight: bool = False
    cv_scoring: str = "f1"
    use_local_hidden_penalty: bool = False
    local_hidden_penalty_weight: float = 0.0
    local_hidden_penalty_k: int = 3
    use_centroid_feature: bool = True
    use_pos_knn_feature: bool = True
    use_neg_knn_feature: bool = True
    use_log_points_feature: bool = False

@dataclass(frozen=True)
class ClusteringConfig:
    """Multi-interest clustering parameters."""
    algorithm: str = "agglomerative"
    linkage: str = "ward"
    metric: str = "euclidean"
    distance_threshold: float = 1.3282321556
    similarity_threshold: float = 0.93
    outlier_similarity_threshold: float = 0.0
    min_samples_per_cluster: int = 1
    max_cluster_fraction: float = 0.25
    max_cluster_size: int = 40
    refine_iters: int = 2
    default_count: int = 30
    min_clusters: int = 2
    max_clusters: int = 40
    spectral_neighbors: int = 15

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
class CrossEncoderConfig:
    """Cross-encoder reranking parameters."""
    enabled: bool = True
    top_n: int = 50
    model_dir: str = "onnx_ce_model"
    weight: float = 0.8

@dataclass(frozen=True)
class ArchiveConfig:
    """Historical HN archive fetching parameters."""
    open_index_enabled: bool = False
    use_cached_stories: bool = True
    open_index_candidate_limit: int = 50
    bigquery_enabled: bool | None = None
    bigquery_candidate_limit: int | None = None

    def __post_init__(self) -> None:
        # Backward-compatible aliases for existing TOML/CLI usage.
        if self.bigquery_enabled is not None and self.bigquery_enabled:
            object.__setattr__(self, "open_index_enabled", True)
        if self.bigquery_candidate_limit is not None:
            object.__setattr__(
                self, "open_index_candidate_limit", self.bigquery_candidate_limit
            )

@dataclass(frozen=True)
class LearnedRankerConfig:
    """Shadow/active learned final ranking parameters."""
    shadow_enabled: bool = False
    active_enabled: bool = False
    model_path: Path = Path(".cache/learned_ranker/final_ranker.joblib")
    min_positive_labels: int = 10
    min_negative_labels: int = 10
    training_sources: str = "dashboard_feedback"
    source_feature_weight: float = 0.0
    balance_training_labels: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))

@dataclass(frozen=True)
class AppConfig:
    """Root configuration object."""
    username: str = field(default_factory=lambda: os.getlogin() if os.name != "nt" else "user")
    output_path: Path = Path("public/index.html")
    days: int = 30
    count: int = 40
    candidates: int = 2000
    signals: int = 2000
    use_classifier: bool = True
    contrastive: bool = False
    no_rss: bool = False
    no_tldr: bool = False
    no_naming: bool = False
    debug_scores: bool = False
    debug_scores_path: Path | None = None
    debug_clusters: bool = False
    debug_clusters_path: Path | None = None
    
    ranking: RankingConfig = field(default_factory=RankingConfig)
    adaptive_hn: AdaptiveHNConfig = field(default_factory=AdaptiveHNConfig)
    freshness: FreshnessConfig = field(default_factory=FreshnessConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    learned_ranker: LearnedRankerConfig = field(default_factory=LearnedRankerConfig)

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

        # Initialize sub-configs
        ranking = RankingConfig(**_get_section("ranking"))
        adaptive_hn = AdaptiveHNConfig(**_get_section("adaptive_hn"))
        freshness = FreshnessConfig(**_get_section("freshness"))
        semantic = SemanticConfig(**_get_section("semantic"))
        classifier = ClassifierConfig(**_get_section("classifier"))
        clustering = ClusteringConfig(**_get_section("clustering"))
        llm = LLMConfig(**_get_section("llm"))
        cross_encoder = CrossEncoderConfig(**_get_section("cross_encoder"))
        archive = ArchiveConfig(**_get_section("archive"))
        learned_ranker = LearnedRankerConfig(**_get_section("learned_ranker"))

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
            signals=int(_get_root("signals", 2000)),
            use_classifier=bool(_get_root("use_classifier", True)),
            contrastive=bool(_get_root("contrastive", False)),
            no_rss=bool(_get_root("no_rss", False)),
            no_tldr=bool(_get_root("no_tldr", False)),
            no_naming=bool(_get_root("no_naming", False)),
            debug_scores=bool(_get_root("debug_scores", False)),
            debug_clusters=bool(_get_root("debug_clusters", False)),
            ranking=ranking,
            adaptive_hn=adaptive_hn,
            freshness=freshness,
            semantic=semantic,
            classifier=classifier,
            clustering=clustering,
            llm=llm,
            cross_encoder=cross_encoder,
            archive=archive,
            learned_ranker=learned_ranker,
        )
