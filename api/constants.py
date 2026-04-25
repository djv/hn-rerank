"""
Constants and configuration values for HN reranking.
Loads overrides from hn_rerank.toml if present.
"""
import sys
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

def _load_config() -> dict[str, Any]:
    """Load configuration from hn_rerank.toml."""
    config_path = Path("hn_rerank.toml")
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("hn_rerank", {})
    except Exception as e:
        print(f"Warning: Failed to load config: {e}", file=sys.stderr)
        return {}

_CONFIG = _load_config()

def _get(section: str, key: str, default: Any) -> Any:
    """Get config value with fallback to default."""
    return _CONFIG.get(section, {}).get(key, default)


# Content Limits
ARTICLE_SNIPPET_LENGTH = 1000
TEXT_CONTENT_MAX_TOKENS = 512

# Cache Paths
EMBEDDING_CACHE_DIR = ".cache/embeddings"
CLUSTER_EMBEDDING_CACHE_DIR = ".cache/embeddings_cluster"
EMBEDDING_CACHE_MAX_FILES = 20000
STORY_CACHE_DIR = ".cache/stories"
STORY_CACHE_VERSION = "v2"
USER_CACHE_DIR = ".cache/user"
STORY_CACHE_TTL = 86400
USER_CACHE_TTL = 120  # 2 minutes
CANDIDATE_CACHE_DIR = ".cache/candidates"
CANDIDATE_CACHE_VERSION = "v2"
CANDIDATE_CACHE_TTL_SHORT = 1800  # 30 minutes
CANDIDATE_CACHE_TTL_LONG = 604800  # 1 week
CANDIDATE_CACHE_TTL_ARCHIVE = 7776000  # 90 days
STORY_CACHE_MAX_FILES = 25000  # LRU eviction threshold
RSS_CACHE_DIR = ".cache/rss"
RSS_OPML_CACHE_TTL = 86400  # 1 day
RSS_FEED_CACHE_TTL = 3600  # 1 hour
RSS_FEED_CACHE_VERSION = 2
RSS_ARTICLE_CACHE_TTL = 86400  # 1 day
RSS_CACHE_MAX_FILES = 5000

# Concurrency
EXTERNAL_REQUEST_SEMAPHORE = 10  # Reduced to avoid API throttling

# User Limits
MAX_USER_STORIES = 2000

# Discovery Pool
ALGOLIA_MIN_POINTS = 5
ALGOLIA_DEFAULT_DAYS = 30
CANDIDATE_FETCH_COUNT = 1000
RSS_OPML_URL = (
    "https://gist.githubusercontent.com/emschwartz/e6d2bf860ccc367fe37ff953ba6de66b/raw/hn-popular-blogs-2025.opml"
)
RSS_EXTRA_FEEDS = [
    "https://jack-clark.net/feed/",
    "https://lobste.rs/rss",
    "https://tildes.net/topics.rss",
]  # Additional feeds not in OPML
RSS_MAX_FEEDS = 0  # 0 = no max feed limit
RSS_PER_FEED_LIMIT = 5
RSS_CURATED_NEWS_PER_FEED_LIMIT = 50
RSS_ALLOWED_SOURCE_LANGUAGES = ("en", "fr", "es")

# Inference
DEFAULT_EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MIN_CLIP = 1e-9
# Current live ONNX artifact timestamped 2026-01-31. Exact checkpoint label is
# not confirmed, so keep a neutral version id for cache invalidation/provenance.
EMBEDDING_MODEL_VERSION = "prod-e5-2026-04-22"
CLUSTER_EMBEDDING_MODEL_VERSION = "prod-e5-2026-04-24"
CLUSTER_EMBEDDING_MODEL_DIR = "onnx_model"

# Similarity Bounds
SIMILARITY_MIN = -1.0
SEMANTIC_MATCH_THRESHOLD = 0.85
KNN_NEIGHBORS = _get("semantic", "knn_neighbors", 1)

# Multi-Interest Clustering
DEFAULT_CLUSTER_COUNT = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 40
MIN_SAMPLES_PER_CLUSTER = _get("clustering", "min_samples_per_cluster", 1)
MAX_CLUSTER_FRACTION = _get("clustering", "max_cluster_fraction", 0.25)
MAX_CLUSTER_SIZE = _get("clustering", "max_cluster_size", 40)
CLUSTER_REFINE_ITERS = _get("clustering", "refine_iters", 2)
CLUSTER_SIMILARITY_THRESHOLD = _get("clustering", "similarity_threshold", 0.93)
CLUSTER_OUTLIER_SIMILARITY_THRESHOLD = _get("clustering", "outlier_similarity_threshold", 0.0)

# Ranking Weights
RANKING_HN_WEIGHT = 0.05
RANKING_NEGATIVE_WEIGHT = _get("ranking", "negative_weight", 0.5529047831)
RANKING_DIVERSITY_LAMBDA = _get("ranking", "diversity_lambda", 0.2396634418)
RANKING_MAX_RESULTS = _get("ranking", "max_results", 500)

# Adaptive HN Weight (age-based)
ADAPTIVE_HN_WEIGHT_MIN = _get("adaptive_hn", "weight_min", 0.0449033752)
ADAPTIVE_HN_WEIGHT_MAX = _get("adaptive_hn", "weight_max", 0.0585400016)
ADAPTIVE_HN_THRESHOLD_YOUNG = _get("adaptive_hn", "threshold_young", 5.8379569842)
ADAPTIVE_HN_THRESHOLD_OLD = _get("adaptive_hn", "threshold_old", 50.1201376639)

# Freshness Decay
FRESHNESS_ENABLED = _get("freshness", "enabled", True)
FRESHNESS_HALF_LIFE_HOURS = _get("freshness", "half_life_hours", 66.0122091339)
FRESHNESS_MAX_BOOST = _get("freshness", "max_boost", 0.0411369570)

# Positive-signal Recency Weighting
POSITIVE_RECENCY_ENABLED = _get("recency", "enabled", True)
POSITIVE_RECENCY_HALF_LIFE_DAYS = _get("recency", "half_life_days", 90.0)

# Semantic Scoring
SEMANTIC_MAXSIM_WEIGHT = _get("semantic", "maxsim_weight", 0.95)
SEMANTIC_MEANSIM_WEIGHT = _get("semantic", "meansim_weight", 0.05)
SEMANTIC_SIGMOID_K = _get("semantic", "sigmoid_k", 31.2249293861)
SEMANTIC_SIGMOID_THRESHOLD = _get("semantic", "sigmoid_threshold", 0.4749411784)
KNN_SIGMOID_K = _get("semantic", "knn_sigmoid_k", 6.3521201201)
KNN_MAXSIM_WEIGHT = _get("semantic", "knn_maxsim_weight", 0.2635706275)
HN_SCORE_NORMALIZATION_CAP = _get("adaptive_hn", "score_normalization_cap", 1392.4125765115)

# Classifier Tuning
CLASSIFIER_K_FEAT = _get("classifier", "k_feat", 5)
CLASSIFIER_NEG_SAMPLE_WEIGHT = _get("classifier", "neg_sample_weight", 1.6984260758)
CLASSIFIER_USE_BALANCED_CLASS_WEIGHT = _get(
    "classifier", "use_balanced_class_weight", True
)
CLASSIFIER_CV_SCORING = _get("classifier", "cv_scoring", "f1")
CLASSIFIER_USE_LOCAL_HIDDEN_PENALTY = _get(
    "classifier", "use_local_hidden_penalty", False
)
CLASSIFIER_LOCAL_HIDDEN_PENALTY_WEIGHT = _get(
    "classifier", "local_hidden_penalty_weight", 0.0
)
CLASSIFIER_LOCAL_HIDDEN_PENALTY_K = _get(
    "classifier", "local_hidden_penalty_k", 3
)
CLASSIFIER_USE_CENTROID_FEATURE = _get("classifier", "use_centroid_feature", True)
CLASSIFIER_USE_POS_KNN_FEATURE = _get("classifier", "use_pos_knn_feature", True)
CLASSIFIER_USE_NEG_KNN_FEATURE = _get("classifier", "use_neg_knn_feature", True)

# Clustering
CLUSTER_ALGORITHM = _get("clustering", "algorithm", "agglomerative")
CLUSTER_AGGLOMERATIVE_LINKAGE = _get("clustering", "linkage", "ward")
CLUSTER_AGGLOMERATIVE_METRIC = _get("clustering", "metric", "euclidean")
CLUSTER_AGGLOMERATIVE_THRESHOLD = _get("clustering", "distance_threshold", 0.75)
CLUSTER_SPECTRAL_NEIGHBORS = _get("clustering", "spectral_neighbors", 15)

# Comment Pool
MIN_STORY_COMMENTS = 20  # Historical threshold kept for tests/tuning context
MIN_CANDIDATE_COMMENTS = 0  # 0 = do not prefilter candidates by comment count
MAX_COMMENTS_COLLECTED = 200  # Increased for richer signal
TOP_COMMENTS_FOR_RANKING = 12  # Aligned with 512-token limit (Title + ~10-12 comments)
TOP_COMMENTS_FOR_UI = 10
MIN_COMMENT_LENGTH = 30  # Filter short low-value comments (relaxed)

# LLM Configuration
LLM_PROVIDER = _get("llm", "provider", "groq")
LLM_CLUSTER_NAME_MODEL_PRIMARY = "llama-3.3-70b-versatile"
LLM_CLUSTER_NAME_MODEL_FALLBACK = "llama-3.1-8b-instant"
LLM_MISTRAL_MODEL = "mistral-small-latest"
LLM_CLUSTER_NAME_PROMPT_VERSION = "v13"
LLM_TLDR_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
LLM_TLDR_MAX_TOKENS = 2000
LLM_CLUSTER_MAX_TOKENS = 64
LLM_TLDR_BATCH_SIZE = 3  # Stories per TLDR request
LLM_CLUSTER_TITLE_MAX_CHARS = 140  # Max chars per title in naming payload
LLM_CLUSTER_NAME_MAX_WORDS = 6  # Max words in cluster name
LLM_CLUSTER_NAME_MIN_COVERAGE = 0.35  # Min title token overlap with label
LLM_CLUSTER_MAX_RETRIES = 4
LLM_CLUSTER_MAX_ROUNDS = 2
LLM_CLUSTER_MAX_TOTAL_SECONDS = 600.0  # Fail fast if naming stalls too long

# Rate Limiting (Token Bucket)
RATE_LIMIT_REFILL_RATE = 0.25  # Tokens per second (1 call per 4 seconds)
RATE_LIMIT_MAX_TOKENS = 1.0
RATE_LIMIT_JITTER_MAX = 0.5  # Max random jitter in seconds
RATE_LIMIT_429_BACKOFF_BASE = 2.0  # Base delay on 429 response
RATE_LIMIT_ERROR_BACKOFF_BASE = 1.0  # Base delay on other errors
RATE_LIMIT_ERROR_BACKOFF_MAX = 30.0
LLM_HTTP_CONNECT_TIMEOUT = 10.0
LLM_HTTP_READ_TIMEOUT = 30.0
LLM_HTTP_WRITE_TIMEOUT = 10.0
LLM_HTTP_POOL_TIMEOUT = 5.0
LLM_MIN_REQUEST_INTERVAL = 8.0  # Minimum seconds between Groq requests
LLM_429_COOLDOWN_BASE = 2.0  # Base cooldown when rate-limited
LLM_429_COOLDOWN_MAX = 60.0  # Max cooldown when rate-limited
LLM_HTTP_USER_AGENT = "hn-rerank/1.0"
