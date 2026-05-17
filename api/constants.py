"""
Constants and configuration values for HN reranking.
Loads overrides from hn_rerank.toml if present.
"""


from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


from api.config import AppConfig  # noqa: E402

_config = AppConfig.load()


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
RSS_FEED_CACHE_TTL = 86400  # 1 day
RSS_FEED_CACHE_VERSION = 3
RSS_ARTICLE_CACHE_TTL = 86400  # 1 day
RSS_CACHE_MAX_FILES = 5000

# Concurrency
EXTERNAL_REQUEST_SEMAPHORE = 10  # Reduced to avoid API throttling

# User Limits
MAX_USER_STORIES = 2000

# Discovery Pool
ALGOLIA_MIN_POINTS = 5
ALGOLIA_DEFAULT_DAYS = 30
CANDIDATE_FETCH_COUNT = 2000
RSS_OPML_URL = "https://gist.githubusercontent.com/emschwartz/e6d2bf860ccc367fe37ff953ba6de66b/raw/hn-popular-blogs-2025.opml"
RSS_EXTRA_FEEDS = [
    "https://jack-clark.net/feed/",
    "https://lobste.rs/top/rss",
    "https://tildes.net/topics.rss",
    "https://www.lesswrong.com/feed.xml?view=frontpage&karmaThreshold=20",
    "https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25",
    "https://www.reddit.com/r/programming/top/.rss?t=week&limit=25",
    "https://www.reddit.com/r/compsci/top/.rss?t=week&limit=25",
    "https://digg.com/ai",
]  # Additional feeds not in OPML
RSS_EXCLUDED_FEEDS = {
    "http://www.aaronsw.com/2002/feeds/pgessays.rss",
    "https://chadnauseam.com/rss.xml",
    "https://rachelbythebay.com/w/atom.xml",
    "https://www.tedunangst.com/flak/rss",
}
RSS_MAX_FEEDS = 0  # 0 = no max feed limit
RSS_PER_FEED_LIMIT = 70
RSS_CURATED_NEWS_PER_FEED_LIMIT = 50
RSS_ALLOWED_SOURCE_LANGUAGES = ("en", "fr", "es")

# Inference
DEFAULT_EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MIN_CLIP = 1e-9
# BGE-small CLS/query was selected by the local model bakeoff.
EMBEDDING_MODEL_VERSION = "bge-small-cls-query-all-2026-04-30"
CLUSTER_EMBEDDING_MODEL_VERSION = "bge-small-cls-query-all-2026-04-30"
CLUSTER_EMBEDDING_MODEL_DIR = "onnx_model"

# Similarity Bounds
SIMILARITY_MIN = -1.0
SEMANTIC_MATCH_THRESHOLD = 0.85
KNN_NEIGHBORS = _config.semantic.knn_neighbors

# Multi-Interest Clustering
DEFAULT_CLUSTER_COUNT = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 40
MIN_SAMPLES_PER_CLUSTER = _config.clustering.min_samples_per_cluster
MAX_CLUSTER_FRACTION = _config.clustering.max_cluster_fraction
MAX_CLUSTER_SIZE = _config.clustering.max_cluster_size
CLUSTER_REFINE_ITERS = _config.clustering.refine_iters
CLUSTER_SIMILARITY_THRESHOLD = _config.clustering.similarity_threshold
CLUSTER_OUTLIER_SIMILARITY_THRESHOLD = _config.clustering.outlier_similarity_threshold

# Ranking Weights
RANKING_NON_SEMANTIC_WEIGHT = _config.ranking.non_semantic_weight
RANKING_COMMENT_RATIO = _config.ranking.comment_ratio
RANKING_NEGATIVE_WEIGHT = _config.ranking.negative_weight
RANKING_DIVERSITY_LAMBDA = _config.ranking.diversity_lambda
RANKING_MAX_RESULTS = _config.ranking.max_results

# Adaptive HN Weight (age-based)
ADAPTIVE_HN_WEIGHT_MIN = _config.adaptive_hn.weight_min
ADAPTIVE_HN_WEIGHT_MAX = _config.adaptive_hn.weight_max
ADAPTIVE_HN_THRESHOLD_YOUNG = _config.adaptive_hn.threshold_young
ADAPTIVE_HN_THRESHOLD_OLD = _config.adaptive_hn.threshold_old

# Freshness Decay
FRESHNESS_ENABLED = _config.freshness.enabled
FRESHNESS_HALF_LIFE_HOURS = _config.freshness.half_life_hours
FRESHNESS_MAX_BOOST = _config.freshness.max_boost

# Semantic Scoring
SEMANTIC_MAXSIM_WEIGHT = _config.semantic.maxsim_weight
SEMANTIC_MEANSIM_WEIGHT = _config.semantic.meansim_weight
SEMANTIC_SIGMOID_K = _config.semantic.sigmoid_k
SEMANTIC_SIGMOID_THRESHOLD = _config.semantic.sigmoid_threshold
HN_SCORE_NORMALIZATION_CAP = _config.adaptive_hn.score_normalization_cap

# Classifier Tuning
CLASSIFIER_SCORING_MODE = _config.classifier.scoring_mode
CLASSIFIER_FEATURE_MODE = _config.classifier.feature_mode
CLASSIFIER_PAIRWISE_NEGATIVES = _config.classifier.pairwise_negatives
CLASSIFIER_PAIRWISE_C = _config.classifier.pairwise_c
CLASSIFIER_K_FEAT = _config.classifier.k_feat
CLASSIFIER_USE_BALANCED_CLASS_WEIGHT = _config.classifier.use_balanced_class_weight
CLASSIFIER_CV_SCORING = _config.classifier.cv_scoring
CLASSIFIER_USE_CENTROID_FEATURE = _config.classifier.use_centroid_feature
CLASSIFIER_USE_POS_KNN_FEATURE = _config.classifier.use_pos_knn_feature
CLASSIFIER_USE_NEG_KNN_FEATURE = _config.classifier.use_neg_knn_feature
CLASSIFIER_USE_LOG_POINTS_FEATURE = _config.classifier.use_log_points_feature

# Clustering
CLUSTER_ALGORITHM = _config.clustering.algorithm
CLUSTER_AGGLOMERATIVE_LINKAGE = _config.clustering.linkage
CLUSTER_AGGLOMERATIVE_METRIC = _config.clustering.metric
CLUSTER_AGGLOMERATIVE_THRESHOLD = _config.clustering.distance_threshold
CLUSTER_SPECTRAL_NEIGHBORS = _config.clustering.spectral_neighbors

# Comment Pool
MIN_STORY_COMMENTS = 20  # Historical threshold kept for tests/tuning context
MIN_CANDIDATE_COMMENTS = 0  # 0 = do not prefilter candidates by comment count
MAX_COMMENTS_COLLECTED = 200  # Increased for richer signal
TOP_COMMENTS_FOR_RANKING = 12  # Aligned with 512-token limit (Title + ~10-12 comments)
TOP_COMMENTS_FOR_UI = 10
MIN_COMMENT_LENGTH = 30  # Filter short low-value comments (relaxed)

# LLM Configuration
LLM_PROVIDER = _config.llm.provider
LLM_CLUSTER_NAME_MODEL_PRIMARY = "llama-3.3-70b-versatile"
LLM_CLUSTER_NAME_MODEL_FALLBACK = "llama-3.1-8b-instant"
LLM_MISTRAL_MODEL = "mistral-small-latest"
LLM_CLUSTER_NAME_PROMPT_VERSION = "v13"
LLM_TLDR_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
LLM_TLDR_MAX_TOKENS = 2000
LLM_CLUSTER_MAX_TOKENS = 150
LLM_TLDR_BATCH_SIZE = 3  # Stories per TLDR request
LLM_CLUSTER_TITLE_MAX_CHARS = 140  # Max chars per title in naming payload
LLM_CLUSTER_NAME_MAX_WORDS = 6  # Max words in cluster name
LLM_CLUSTER_NAME_MIN_COVERAGE = 0.35  # Min title token overlap with label
LLM_CLUSTER_MAX_RETRIES = 4
LLM_CLUSTER_MAX_ROUNDS = 2
LLM_CLUSTER_MAX_TOTAL_SECONDS = 600.0  # Fail fast if naming stalls too long

# Cross-Encoder Configuration
CROSS_ENCODER_ENABLED = _config.cross_encoder.enabled
CROSS_ENCODER_TOP_N = _config.cross_encoder.top_n
CROSS_ENCODER_MODEL_DIR = _config.cross_encoder.model_dir
CROSS_ENCODER_WEIGHT = _config.cross_encoder.weight

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
