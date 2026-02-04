"""
Constants and configuration values for HN reranking.
"""

# HN Score Calculation (Gravity Decay Formula)
HN_SCORE_POINTS_EXP = 0.8
HN_SCORE_TIME_EXP = 1.8
HN_SCORE_TIME_OFFSET = 2

# Content Limits
ARTICLE_RANKING_LENGTH = 3000
ARTICLE_SNIPPET_LENGTH = 1000
TEXT_CONTENT_MAX_TOKENS = 512

# Cache Paths
EMBEDDING_CACHE_DIR = ".cache/embeddings"
CLUSTER_EMBEDDING_CACHE_DIR = ".cache/embeddings_cluster"
EMBEDDING_CACHE_MAX_FILES = 20000
STORY_CACHE_DIR = ".cache/stories"
USER_CACHE_DIR = ".cache/user"
STORY_CACHE_TTL = 86400
USER_CACHE_TTL = 120  # 2 minutes
CANDIDATE_CACHE_DIR = ".cache/candidates"
CANDIDATE_CACHE_TTL_SHORT = 1800  # 30 minutes
CANDIDATE_CACHE_TTL_LONG = 604800  # 1 week
CANDIDATE_CACHE_TTL_ARCHIVE = 7776000  # 90 days
STORY_CACHE_MAX_FILES = 25000  # LRU eviction threshold
RSS_CACHE_DIR = ".cache/rss"
RSS_OPML_CACHE_TTL = 86400  # 1 day
RSS_FEED_CACHE_TTL = 3600  # 1 hour
RSS_ARTICLE_CACHE_TTL = 86400  # 1 day
RSS_CACHE_MAX_FILES = 5000

# Concurrency
EXTERNAL_REQUEST_SEMAPHORE = 10  # Reduced to avoid API throttling

# User Limits
MAX_USER_STORIES = 2000

# Discovery Pool
ALGOLIA_MIN_POINTS = 5
ALGOLIA_DEFAULT_DAYS = 30
CANDIDATE_FETCH_COUNT = 500
RSS_OPML_URL = (
    "https://gist.githubusercontent.com/emschwartz/e6d2bf860ccc367fe37ff953ba6de66b/raw/hn-popular-blogs-2025.opml"
)
RSS_EXTRA_FEEDS = [
    "https://jack-clark.net/feed/",
]  # Additional feeds not in OPML
RSS_MAX_FEEDS = 0  # 0 = no max feed limit
RSS_PER_FEED_LIMIT = 5

# Inference
DEFAULT_EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MIN_CLIP = 1e-9
EMBEDDING_MODEL_VERSION = "v10-tuned"  # Fine-tuned on 218 triplets (96% accuracy)
CLUSTER_EMBEDDING_MODEL_VERSION = "v1-base"
CLUSTER_EMBEDDING_MODEL_DIR = "onnx_model_backup_base"

# Similarity Bounds
SIMILARITY_MIN = -1.0
SIMILARITY_MAX = 1.0
SEMANTIC_MATCH_THRESHOLD = 0.50
KNN_NEIGHBORS = 5  # Optuna (10-fold CV, 500 candidates)

# Multi-Interest Clustering
DEFAULT_CLUSTER_COUNT = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 40
MIN_SAMPLES_PER_CLUSTER = 2  # Minimum cluster size (samples per cluster)
MAX_CLUSTER_FRACTION = 0.25  # Max cluster size vs total signals
MAX_CLUSTER_SIZE = 40  # Absolute max cluster size
CLUSTER_REFINE_ITERS = 2  # Reassign to nearest centroid for tighter clusters
CLUSTER_SIMILARITY_THRESHOLD = 0.85  # Min similarity to belong to a cluster (raised for fine-tuned model)

# Ranking Weights
RANKING_HN_WEIGHT = 0.05  # Weight for HN score vs semantic (legacy, unused with adaptive)
RANKING_NEGATIVE_WEIGHT = 0.3829876086  # Optuna (10-fold CV, 500 candidates)
RANKING_DIVERSITY_LAMBDA = 0.1730644067  # Optuna (10-fold CV, 500 candidates)
RANKING_DIVERSITY_LAMBDA_CLASSIFIER = 0.1730644067  # Match default for classifier
RANKING_MAX_RESULTS = 500  # Max stories to rank via MMR (Increased)

# Adaptive HN Weight (age-based)
# Optimization showed preference for strong semantic signal (low HN weight)
ADAPTIVE_HN_WEIGHT_MIN = 0.0808211678  # For stories < 6h old (Optuna 10-fold CV)
ADAPTIVE_HN_WEIGHT_MAX = 0.1142705634  # For stories > 48h old (Optuna 10-fold CV)
ADAPTIVE_HN_THRESHOLD_YOUNG = 6  # Hours - below this, use min weight
ADAPTIVE_HN_THRESHOLD_OLD = 48  # Hours - above this, use max weight

# Freshness Decay
FRESHNESS_HALF_LIFE_HOURS = 71.4907953229  # Optuna (10-fold CV, 500 candidates)
FRESHNESS_MAX_BOOST = 0.00220351924156  # Optuna (10-fold CV, 500 candidates)

# Semantic Scoring
SEMANTIC_MAXSIM_WEIGHT = 0.95  # Weight for max cluster similarity
SEMANTIC_MEANSIM_WEIGHT = 0.05  # Weight for mean cluster similarity
SEMANTIC_SIGMOID_K = 31.2249293861  # Optuna (10-fold CV, 500 candidates)
SEMANTIC_SIGMOID_THRESHOLD = 0.4749411784  # Optuna (10-fold CV, 500 candidates)
HN_SCORE_NORMALIZATION_CAP = 500  # Cap for log normalization of HN points

# Clustering
CLUSTER_ALGORITHM = "spectral"  # "spectral", "agglomerative", or "kmeans"
CLUSTER_AGGLOMERATIVE_LINKAGE = "complete"
CLUSTER_AGGLOMERATIVE_METRIC = "cosine"
CLUSTER_SPECTRAL_NEIGHBORS = 15

# Comment Pool
MIN_STORY_COMMENTS = 20  # Filter in Algolia query + fetch validation
MAX_COMMENTS_COLLECTED = 200  # Increased for richer signal
TOP_COMMENTS_FOR_RANKING = 150  # Use more comments for embedding (Increased)
TOP_COMMENTS_FOR_UI = 10
RANKING_DEPTH_PENALTY = 10
MIN_COMMENT_LENGTH = 30  # Filter short low-value comments (relaxed)

# LLM Configuration
LLM_CLUSTER_NAME_MODEL_PRIMARY = "llama-3.3-70b-versatile"
LLM_CLUSTER_NAME_MODEL_FALLBACK = "llama-3.1-8b-instant"
LLM_CLUSTER_NAME_PROMPT_VERSION = "v2"
LLM_TLDR_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
LLM_TLDR_MAX_TOKENS = 2000
LLM_CLUSTER_MAX_TOKENS = 64
LLM_TLDR_BATCH_SIZE = 5  # Stories per TLDR request
LLM_CLUSTER_TITLE_SAMPLES = 20  # Titles per cluster for naming context
LLM_CLUSTER_TITLE_MAX_CHARS = 140  # Max chars per title in naming payload
LLM_CLUSTER_NAME_MAX_WORDS = 6  # Max words in cluster name
LLM_CLUSTER_NAME_MIN_COVERAGE = 0.35  # Min title token overlap with label
LLM_CLUSTER_MAX_RETRIES = 4
LLM_CLUSTER_MAX_ROUNDS = 2
LLM_CLUSTER_MAX_TOTAL_SECONDS = 600.0  # Fail fast if naming stalls too long
LLM_CLUSTER_RESCUE_RETRIES = 2  # Extra retries for single-cluster rescue pass

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
LLM_MIN_REQUEST_INTERVAL = 6.0  # Minimum seconds between Groq requests
LLM_429_COOLDOWN_BASE = 2.0  # Base cooldown when rate-limited
LLM_429_COOLDOWN_MAX = 60.0  # Max cooldown when rate-limited
