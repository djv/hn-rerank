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
TEXT_CONTENT_MAX_LENGTH = 8000

# Cache Paths
EMBEDDING_CACHE_DIR = ".cache/embeddings"
STORY_CACHE_DIR = ".cache/stories"
USER_CACHE_DIR = ".cache/user"
STORY_CACHE_TTL = 86400
USER_CACHE_TTL = 120  # 2 minutes
CANDIDATE_CACHE_DIR = ".cache/candidates"
CANDIDATE_CACHE_TTL_SHORT = 1800  # 30 minutes
CANDIDATE_CACHE_TTL_LONG = 604800  # 1 week
CANDIDATE_CACHE_TTL_ARCHIVE = 7776000  # 90 days
STORY_CACHE_MAX_FILES = 25000  # LRU eviction threshold

# Concurrency
EXTERNAL_REQUEST_SEMAPHORE = 10  # Reduced to avoid API throttling

# Comment Pool
MIN_STORY_COMMENTS = 15  # Filter in Algolia query + fetch validation
MAX_COMMENTS_COLLECTED = 100
TOP_COMMENTS_FOR_RANKING = 50
TOP_COMMENTS_FOR_UI = 10
RANKING_DEPTH_PENALTY = 10
MIN_COMMENT_LENGTH = 50  # Filter short low-value comments

# User Limits
MAX_USER_STORIES = 1000

# Discovery Pool
ALGOLIA_MIN_POINTS = 5
ALGOLIA_DEFAULT_DAYS = 30
CANDIDATE_FETCH_COUNT = 200

# Inference
DEFAULT_EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MIN_CLIP = 1e-9
EMBEDDING_MODEL_VERSION = "v8-base"  # Reset to base model after tuning experiment

# Similarity Bounds
SIMILARITY_MIN = -1.0
SIMILARITY_MAX = 1.0
SEMANTIC_MATCH_THRESHOLD = 0.50
KNN_NEIGHBORS = 2  # Optimized: 2 neighbors provide best stability

# Multi-Interest Clustering
DEFAULT_CLUSTER_COUNT = 12  # Fixed k; LLM naming handles coherence
MIN_CLUSTERS = 2
MAX_CLUSTERS = 50
MIN_SAMPLES_PER_CLUSTER = 2  # Smaller clusters = more granularity
CLUSTER_SIMILARITY_THRESHOLD = 0.70  # Min similarity to belong to a cluster

# Ranking Weights
RANKING_HN_WEIGHT = 0.05  # Weight for HN score vs semantic (legacy, unused with adaptive)
RANKING_NEGATIVE_WEIGHT = 0.5  # Penalty for similarity to hidden stories
RANKING_DIVERSITY_LAMBDA = 0.30  # MMR diversity penalty (lowered from 0.45 for better NDCG)
RANKING_DIVERSITY_LAMBDA_CLASSIFIER = 0.3  # Moderate diversity for classifier (was 0.6, lowered for NDCG)
RANKING_MAX_RESULTS = 300  # Max stories to rank via MMR (increased to ensure coverage)

# Adaptive HN Weight (age-based)
ADAPTIVE_HN_WEIGHT_MIN = 0.05  # For stories < 6h old (trust semantic)
ADAPTIVE_HN_WEIGHT_MAX = 0.25  # For stories > 48h old (trust HN score)
ADAPTIVE_HN_THRESHOLD_YOUNG = 6  # Hours - below this, use min weight
ADAPTIVE_HN_THRESHOLD_OLD = 48  # Hours - above this, use max weight

# Freshness Decay
FRESHNESS_HALF_LIFE_HOURS = 24.0  # Score halves every 24 hours
FRESHNESS_MAX_BOOST = 0.15  # Max freshness contribution to hybrid score

# Semantic Scoring
SEMANTIC_MAXSIM_WEIGHT = 0.95  # Weight for max cluster similarity
SEMANTIC_MEANSIM_WEIGHT = 0.05  # Weight for mean cluster similarity
SEMANTIC_SIGMOID_K = 15.0  # Steepness of sigmoid activation
SEMANTIC_SIGMOID_THRESHOLD = 0.30  # Optimized: Lower threshold captures more relevant signals
HN_SCORE_NORMALIZATION_CAP = 500  # Cap for log normalization of HN points

# LLM Configuration
LLM_CLUSTER_NAME_MODEL = "llama-3.3-70b-versatile"
LLM_TLDR_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
LLM_TLDR_MAX_TOKENS = 2000
LLM_CLUSTER_BATCH_SIZE = 10  # Clusters per API request
LLM_TLDR_BATCH_SIZE = 5  # Stories per TLDR request
LLM_CLUSTER_NAME_MAX_WORDS = 6  # Max words in cluster name

# Rate Limiting (Token Bucket)
RATE_LIMIT_REFILL_RATE = 0.25  # Tokens per second (1 call per 4 seconds)
RATE_LIMIT_MAX_TOKENS = 1.0
RATE_LIMIT_JITTER_MAX = 0.5  # Max random jitter in seconds
RATE_LIMIT_429_BACKOFF_BASE = 20.0  # Base delay on 429 response
RATE_LIMIT_ERROR_BACKOFF_BASE = 10.0  # Base delay on other errors
LLM_HTTP_TIMEOUT = 30.0  # HTTP request timeout in seconds
