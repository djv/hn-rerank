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
STORY_CACHE_MAX_FILES = 10000  # LRU eviction threshold

# Concurrency
EXTERNAL_REQUEST_SEMAPHORE = 10  # Reduced to avoid API throttling

# Comment Pool
MIN_STORY_COMMENTS = 10  # Filter in Algolia query + fetch validation
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
EMBEDDING_MODEL_VERSION = "v6"  # Bump to invalidate cache on model change

# Recency Weighting
RECENCY_DECAY_RATE = 0.003

# Similarity Bounds
SIMILARITY_MIN = -1.0
SIMILARITY_MAX = 1.0
SEMANTIC_MATCH_THRESHOLD = 0.50

# Multi-Interest Clustering
MIN_CLUSTERS = 2
MAX_CLUSTERS = 50
MIN_SAMPLES_PER_CLUSTER = 2  # Smaller clusters = more granularity
