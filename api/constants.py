"""
Constants and configuration values for HN reranking.
"""

# HN Score Calculation (Gravity Decay Formula)
HN_SCORE_POINTS_EXP = 0.8  # Exponent for (P - 1)^x
HN_SCORE_TIME_EXP = 1.8  # Exponent for (T + 2)^x
HN_SCORE_TIME_OFFSET = 2  # Hours offset for denominator

# Similarity Bounds
SIMILARITY_MIN = -1.0
SIMILARITY_MAX = 1.0

# Article Text Truncation (characters)
ARTICLE_RANKING_LENGTH = 2000  # For ranking embeddings
ARTICLE_SNIPPET_LENGTH = 1000  # For UI display
TEXT_CONTENT_MAX_LENGTH = 5000  # Total text content limit

# Embedding Cache
EMBEDDING_CACHE_DIR = ".cache/embeddings"

# Story Cache
STORY_CACHE_DIR = ".cache/stories"
STORY_CACHE_TTL = 86400  # 24 hours in seconds

# Concurrency Limits
EXTERNAL_REQUEST_SEMAPHORE = 50  # Max concurrent external HTTP requests

# Clustering
CLUSTER_DISTANCE_THRESHOLD = 0.8  # Agglomerative clustering threshold
CLUSTER_MIN_NORM = 1e-9  # Minimum norm for normalization

# Comment Collection
MAX_COMMENTS_COLLECTED = 40  # Max comments to collect from story
TOP_COMMENTS_FOR_RANKING = 5  # Top N comments used in ranking
TOP_COMMENTS_FOR_UI = 10  # Top N comments stored for display

# User Data Limits
MAX_USER_STORIES = 50  # Max stories to fetch per user for ranking

# Algolia Search
ALGOLIA_MIN_POINTS = 20  # Minimum story score for candidates
ALGOLIA_DEFAULT_DAYS = 30  # Default time window for story search

# Embedding Model
DEFAULT_EMBEDDING_BATCH_SIZE = 4
EMBEDDING_MIN_CLIP = 1e-9  # Minimum value for clipping in normalization

# Recency Weighting (Exponential Decay)
# Formula: weight = exp(-RECENCY_DECAY_RATE × age_in_days)
# Default decay: 0.01 means ~99% weight at 1 day, ~90% at 10 days, ~37% at 100 days
RECENCY_DECAY_RATE = 0.01  # Per-day decay rate for positive story weights
