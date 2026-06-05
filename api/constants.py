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
ARTICLE_SNIPPET_LENGTH = 1500
TEXT_CONTENT_MAX_TOKENS = 512

# Cache Paths
EMBEDDING_CACHE_DIR = ".cache/embeddings"
EMBEDDING_CACHE_MAX_FILES = 20000
STORY_CACHE_DIR = ".cache/stories"
STORY_CACHE_VERSION = "v2"
USER_CACHE_DIR = ".cache/user"
STORY_CACHE_TTL = 259200  # 72 hours
USER_CACHE_TTL = 120  # 2 minutes
CANDIDATE_CACHE_DIR = ".cache/candidates"
CANDIDATE_CACHE_VERSION = "v2"
CANDIDATE_CACHE_TTL_SHORT = 1800  # 30 minutes
CANDIDATE_CACHE_TTL_LONG = 302400  # 3.5 days
CANDIDATE_CACHE_TTL_ARCHIVE = 3888000  # 45 days
STORY_CACHE_MAX_FILES = 25000  # LRU eviction threshold
RSS_CACHE_DIR = ".cache/rss"
RSS_OPML_CACHE_TTL = 43200  # 12 hours
RSS_FEED_CACHE_TTL = 43200  # 12 hours
RSS_FEED_CACHE_VERSION = 3
RSS_ARTICLE_CACHE_TTL = 43200  # 12 hours
RSS_CACHE_MAX_FILES = 5000

# Concurrency
EXTERNAL_REQUEST_SEMAPHORE = 10  # Reduced to avoid API throttling

# User Limits
MAX_USER_STORIES = 2000

# Discovery Pool
ALGOLIA_MIN_POINTS = 5
ALGOLIA_DEFAULT_DAYS = 30
CANDIDATE_FETCH_COUNT = 2000
RSS_OPML_URL = "api/popular-blogs-2025.opml"
RSS_EXTRA_FEEDS = [
    "https://hackaday.com/blog/feed/",
    "https://jack-clark.net/feed/",
    "https://lobste.rs/top/rss",
    "https://tildes.net/topics.rss",
    "https://www.lesswrong.com/feed.xml?view=frontpage&karmaThreshold=20",
    "https://rss.slashdot.org/Slashdot/slashdotMain",
    "https://mshibanami.github.io/GitHubTrendingRSS/monthly/python.xml",
    "https://mshibanami.github.io/GitHubTrendingRSS/monthly/all.xml",
    "https://mshibanami.github.io/GitHubTrendingRSS/monthly/haskell.xml",
    "https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25",
    "https://www.reddit.com/r/programming/top/.rss?t=week&limit=25",
    "https://www.reddit.com/r/compsci/top/.rss?t=week&limit=25",
    "https://digg.com/ai",
    "https://www.construction-physics.com/feed",
    "https://pedestrianobservations.com/feed/",
    "https://theprepared.org/feed/",
    "https://www.bitsaboutmoney.com/archive/rss/",
    "https://eugeneyan.com/rss/",
    "https://huyenchip.com/feed.xml",
    "https://www.latent.space/feed",
    "https://thesequence.substack.com/feed",
    "https://gowers.wordpress.com/feed/",
    "https://johncarlosbaez.wordpress.com/feed/",
    "https://bartoszmilewski.com/feed/",
    "https://okmij.org/ftp/rss.xml",
    "https://aphyr.com/posts.atom",
    "https://googleprojectzero.blogspot.com/feeds/posts/default",
    "https://blog.trailofbits.com/feed/",
    "https://felt.com/blog/feed.xml",
    "https://pudding.cool/feed.xml",
    "https://nautil.us/feed/",
    "https://progressforum.org/feed.xml",
    "https://stratechery.com/feed/",
    "https://www.stephendiehl.com/posts.rss",
    "https://www.citationneeded.news/rss/",
    "https://scottaaronson.blog/?feed=rss2",
    "https://erikbern.com/feed",
    "https://kevinlynagh.com/feed.xml",
    "https://feedpress.me/thetechnium",
    "https://www.countbayesie.com/blog?format=rss",
    "https://anderegg.ca/feed.xml",
    "https://artemis.sh/feed.xml",
    "https://blog.davep.org/feed.xml",
    "https://den.dev/index.xml",
    "https://jonlu.ca/feed.xml",
    "https://ludic.mataroa.blog/rss/",
    "https://rednafi.com/index.xml",
    "https://taylor.town/feed.xml",
]  # Additional feeds not in OPML
RSS_MAX_FEEDS = 0  # 0 = no max feed limit
RSS_PER_FEED_LIMIT = 70
RSS_CURATED_NEWS_PER_FEED_LIMIT = 50
RSS_ALLOWED_SOURCE_LANGUAGES = ("en", "fr", "es")

# Inference
DEFAULT_EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MIN_CLIP = 1e-9
# BGE-small CLS/query was selected by the local model bakeoff.
EMBEDDING_MODEL_VERSION = "all-MiniLM-L6-v2-2026-05-26"

# Similarity Bounds
SIMILARITY_MIN = -1.0

# Multi-Interest Clustering

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
LLM_CLUSTER_NAME_PROMPT_VERSION = "v15"
LLM_TLDR_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.2
LLM_TLDR_MAX_TOKENS = 2000
LLM_CLUSTER_MAX_TOKENS = 150
LLM_TLDR_BATCH_SIZE = 2  # Stories per TLDR request
LLM_CLUSTER_TITLE_MAX_CHARS = 140  # Max chars per title in naming payload
LLM_CLUSTER_CONTEXT_STORY_COUNT = 5  # Top stories per cluster that include context
LLM_CLUSTER_CONTEXT_MAX_CHARS = 500  # Max chars of story context in naming payload
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
