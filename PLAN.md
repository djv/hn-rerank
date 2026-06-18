# hn-rewrite: Elaborated Implementation Plan

> All design decisions resolved via interview on 2026-06-17.

---

## Resolved Decisions Summary

| Decision | Choice |
|----------|--------|
| RSS in v1 | ✅ Include |
| HTTP server | stdlib `ThreadingHTTPServer` |
| ONNX model | Download pre-exported from HuggingFace Hub (no PyTorch) |
| Ranking model | Single `SVC(probability=True)`, score = `predict_proba[:, up_class]` |
| Feedback migration | One-time `migrate_feedback.py` script |
| CSS framework | Pico CSS, inlined into template (fully offline) |
| Entrypoints | Two: `generate.py` (one-shot) + `server.py` (persistent) |
| Auth | None — localhost-only binding |
| Tests | Full suite (~300 lines) |
| Diversity | MMR with cosine > 0.85 threshold |
| Logging | stdlib `logging` to stderr |
| RSS feed list | Full curated list in `config.toml` |
| Regen triggers | Feedback-driven + configurable timer in `server.py` |

---

## Deviations & Implementation Changes

During the development of the `hn-rewrite` system, several modifications were made to the original design to optimize performance, simplify execution, and ensure data integrity.

### 1. SVM Probability Simplification
* **Plan**: Class-probability estimates computed via Platt scaling.
* **Implementation**: Standard Support Vector Classifier decision values.
* **Rationale**: Platt scaling introduces substantial computational latency during model training. Instead, candidate utility scores are computed via manual softmax normalization directly over the multi-class decision values output by the SVM decision function.

### 2. Simplified Ranking Fallbacks
* **Plan**: Multi-stage fallback flow (Feedback SVM $\rightarrow$ Signal SVM $\rightarrow$ Cosine similarity fallback).
* **Implementation**: Binary fallback path:
  * **Feedback Path**: Fits the SVM using feedback records. To satisfy the multi-class constraint and prevent crashes when fewer than three classes are present in the feedback, dummy samples with zero-vectors are appended for missing classes with negligible sample weights.
  * **Fallback Path**: Directly sorts candidates by raw popularity metrics.
* **Rationale**: Bypasses legacy scraping bottlenecks and removes highly complex fallback paths in favor of a clean, robust, and deterministic baseline.

### 3. MMR Threshold Tuning
* **Plan**: High similarity threshold.
* **Implementation**: Tuned to a lower similarity threshold.
* **Rationale**: A more aggressive deduplication threshold was selected to cluster and filter similar or duplicate topics more effectively on the dashboard.

### 4. Port Number Allocation
* **Plan**: Default server port.
* **Implementation**: Alternate server port.
* **Rationale**: Prevents conflicts with the active legacy feedback server during testing and migration.

### 5. Integrity-Preserving Story Pruning
* **Plan**: General story pruning based purely on age.
* **Implementation**: Stories that are referenced by user feedback records are explicitly excluded from pruning.
* **Rationale**: Prevents cascading foreign key deletions of valuable user feedback history when candidate story records exceed the retention age limit.

### 6. Dynamic Autohide Collapse Animation
* **Plan**: Basic CSS card collapse transition.
* **Implementation**: Added inline height calculation in JavaScript immediately prior to animating the transition to zero height.
* **Rationale**: Solves a layout issue where static max-height bounds caused text truncation or sudden layout jumps during card dismissal.

### 7. LLM Detailed Analysis Endpoint
* **Plan**: Unspecified / deferred.
* **Implementation**: Created a proxy endpoint supporting large context windows, rendered client-side via a custom regex-free markdown compiler.
* **Rationale**: Provides interactive, deep summaries directly inside the user dashboard on demand.

---

## File Layout

```
~/hn-rewrite/
├── pyproject.toml              # uv project, 8 runtime deps
├── config.toml                 # Runtime config with RSS feed list
├── onnx_model/                 # Downloaded from HuggingFace Hub
│   ├── model.onnx              # ~90MB, all-MiniLM-L6-v2
│   ├── tokenizer.json
│   ├── config.json
│   └── special_tokens_map.json
├── database.py                 # SQLite storage layer
├── pipeline.py                 # Fetch → Embed → Rank → Render
├── generate.py                 # One-shot CLI entrypoint
├── server.py                   # Feedback API + static serving + regen loop
├── setup_model.py              # Download ONNX model from HuggingFace Hub
├── migrate_feedback.py         # One-time import from hn_rerank JSON
├── templates/
│   └── index.html              # Jinja2 template with inlined Pico CSS
├── public/                     # Generated output (gitignored)
│   └── index.html
└── tests/
    ├── test_database.py
    ├── test_pipeline.py
    └── test_server.py
```

---

## 1. `pyproject.toml`

```toml
[project]
name = "hn-rewrite"
version = "0.1.0"
description = "Minimalist HN reranking dashboard"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28",
    "beautifulsoup4>=4.14",
    "numpy<2.4",
    "onnxruntime>=1.23",
    "transformers>=4.57",
    "scikit-learn>=1.8",
    "jinja2>=3.1",
    "feedparser>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0",
    "pytest-asyncio>=1.3",
    "hypothesis>=6.150",
    "ruff>=0.14",
]

[tool.pytest.ini_options]
pythonpath = ["."]
asyncio_mode = "auto"
```

~25 lines. No optional extras — model download uses only `httpx` (already a dep).

---

## 2. `config.toml` + Config Dataclass

### `config.toml` (~80 lines with feed list)

```toml
[hn_rewrite]
username = "pure_coder"
db_path = "hn_rewrite.db"
output = "public/index.html"
days = 30
count = 40
onnx_model_dir = "onnx_model"
server_port = 8765
regen_interval_seconds = 10800  # 3 hours

[hn_rewrite.model]
svm_c = 0.3
svm_gamma = 0.1
svm_kernel = "rbf"
min_feedback_labels = 10       # min up + min down before SVM activates
min_signal_examples = 5        # fallback: SVM on HN signals

[hn_rewrite.rss]
enabled = true
per_feed_limit = 70
feeds = [
    "https://lobste.rs/top/rss",
    "https://tildes.net/topics.rss",
    "https://www.lesswrong.com/feed.xml?view=frontpage&karmaThreshold=20",
    "https://rss.slashdot.org/Slashdot/slashdotMain",
    "https://mshibanami.github.io/GitHubTrendingRSS/weekly/all.xml",
    "https://mshibanami.github.io/GitHubTrendingRSS/weekly/python.xml",
    "https://mshibanami.github.io/GitHubTrendingRSS/weekly/haskell.xml",
    "https://www.reddit.com/r/MachineLearning/top/.rss?t=week&limit=25",
    "https://www.reddit.com/r/programming/top/.rss?t=week&limit=25",
    "https://www.reddit.com/r/compsci/top/.rss?t=week&limit=25",
    "https://jack-clark.net/feed/",
    "https://www.construction-physics.com/feed",
    "https://www.bitsaboutmoney.com/archive/rss/",
    "https://eugeneyan.com/rss/",
    "https://huyenchip.com/feed.xml",
    "https://www.latent.space/feed",
    "https://bartoszmilewski.com/feed/",
    "https://aphyr.com/posts.atom",
    "https://googleprojectzero.blogspot.com/feeds/posts/default",
    "https://blog.trailofbits.com/feed/",
    "https://pudding.cool/feed.xml",
    "https://scottaaronson.blog/?feed=rss2",
    "https://erikbern.com/feed",
    "https://discourse.haskell.org/latest.rss",
]
```

### Config dataclass (in `pipeline.py`, ~40 lines)

```python
@dataclass(frozen=True)
class ModelConfig:
    svm_c: float = 0.3
    svm_gamma: float | str = 0.1
    svm_kernel: str = "rbf"
    min_feedback_labels: int = 10
    min_signal_examples: int = 5

@dataclass(frozen=True)
class RssConfig:
    enabled: bool = True
    per_feed_limit: int = 70
    feeds: tuple[str, ...] = ()

@dataclass(frozen=True)
class Config:
    username: str = "user"
    db_path: str = "hn_rewrite.db"
    output: str = "public/index.html"
    days: int = 30
    count: int = 40
    onnx_model_dir: str = "onnx_model"
    server_port: int = 8765
    regen_interval_seconds: int = 10800
    model: ModelConfig = field(default_factory=ModelConfig)
    rss: RssConfig = field(default_factory=RssConfig)

    @classmethod
    def load(cls, path: str = "config.toml") -> Config: ...
```

---

## 3. `database.py` (~200 lines)

### Schema

```sql
CREATE TABLE IF NOT EXISTS stories (
    id             INTEGER PRIMARY KEY,
    title          TEXT NOT NULL,
    url            TEXT,
    score          INTEGER NOT NULL DEFAULT 0,
    time           INTEGER NOT NULL DEFAULT 0,
    text_content   TEXT NOT NULL DEFAULT '',
    source         TEXT NOT NULL DEFAULT 'hn',  -- 'hn' or rss source name
    comment_count  INTEGER,
    discussion_url TEXT,
    fetched_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stories_time ON stories(time);
CREATE INDEX IF NOT EXISTS idx_stories_source ON stories(source);

CREATE TABLE IF NOT EXISTS embeddings (
    story_id      INTEGER PRIMARY KEY REFERENCES stories(id),
    model_version TEXT NOT NULL,
    embedding     BLOB NOT NULL  -- 384 × float32 = 1536 bytes
);

CREATE TABLE IF NOT EXISTS user_signals (
    story_id    INTEGER NOT NULL,
    signal_type TEXT NOT NULL,  -- 'favorite', 'upvote', 'hidden'
    scraped_at  REAL NOT NULL,
    PRIMARY KEY (story_id, signal_type)
);

CREATE TABLE IF NOT EXISTS feedback (
    story_id   INTEGER PRIMARY KEY,
    action     TEXT NOT NULL CHECK(action IN ('up', 'neutral', 'down')),
    title      TEXT NOT NULL DEFAULT '',
    url        TEXT,
    text_content TEXT NOT NULL DEFAULT '',
    source     TEXT NOT NULL DEFAULT 'hn',
    updated_at REAL NOT NULL
);
```

> [!NOTE]
> Feedback stores `title`, `url`, `text_content`, `source` so the SVM can
> train from feedback records without requiring the story to still be in the
> `stories` table (which gets pruned by TTL).

### Class API

```python
class Database:
    def __init__(self, path: str = "hn_rewrite.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # Stories
    def upsert_story(self, story: Story) -> None: ...
    def get_story(self, story_id: int) -> Story | None: ...
    def get_stories(self, ids: list[int]) -> list[Story]: ...
    def prune_stories(self, max_age_days: int = 60) -> int: ...

    # Embeddings — stored as numpy.tobytes(), loaded with np.frombuffer()
    def upsert_embedding(self, story_id: int, model_version: str, vec: NDArray) -> None: ...
    def get_embedding(self, story_id: int, model_version: str) -> NDArray | None: ...
    def get_embeddings_batch(self, ids: list[int], model_version: str) -> dict[int, NDArray]: ...

    # User signals
    def set_user_signals(self, signal_type: str, ids: set[int]) -> None: ...
    def get_user_signals(self) -> dict[str, set[int]]: ...

    # Feedback
    def upsert_feedback(self, story_id: int, action: str, title: str, url: str | None,
                        text_content: str, source: str) -> None: ...
    def get_all_feedback(self) -> list[FeedbackRecord]: ...
    def delete_feedback(self, story_id: int) -> None: ...
    def get_feedback_for_training(self) -> tuple[list[Story], list[int]]:
        """Returns (stories, labels) where label: 0=down, 1=neutral, 2=up."""
        ...
```

All queries use `?` parameterization. No f-strings in SQL.

---

## 4. `pipeline.py` — Core Logic (~650 lines)

### 4.1 Data Types (~20 lines)

```python
@dataclass(frozen=True)
class Story:
    id: int
    title: str
    url: str | None
    score: int
    time: int
    text_content: str
    source: str = "hn"
    comment_count: int | None = None
    discussion_url: str | None = None

@dataclass(frozen=True)
class FeedbackRecord:
    story_id: int
    action: Literal["up", "neutral", "down"]
    title: str
    url: str | None
    text_content: str
    source: str
    updated_at: float

@dataclass(frozen=True)
class RankedStory:
    story: Story
    score: float          # model probability or cosine similarity
    best_match_title: str # title of most similar positive signal
```

### 4.2 HN Signal Scraping (`fetch_user_signals`) (~80 lines)

Ported from [api/client.py](file:///home/dev/hn_rerank/api/client.py).

```python
async def fetch_user_signals(username: str, db: Database) -> dict[str, set[int]]:
    """Scrape favorites/upvotes/hidden from HN, persist to DB.

    Returns dict with keys: 'favorite', 'upvote', 'hidden'.
    """
```

Key behaviors preserved:
- Cookie-backed `httpx.AsyncClient` with `base_url="https://news.ycombinator.com"`
- Load cookies from `db_dir/cookies.json` if exists
- Scrape `_scrape_items(path, max_pages)` — parse `tr.athing` rows for IDs
- Pagination via `?p=N`, stop when no `a.morelink`
- Favorites: `/favorites?id={user}` (public, always available)
- Upvotes: `/upvoted?id={user}` (requires login as same user)
- Hidden: `/hidden?id={user}` (requires login as same user)
- Positive = `(favorites | upvotes) - hidden`
- Persist to `user_signals` table via `db.set_user_signals()`

**Simplified from original:**
- No URL normalization on signal items (not needed — dedup by story ID)
- No short-lived user cache TTL — always scrape fresh, persist to DB
- No login flow — assume cookies are pre-configured

### 4.3 Candidate Discovery (`fetch_candidates`) (~150 lines)

Ported from [api/fetching.py](file:///home/dev/hn_rerank/api/fetching.py).

```python
async def fetch_candidates(
    config: Config,
    exclude_ids: set[int],
    exclude_urls: set[str],
    db: Database,
) -> list[Story]:
    """Discover candidates from Algolia + RSS feeds."""
```

#### Algolia path (~100 lines)

Preserved from original:
- **Live window**: last 7 days, daily chunks, each queried separately
- **Archive window**: `config.days` - 7 days, scanned from story cache in DB
- **Algolia search endpoint**: `https://hn.algolia.com/api/v1/search`
  with `tags=story`, `numericFilters=created_at_i>=...,created_at_i<...`,
  `hitsPerPage=1000`
- **Algolia item endpoint**: `https://hn.algolia.com/api/v1/items/{id}`
  for full story + comments
- **Comment extraction**: recursive walk of `children[]`, depth-penalized
  scoring, take top-N by score
- **Text composition**: `title + comments + article_snippet` via `compose_text()`
- **Minimum filters**: `points > 5`, configurable min comments
- **Concurrency**: `asyncio.Semaphore(10)` on outbound requests
- **Negative caching**: stories that 404 or aren't type=story get cached as
  `text_content=""` to avoid re-fetching

**Simplified from original:**
- No candidate cache files — use `stories` table with `fetched_at` TTL
- No separate live/archive budgets — single budget of `config.count * 50`
- No full-text article fetching via trafilatura — use title + comments only.
  (Article text was useful for RSS but HN stories have comment text which is
  richer signal for embedding.)

#### RSS path (~50 lines)

```python
async def fetch_rss_feeds(
    feeds: list[str],
    per_feed: int,
    days: int,
    exclude_urls: set[str],
    db: Database,
) -> list[Story]:
```

Preserved:
- `feedparser.parse()` for each feed URL
- Synthetic negative ID: `-(abs(hash(entry.link)) % 2**31)` to avoid collision
  with HN IDs
- Entries filtered by age (published date within `days`)
- URL dedup against `exclude_urls` and already-seen
- Full text: use `entry.summary` or `entry.content[0].value` (feedparser gives
  this), strip HTML with BS4
- Text composition: `title + summary_text`
- Source identification: derive source name from feed URL domain
  (e.g., `lobste.rs` → `"lobsters"`)

**Simplified from original:**
- No OPML parsing — flat list from config
- No async article fetching for RSS items — use feed-provided content only
- No `langdetect` filtering
- No per-feed caching — re-fetch each run (feeds are cheap, <100 entries)

### 4.4 Embedder (`Embedder` class) (~80 lines)

Thin wrapper around ONNX InferenceSession. Ported from
[ONNXEmbeddingModel](file:///home/dev/hn_rerank/api/rerank.py#L563-L700).

```python
class Embedder:
    def __init__(self, model_dir: str = "onnx_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(
            f"{model_dir}/model.onnx",
            providers=["CPUExecutionProvider"],
        )
        self.max_tokens = 256  # from hn_embedding_model.json
        self._lock = threading.Lock()  # tokenizer is not thread-safe

    def encode(self, texts: list[str], batch_size: int = 64) -> NDArray[np.float32]:
        """Encode texts to L2-normalized 384-dim embeddings."""
```

Encoding logic (same as original):
1. Tokenize with padding + truncation to `max_tokens`
2. Run ONNX session → `last_hidden_state` (batch, seq_len, 384)
3. Mean pooling: `sum(hidden * attention_mask) / sum(attention_mask)`
4. L2 normalize: `emb / max(||emb||, 1e-12)`

Cache strategy:
- Before encoding, check `db.get_embeddings_batch(ids, model_version)`
- Only encode texts for stories with cache misses
- After encoding, `db.upsert_embedding()` for each new embedding
- `model_version` = `"all-MiniLM-L6-v2|mean|norm|256"` (stable cache key)

### 4.5 Ranker (`rank_stories`) (~120 lines)

```python
def rank_stories(
    candidates: list[Story],
    candidate_embeddings: NDArray[np.float32],
    db: Database,
    config: Config,
    embedder: Embedder,
) -> list[RankedStory]:
    """Score and rank candidates. Returns sorted descending by score."""
```

#### Scoring paths (in priority order):

**Path 1: Feedback-trained SVM** (when ≥10 up + ≥10 down feedback records)

```python
feedback_stories, feedback_labels = db.get_feedback_for_training()
feedback_texts = [s.text_content for s in feedback_stories]
feedback_embeddings = embedder.encode(feedback_texts)

svm = SVC(
    C=config.model.svm_c,
    kernel=config.model.svm_kernel,
    gamma=config.model.svm_gamma,
    probability=True,
    class_weight="balanced",
    random_state=0,
)
svm.fit(feedback_embeddings, feedback_labels)
scores = svm.predict_proba(candidate_embeddings)[:, up_class_index]
```

No `StandardScaler` — embeddings are L2-normalized (unit hypersphere),
all dimensions are on the same scale.

No ordinal threshold — single 3-class SVM. `predict_proba[:, 2]` (P(up))
is the ranking score. This replaces the dual-binary model that averaged
P(≥neutral) + P(upvote).

**Path 2: Signal-trained SVM** (when ≥5 pos + ≥5 neg HN signals but
insufficient feedback)

Same SVM architecture, trained on `positive_embeddings` (label=2) vs
`negative_embeddings` (label=0). No neutral class in this path — binary
classification, score = P(positive).

**Path 3: Cosine similarity fallback** (cold start, <5 signals)

```python
pos_mean = np.mean(positive_embeddings, axis=0, keepdims=True)
scores = cosine_similarity(candidate_embeddings, pos_mean).ravel()
```

#### Best-match identification

For display purposes, find the most similar positive signal for each candidate:

```python
if len(positive_embeddings) > 0:
    sim_matrix = cosine_similarity(candidate_embeddings, positive_embeddings)
    best_indices = np.argmax(sim_matrix, axis=1)
    # best_match_title = positive_stories[best_indices[i]].title
```

#### MMR diversity filter (~15 lines)

```python
def mmr_filter(
    stories: list[RankedStory],
    embeddings: NDArray[np.float32],
    threshold: float = 0.85,
    limit: int = 40,
) -> list[RankedStory]:
    selected: list[int] = []
    selected_embs: list[NDArray] = []
    for i, story in enumerate(stories):
        if selected_embs:
            sims = cosine_similarity([embeddings[i]], selected_embs)[0]
            if np.max(sims) > threshold:
                continue
        selected.append(i)
        selected_embs.append(embeddings[i])
        if len(selected) >= limit:
            break
    return [stories[i] for i in selected]
```

### 4.6 HTML Generation (`generate_dashboard`) (~40 lines)

```python
def generate_dashboard(
    ranked: list[RankedStory],
    output_path: Path,
    username: str,
    timestamp: str,
) -> None:
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    template = env.get_template("index.html")
    html = template.render(
        username=username,
        timestamp=timestamp,
        stories=ranked,
        server_port=config.server_port,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
```

### 4.7 Orchestrator (`run_pipeline`) (~40 lines)

```python
async def run_pipeline(config: Config) -> None:
    db = Database(config.db_path)
    embedder = Embedder(config.onnx_model_dir)

    # 1. Signals
    signals = await fetch_user_signals(config.username, db)
    positive_ids = (signals.get("favorite", set()) | signals.get("upvote", set())) \
                   - signals.get("hidden", set())
    negative_ids = signals.get("hidden", set())

    # 2. Fetch positive/negative story details
    pos_stories = await fetch_stories_by_id(list(positive_ids), db)
    neg_stories = await fetch_stories_by_id(list(negative_ids), db)

    # 3. Candidates
    exclude_ids = positive_ids | negative_ids
    exclude_urls = {s.url for s in pos_stories + neg_stories if s.url}
    candidates = await fetch_candidates(config, exclude_ids, exclude_urls, db)

    # 4. Embed candidates
    cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    # 5. Rank
    ranked = rank_stories(candidates, cand_embeddings, db, config, embedder)

    # 6. MMR diversity filter
    final = mmr_filter(ranked, cand_embeddings, limit=config.count)

    # 7. Render
    generate_dashboard(
        final,
        Path(config.output),
        config.username,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # 8. Prune old stories
    db.prune_stories(max_age_days=config.days * 2)
    logging.info("Dashboard generated: %d stories", len(final))
```

---

## 5. `setup_model.py` — Download ONNX Model (~50 lines)

Downloads pre-exported ONNX model directly from HuggingFace Hub.
No PyTorch or `optimum` needed.

```python
"""Download all-MiniLM-L6-v2 ONNX model from HuggingFace Hub."""

import httpx
from pathlib import Path

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
ONNX_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main/onnx"

FILES = {
    # ONNX model file
    f"{ONNX_BASE}/model.onnx": "model.onnx",
    # Tokenizer files (from repo root, not onnx/ subdir)
    f"{HF_BASE}/tokenizer.json": "tokenizer.json",
    f"{HF_BASE}/tokenizer_config.json": "tokenizer_config.json",
    f"{HF_BASE}/config.json": "config.json",
    f"{HF_BASE}/special_tokens_map.json": "special_tokens_map.json",
    f"{HF_BASE}/vocab.txt": "vocab.txt",
}

def download_model(target_dir: str = "onnx_model") -> None:
    out = Path(target_dir)
    if (out / "model.onnx").exists():
        print("Model already exists.")
        return
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for url, filename in FILES.items():
            dest = out / filename
            print(f"Downloading {filename}...")
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
    print(f"Model downloaded to {target_dir}/")

if __name__ == "__main__":
    download_model()
```

---

## 6. `templates/index.html` — Dashboard Template (~250 lines)

### Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HN Rerank | {{ username }}</title>
  <style>
    /* Inlined Pico CSS (minified, ~10KB) */
    {{ pico_css }}

    /* Custom overrides */
    .story-card { ... }
    .score-badge { ... }
    .source-badge { ... }
    .feedback-btn { ... }
    .feedback-btn.active { ... }
  </style>
</head>
<body>
  <main class="container">
    <header>
      <h1>HN <span style="color:#ff6600">Rerank</span></h1>
      <p>@{{ username }} · {{ timestamp }}</p>
      <select id="sort-mode">
        <option value="score" selected>By Score</option>
        <option value="date">By Date</option>
      </select>
    </header>

    <div id="stories">
      {% for item in stories %}
      <article class="story-card" data-story-id="{{ item.story.id }}"
               data-story-time="{{ item.story.time }}" data-rank="{{ loop.index0 }}">
        <div class="story-meta">
          <span class="score-badge">{{ (item.score * 100) | int }}%</span>
          {% if item.story.source != "hn" %}
          <span class="source-badge">{{ item.story.source }}</span>
          {% endif %}
          {% if item.story.score > 0 %}
          <span>{{ item.story.score }} pts</span>
          {% endif %}
          <span>{{ item.story.time | time_ago }}</span>
          {% if item.story.discussion_url %}
          <a href="{{ item.story.discussion_url }}" target="_blank">
            💬{% if item.story.comment_count is not none %} {{ item.story.comment_count }}{% endif %}
          </a>
          {% endif %}
          <span class="feedback-group">
            <button data-fb="up" title="Upvote">▲</button>
            <button data-fb="neutral" title="Neutral">✓</button>
            <button data-fb="down" title="Downvote">▼</button>
          </span>
        </div>
        <h2><a href="{{ item.story.url or item.story.discussion_url }}"
               target="_blank">{{ item.story.title }}</a></h2>
        {% if item.best_match_title %}
        <p class="match-reason">Similar to: {{ item.best_match_title }}</p>
        {% endif %}
      </article>
      {% endfor %}
    </div>

    <footer>HN Rerank · Local Semantic Analysis</footer>
  </main>

  <script>
    // Sort toggle
    // Feedback POST to /api/feedback (no auth, localhost only)
    // Highlight active feedback state from data attributes
  </script>
</body>
</html>
```

### Pico CSS integration

Pico CSS v2 minified is ~10KB. We'll download it once at build time
and inline it into the template via a Jinja2 variable `{{ pico_css }}`.

The `generate_dashboard()` function reads `pico.min.css` from a known
path and passes it to the template:

```python
PICO_CSS_PATH = Path(__file__).parent / "templates" / "pico.min.css"
pico_css = PICO_CSS_PATH.read_text() if PICO_CSS_PATH.exists() else ""
```

### Feedback JS (~30 lines)

```javascript
document.querySelectorAll('[data-fb]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const card = btn.closest('[data-story-id]');
        const action = btn.dataset.fb;
        const storyId = Number(card.dataset.storyId);
        const resp = await fetch('/api/feedback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                story_id: storyId,
                action: action,
                title: card.querySelector('h2 a')?.textContent || '',
                url: card.querySelector('h2 a')?.href || null,
                source: card.dataset.storySource || 'hn',
            }),
        });
        if (resp.ok) {
            // Highlight active button, dim others
            card.querySelectorAll('[data-fb]').forEach(b =>
                b.classList.toggle('active', b === btn));
        }
    });
});
```

---

## 7. `generate.py` — One-shot CLI (~30 lines)

```python
#!/usr/bin/env -S uv run
"""Generate the HN Rerank dashboard (one-shot)."""

import asyncio
import argparse
import logging
from pipeline import Config, run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HN Rerank dashboard")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = Config.load(args.config)
    asyncio.run(run_pipeline(config))

if __name__ == "__main__":
    main()
```

---

## 8. `server.py` — Feedback + Static Serving + Regen (~200 lines)

```python
#!/usr/bin/env -S uv run
"""Combined feedback API, static file server, and regen scheduler."""

import asyncio
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from database import Database
from pipeline import Config, run_pipeline

class Handler(BaseHTTPRequestHandler):
    server_version = "HNRewrite/1.0"
    config: Config   # set on class before server starts
    db: Database     # set on class before server starts
    regen_event: threading.Event  # set on class before server starts

    def do_GET(self) -> None:
        """Serve static files from public/."""
        ...

    def do_POST(self) -> None:
        if self.path == "/api/feedback":
            self._handle_feedback()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_feedback(self) -> None:
        """Accept feedback, persist to DB, trigger regen."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        story_id = body["story_id"]
        action = body["action"]  # "up", "neutral", "down", "clear"

        if action == "clear":
            self.db.delete_feedback(story_id)
        else:
            self.db.upsert_feedback(
                story_id=story_id,
                action=action,
                title=body.get("title", ""),
                url=body.get("url"),
                text_content=body.get("text_content", ""),
                source=body.get("source", "hn"),
            )
        self.regen_event.set()
        self._json_response({"ok": True})

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._json_response({"ok": True})

    def log_message(self, format, *args) -> None:
        return  # silence access logs


def regen_loop(config: Config, event: threading.Event) -> None:
    """Background thread: regen on event or timer."""
    while True:
        triggered = event.wait(timeout=config.regen_interval_seconds)
        if triggered:
            event.clear()
            time.sleep(2)  # debounce rapid feedback clicks
        try:
            asyncio.run(run_pipeline(config))
        except Exception:
            logging.exception("Regen failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = Config.load()
    db = Database(config.db_path)

    regen_event = threading.Event()
    Handler.config = config
    Handler.db = db
    Handler.regen_event = regen_event

    # Initial generation
    asyncio.run(run_pipeline(config))

    # Start regen thread
    t = threading.Thread(target=regen_loop, args=(config, regen_event), daemon=True)
    t.start()

    # Start HTTP server
    server = ThreadingHTTPServer(("127.0.0.1", config.server_port), Handler)
    logging.info("Serving on http://127.0.0.1:%d", config.server_port)
    server.serve_forever()
```

---

## 9. `migrate_feedback.py` — One-time Import (~60 lines)

```python
#!/usr/bin/env -S uv run
"""Import feedback from hn_rerank's JSON format into the rewrite's SQLite DB."""

import json
from pathlib import Path
from database import Database

SOURCE_PATH = Path.home() / "hn_rerank/.cache/user_feedback/dashboard_feedback.json"

def migrate(source: Path = SOURCE_PATH, db_path: str = "hn_rewrite.db") -> None:
    db = Database(db_path)
    raw = json.loads(source.read_text())
    records = raw.get("records", {})

    imported = 0
    for key, rec in records.items():
        story_id = rec.get("id")
        action = rec.get("action")
        if not story_id or action not in ("up", "neutral", "down"):
            continue
        # Also insert the story into stories table if it has text_content
        text_content = rec.get("text_content", "")
        title = rec.get("title", "")
        if text_content and title:
            db.upsert_story(Story(
                id=story_id,
                title=title,
                url=rec.get("url"),
                score=rec.get("score", 0) or 0,
                time=rec.get("time", 0),
                text_content=text_content,
                source=rec.get("source", "hn"),
                comment_count=rec.get("comment_count"),
                discussion_url=rec.get("discussion_url"),
            ))
        db.upsert_feedback(
            story_id=story_id,
            action=action,
            title=title,
            url=rec.get("url"),
            text_content=text_content,
            source=rec.get("source", "hn"),
        )
        imported += 1
    print(f"Imported {imported} feedback records into {db_path}")
```

---

## 10. Test Plan (~300 lines)

### `test_database.py` (~100 lines)

| Test | What it validates |
|------|------------------|
| `test_upsert_and_get_story` | Round-trip story persistence |
| `test_upsert_embedding_roundtrip` | BLOB storage/retrieval matches original ndarray |
| `test_get_embeddings_batch` | Batch retrieval returns correct subset |
| `test_feedback_crud` | Insert, update, delete, list feedback records |
| `test_user_signals_overwrite` | `set_user_signals` replaces previous signals |
| `test_prune_stories` | Old stories deleted, recent kept |
| `test_feedback_training_data` | `get_feedback_for_training` returns correct labels |

### `test_pipeline.py` (~120 lines)

| Test | What it validates |
|------|------------------|
| `test_embedder_output_shape` | `encode()` returns (N, 384) normalized array |
| `test_embedder_cache_hit` | Second call uses DB cache, doesn't re-encode |
| `test_rank_svm_path` | SVM path activates with ≥10 up + ≥10 down labels |
| `test_rank_cosine_fallback` | Cosine path activates with <5 signals |
| `test_mmr_dedup` | Near-duplicate stories filtered at threshold |
| `test_compose_text` | Title + comments composed correctly |
| `test_rss_synthetic_id` | RSS entries get stable negative IDs |

Property-based test:
```python
@given(st.lists(st.floats(0, 1, allow_nan=False), min_size=2, max_size=100))
def test_mmr_output_is_subset(scores):
    """MMR output is a subset of input, preserves order."""
```

### `test_server.py` (~80 lines)

| Test | What it validates |
|------|------------------|
| `test_feedback_post` | POST /api/feedback persists to DB |
| `test_feedback_clear` | action="clear" deletes record |
| `test_static_serving` | GET / serves public/index.html |
| `test_cors_headers` | OPTIONS returns CORS headers |

---

## Implementation Order

| Step | File(s) | Est. Lines | Dependencies |
|------|---------|-----------|--------------|
| 1 | `pyproject.toml` + `config.toml` | 105 | None |
| 2 | `database.py` + `test_database.py` | 300 | Step 1 |
| 3 | `setup_model.py` | 50 | Step 1 (httpx) |
| 4 | `pipeline.py` — Embedder class + `test_pipeline.py` (embed tests) | 120 | Steps 2-3 |
| 5 | `pipeline.py` — Ranker (SVM + fallback + MMR) + tests | 150 | Step 4 |
| 6 | `pipeline.py` — HN signal scraping | 80 | Step 2 |
| 7 | `pipeline.py` — Algolia candidates | 150 | Step 2 |
| 8 | `pipeline.py` — RSS feeds | 50 | Step 2 |
| 9 | `pipeline.py` — HTML generation + `templates/index.html` | 290 | Step 5 |
| 10 | `generate.py` | 30 | Steps 6-9 |
| 11 | `server.py` + `test_server.py` | 280 | Step 10 |
| 12 | `migrate_feedback.py` | 60 | Step 2 |
| **Total** | | **~1,665** | |

---

## What's NOT in this plan (explicitly deferred)

- KMeans clustering and cluster visualization
- LLM TL;DR generation (Mistral/Groq)
- Impression tracking / telemetry features
- Explore/exploit acquisition slots
- HN action mirroring (upvote on HN when upvoted in dashboard)
- Multiple ML model types (MLP, RF, GBT)
- Metadata features (source_trust, story_age, local_density, etc.)
- Open-index / DuckDB archive
- `trafilatura` full-text extraction
- Systemd service files / Caddy proxy
- Authentication on feedback API
- URL normalization library

> [!IMPORTANT]
> The ONNX model will be downloaded from HuggingFace Hub using only `httpx`.
> No PyTorch, `optimum`, or `sentence-transformers` needed at any point.

> [!WARNING]
> The existing `hn_rerank` project remains untouched. The rewrite shares no
> runtime state, no database files, and no cache directories with the original.
