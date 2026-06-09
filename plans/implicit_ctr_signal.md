# Plan: Implicit CTR Telemetry Plumbing (Phase 1)

Implement the frontend visible impressions tracking (IntersectionObserver) and update the SQLite schema to log original URLs. This ensures we collect clean, position-bias-free telemetry immediately. All model training and Python label integration (Phase 2) is deferred until we have accumulated sufficient data (6+ months).

## Two-phase approach

| Phase | Status | What | Why |
|-------|--------|------|-----|
| **Phase 1** | **Active (This PR)** | Add `IntersectionObserver` to dashboard; update SQLite schema to log `url` column | Collect clean telemetry for future use |
| **Phase 2** | **Deferred** | Build Python label pipeline, sample weights, config additions, model training | Defer until we have ~6+ months of clean data |

---

## Summary of Changes (Phase 1 Active Only)

| File | Status | Change |
|------|--------|--------|
| `generate_html.py` | **Active** | Replace fire-all-impressions-on-load with `IntersectionObserver`-based per-card visible impressions |
| `api/impressions.py` | **Active** | Update `telemetry_events` schema to add nullable `url` column, support payload url logging |
| `scripts/feedback_server.py` | **Active** | Update impressions logging parser to parse and insert the `url` column |
| `api/implicit_feedback.py` | *Deferred* | New module — query SQLite for viewport-logged impressions, merge with explicit labels |
| `api/feedback_single_model.py` | *Deferred* | Merge implicit labels into `build_single_model_feedback_labels()` |
| `api/ordinal_model.py` | *Deferred* | `train_model_from_matrix()` accepts `sample_weight` kwarg and raw safeguards |
| `hn_rerank.toml` | *Deferred* | New `[hn_rerank.implicit]` config section |
| `api/config.py` | *Deferred* | New `ImplicitConfig` dataclass |
| `tests/test_telemetry.py` | **Active** | Add unit tests for schema `url` serialization and IntersectionObserver telemetry payload |
| `tests/test_implicit_feedback.py` | *Deferred* | Unit tests for label derivation, merge dedup, sample weights |

---

## Step 1: IntersectionObserver in dashboard JS (Active)

**File**: `generate_html.py`, the JS embedded around line 360.

Current behavior: `sendImpressions()` fires immediately on page load after `syncServerFeedback()`, collecting ALL cards with `[data-rank-index]` via `querySelectorAll`. Every card gets an impression regardless of whether the user scrolls to it.

New behavior — replace `sendImpressions` with an `IntersectionObserver`:

```js
let impressionTimer = null;
const pendingImpressions = [];

const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
        if (entry.isIntersecting) {
            const card = entry.target;
            if (card.dataset.impressionSent) return;
            card.dataset.impressionSent = '1';

            pendingImpressions.push({
                event: 'impression',
                feedback_key: card.dataset.feedbackKey,
                story_id: Number(card.dataset.storyId),
                story_source: card.dataset.storySource,
                title: card.dataset.storyTitle,
                url: card.dataset.storyUrl || '',
                rank_index: Number(card.dataset.rankIndex),
                model_score: Number(card.dataset.modelScore),
                knn_score: Number(card.dataset.knnScore),
                max_sim_score: Number(card.dataset.maxSimScore),
                max_cluster_score: Number(card.dataset.maxClusterScore),
                acquisition_kind: card.dataset.acquisitionKind || 'exploit',
                config_hash: CONFIG_HASH,
            });

            observer.unobserve(card);

            // Debounce: batch-send pending after 2s of no new visible cards
            if (impressionTimer) clearTimeout(impressionTimer);
            impressionTimer = setTimeout(() => {
                if (pendingImpressions.length === 0) return;
                const batch = pendingImpressions.splice(0);
                const token = localStorage.getItem(TOKEN_KEY);
                if (!token) return;
                fetch(IMPRESSIONS_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-HN-RERANK-FEEDBACK-TOKEN': token },
                    body: JSON.stringify({ impressions: batch }),
                }).catch(() => {});
            }, 2000);
        }
    }
}, { rootMargin: '0px 0px 200px 0px' });

// After render/hide-acted:
for (const card of cards) observer.observe(card);
```

Key design decisions:
- **`rootMargin: 200px`** — pre-loads impressions for cards just below the fold, so fast scrollers don't miss impressions
- **2s debounce** — batches multiple visible cards into one POST, avoids one request per card
- **`observer.unobserve(card)` after first fire** — no duplicate impressions
- **Token check only at send time** — if token is missing at page load but set later, cards already sent won't re-fire (impressionSent flag)
- **`sendBeacon` on beforeunload** — flush any pending impressions if the user closes the tab within the 2s debounce window:

```js
const flushImpressions = () => {
    if (impressionTimer) clearTimeout(impressionTimer);
    if (pendingImpressions.length === 0) return;
    const batch = pendingImpressions.splice(0);
    navigator.sendBeacon(IMPRESSIONS_URL,
        new Blob([JSON.stringify({ impressions: batch })], { type: 'application/json' }));
};
window.addEventListener('beforeunload', flushImpressions);
```

Remove the old `sendImpressions()` function entirely. The old `hidePreviouslyActedCards()` + `syncServerFeedback()` logic runs first (removing already-voted cards), then the observer starts watching remaining cards — so voted cards never get impressions.

Also, update `logClick` (around line 451 in `generate_html.py`) to send `url` in the click event payload:

```js
// Inside the card click handler, where it builds the click impression:
const clickRecord = {
    event: 'click',
    feedback_key: card.dataset.feedbackKey,
    story_id: Number(card.dataset.storyId),
    story_source: card.dataset.storySource,
    title: card.dataset.storyTitle,
    url: card.dataset.storyUrl || '',       // NEW
    rank_index: Number(card.dataset.rankIndex),
    model_score: Number(card.dataset.modelScore),
    knn_score: Number(card.dataset.knnScore),
    max_sim_score: Number(card.dataset.maxSimScore),
    max_cluster_score: Number(card.dataset.maxClusterScore),
    acquisition_kind: card.dataset.acquisitionKind || 'exploit',
    config_hash: CONFIG_HASH,
};
```

After deployment, existing contaminated data stays in SQLite (from the old fire-all approach). New data starts accumulating with clean viewport-based impressions. We'll need to distinguish old vs new — see Step 5.

---

## Step 1b: Database schema update (Active)

**File**: `api/impressions.py` & `scripts/feedback_server.py`

### Schema migration

`init_event_schema` currently uses `CREATE TABLE IF NOT EXISTS`, which won't modify an existing table. Add a migration step after the CREATE TABLE:

```python
def init_event_schema(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    if conn is None:
        conn = connect_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            event TEXT NOT NULL CHECK (event IN ('impression', 'click')),
            server_ts REAL NOT NULL,
            feedback_key TEXT NOT NULL,
            story_id INTEGER NOT NULL,
            story_source TEXT NOT NULL,
            title TEXT NOT NULL,
            rank_index INTEGER NOT NULL,
            model_score REAL NOT NULL,
            knn_score REAL NOT NULL,
            max_sim_score REAL NOT NULL,
            max_cluster_score REAL NOT NULL,
            acquisition_kind TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            url TEXT   -- may not exist on older DBs; added via ALTER below
        )
    """)
    # Schema migration: add url column if it doesn't exist (idempotent)
    try:
        conn.execute("ALTER TABLE telemetry_events ADD COLUMN url TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists (or other transient — safe to ignore)
    ...
```

### Dataclass and parser

Add `url: str = ""` to `ImpressionRecord` dataclass. Update `impression_from_payload` to extract `url` from payload dict. Update `_insert_one` to include the `url` column in the INSERT statement.

Backwards compat: payloads without `url` produce `""`, which inserts as NULL into the nullable column. Old entries in the DB get `NULL` in the `url` column.

---

## Step 2: New module `api/implicit_feedback.py` (Phase 2 - Deferred)

### 2a. Query clean impressions

Only impressions logged **after** the IntersectionObserver deployment are valid. We need a cutoff timestamp.

```python
OBSERVER_DEPLOYMENT_TS: float = ...  # set at deployment time


def fetch_clean_impressions(
    conn: sqlite3.Connection,
    min_impressions: int = 2,
) -> list[sqlite3.Row]:
    """Fetch impression/click events logged via IntersectionObserver."""
    return conn.execute(
        """
        SELECT
            feedback_key,
            story_id,
            -- All impressions of the same story should have the same url;
            -- pick the first non-empty value as a tie-breaker.
            COALESCE(
                MAX(CASE WHEN url IS NOT NULL AND url != '' THEN url END),
                ''
            ) as url,
            SUM(CASE WHEN event='impression' THEN 1 ELSE 0 END) as impressions,
            SUM(CASE WHEN event='click' THEN 1 ELSE 0 END) as clicks
        FROM telemetry_events
        WHERE server_ts >= ?
        GROUP BY feedback_key
        HAVING impressions >= ?
        """,
        (OBSERVER_DEPLOYMENT_TS, min_impressions),
    ).fetchall()
```

### 2b. Story reconstruction from local cache

```python
def _reconstruct_story(
    feedback_key: str,
    story_id: int,
    url: str | None = None,
) -> Story | None:
    """
    Try to build a Story object for a story that has impression data
    but no explicit FeedbackRecord. Checks:
      1. .cache/stories/{story_id}.json (for HN stories)
      2. .cache/rss/article-{sha256}.json (for RSS stories using url parameter)
    Returns None if not found in any cache (skip, don't scrape).
    """
```

### 2c. Derive implicit labels

With viewport-filtered data, no residual math is needed:

```python
@dataclass(frozen=True)
class ImplicitLabel:
    key: str
    story: Story | None   # None if story text unavailable
    ordinal_label: int     # DOWNVOTE_LABEL / UPVOTE_LABEL
    sample_weight: float
    source: str = "implicit_ctr"


def derive_implicit_labels(
    conn: sqlite3.Connection,
    feedback_records: dict[str, FeedbackRecord],
    config: ImplicitConfig,
) -> list[ImplicitLabel]:
```

| Condition | Label | Weight |
|-----------|-------|--------|
| Visible + clicked | up (2) | `config.up_weight` (0.3) |
| Visible + no click + explicit feedback exists | **skip** | — |
| Visible + no click + no explicit feedback | down (0) | `config.down_weight` (0.1) |
| Story not in any local cache | **skip** | — |
| `impressions < min_impressions` | **skip** | — |

No bucket math. No baselines. No residuals. A click is a click, and the user demonstrably saw the card.

---

## Step 3: Merge into label builder (Phase 2 - Deferred)

**File**: `api/feedback_single_model.py`

Modify `build_single_model_feedback_labels()`:

```python
def build_single_model_feedback_labels(
    records: dict[str, FeedbackRecord],
    *,
    implicit_labels: list[ImplicitLabel] | None = None,
) -> FeedbackLabelBuildResult:
```

Logic:
1. Build explicit labels with time-decayed weight: `max(0.5, 1 − age_days / decay_days)`
2. Add `sample_weight` field to `SingleModelLabeledStory` (1.0 before decay)
3. If `implicit_labels` provided, filter out keys that have explicit feedback
4. For remaining, create `SingleModelLabeledStory` with `label` and `sample_weight` from the implicit label
5. Skip any `ImplicitLabel` where `story is None` (no cached text)
6. Append all to result

---

## Step 4: Sample weights in training pipeline (Phase 2 - Deferred)

**File**: `api/feedback_single_model.py`

Add `sample_weight: float = 1.0` and `source: str = "explicit"` to `SingleModelLabeledStory`:

```python
@dataclass(frozen=True)
class SingleModelLabeledStory:
    key: str
    story: Story
    label: int
    feedback_updated_at: float = 0.0
    sample_weight: float = 1.0   # NEW
    source: str = "explicit"      # NEW
```

Extend `build_single_model_training_matrix()` to return sample weights:

```python
def build_single_model_training_matrix(
    ...
) -> tuple[SingleModelFeatureBatch, NDArray[np.int64], NDArray[np.float64]]:
    weights = np.asarray([item.sample_weight for item in labels], dtype=np.float64)
    return batch, y, weights
```

**File**: `api/ordinal_model.py`

Modify `train_model_from_matrix()`:

```python
def train_model_from_matrix(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    config: SingleModelConfig,
    *,
    labels: list[LabeledStory] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> OrdinalThresholdModel:
```

Changes:
- Pass to `.fit()` via parameter routing: `model__sample_weight=sample_weight` (required for sklearn Pipeline)
- When `sample_weight` is provided AND `config.balance_training_labels` is True, skip `_balance_binary_matrix` — interleaving doesn't respect weights. Rely on `class_weight='balanced'` instead.
- Raw explicit safeguard: count explicit labels (`source != "implicit_ctr"`). Reject if `< 5` raw upvotes or `< 5` raw downvotes.
- Weighted effective counts: `sum(sample_weight[y >= threshold])` must still exceed `min_positive_labels` / `min_negative_labels`.

Update `train_single_model` and `train_single_model_from_embeddings` to pass weights through.

---

## Step 5: Configuration (Phase 2 - Deferred)

**`hn_rerank.toml`** — new section:

```toml
[hn_rerank.implicit]
enabled = false
up_weight = 0.3
down_weight = 0.1
min_impressions = 2
decay_days = 14
observer_deployed_at = 0.0
```

**`api/config.py`**:

```python
@dataclass(frozen=True)
class ImplicitConfig:
    enabled: bool = False
    up_weight: float = 0.3
    down_weight: float = 0.1
    min_impressions: int = 2
    decay_days: float = 14.0
    observer_deployed_at: float = 0.0
```

Add to `AppConfig` and `AppConfig.load()` following existing pattern.

Set `observer_deployed_at` to the Unix timestamp of deployment (e.g., when you first deploy the IntersectionObserver). Only impressions after this timestamp are considered clean.

---

## Step 6: Integration in `generate_html.py` (Python side) (Phase 2 - Deferred)

Around line 1618, after `feedback_records` is loaded:

```python
implicit_labels = []
if config.implicit.enabled:
    implicit_labels = derive_implicit_labels_from_db(
        feedback_records=feedback_records,
        config=config.implicit,
    )

feedback_labels = build_single_model_feedback_labels(
    feedback_records,
    implicit_labels=implicit_labels,
).labels
```

After training, log:
```python
explicit_count = sum(1 for l in feedback_labels if l.source != 'implicit_ctr')
implicit_count = sum(1 for l in feedback_labels if l.source == 'implicit_ctr')
implicit_weight = sum(l.sample_weight for l in feedback_labels if l.source == 'implicit_ctr')
logger.info(
    f"Training: {explicit_count} explicit + {implicit_count} implicit "
    f"(weighted eff {implicit_weight:.1f}) = {len(feedback_labels)} total labels"
)
```

---

## Step 7: Tests (Phase 1 Active, Phase 2 Deferred)

### Phase 1 Active Tests — `tests/test_telemetry.py`

1. **`test_url_column_exists`** — after `init_event_schema()`, the schema has a nullable `url` column
2. **`test_url_column_idempotent`** — calling `init_event_schema()` twice on the same DB doesn't crash (ALTER IF NOT EXISTS via try/except)
3. **`test_url_roundtrip`** — insert an impression with `url`, read it back unchanged
4. **`test_url_insert_without_url`** — backwards compat: old-style payload without `url` field inserts as NULL, no crash
5. **`test_click_has_url`** — click event payload includes `url` field

### Phase 2 Deferred Tests — `tests/test_implicit_feedback.py`

1. **`test_click_label`** — story with 3 impressions, 1 click → implicit `up` label
2. **`test_no_click_label`** — story with 3 impressions, 0 clicks → implicit `down` label
3. **`test_dedup_with_explicit`** — story in both explicit and implicit → explicit wins (skipped)
4. **`test_min_impressions`** — story with 1 impression → skipped
5. **`test_missing_cache`** — story not in any local cache → skipped gracefully
6. **`test_explicit_decay`** — old explicit label gets weight decayed to `max(0.5, 1 − age/decay_days)`
7. **`test_sample_weight_dims`** — training matrix returns correctly shaped weight array
8. **`test_raw_explicit_safeguard`** — fewer than 5 raw explicit labels → raises
9. **`test_weighted_count_safeguard`** — even with enough raw labels, weighted counts must clear threshold
10. **`test_fit_with_weights`** — end-to-end: model trains with sample weights, no crash
11. **`test_balance_matrix_skipped_when_weighted`** — `_balance_binary_matrix` not called when `sample_weight` is present
12. **`test_observer_cutoff`** — impressions before `observer_deployed_at` are ignored

---

## Edge Cases & Risks

| Risk | Mitigation |
|------|-----------|
| **Pipeline fit crash on `sample_weight`** | Use `model__sample_weight=sample_weight` prefix for sklearn Pipeline routing |
| **Existing contaminated data** | Filter by `server_ts >= observer_deployed_at`; old data is ignored |
| **Missing story text** | Reconstruct from `.cache/stories/` and `.cache/rss/`; skip if not found (no network fetch) |
| **No implicit labels at all** | Degrade gracefully to explicit-only training (same as `enabled = false`) |
| **Class imbalance** | `class_weight='balanced'` handles it; implicit labels are low-weight (~0.1–0.3) and cannot dominate |
| **Weight inflation** | Total implicit weight capped at ~120 × 0.3 = 36, vs ~1,199 × 1.0 = 1,199 explicit |
| **Decay** | Linear `max(0, 1 − age_days / decay_days)`. Explicit labels also decay (floor 0.5) |
| **IntersectionObserver not supported** | Falls back silently: cards never show as "visible", no impressions sent (same as `enabled = false`) |
| **Multiple tabs / rapid page loads** | Each page load gets unique `event_id` (UUID) in SQLite; no dedup needed across loads |

---

## Verification

1. **JS null check**: Before observer, inspect `public/index.html` — all 40 card impressions go to SQLite (current behavior)
2. **After observer**: Open devtools Network tab, scroll slowly — impressions POST only when cards enter viewport
3. **No regression with `enabled = false`**: `uv run pytest -q -m "not slow"` — all pass
4. **Train with `enabled = true`**: After accumulating some data, verify log line: `"Training: 1199 explicit + N implicit (weighted eff M)"`
5. **Manual**: Compare dashboard before/after with `enabled = false` — identical output
6. **Surface area and risk of data contamination is minimal** since all data is pre-qualified by `observer_deployed_at` timestamp.

---

## Open Questions (resolved)

1. ~~Position-bias correction~~ → **Not needed**. IntersectionObserver produces genuinely visible impressions; a click is a click.
2. ~~Config hash~~ → **Not needed**. Impressions are filtered by `observer_deployed_at` timestamp instead.
3. ~~Residual thresholds~~ → **Not needed**. Visible/no-click is a clean negative signal.
4. ~~Story reconstruction~~ → Local caches work; skip if missing.
5. ~~When to enable~~ → Phase 2 after ~6 months accumulation.

---

## ML Signal Roadmap

Three paths for turning telemetry into model improvement, ordered by ROI.

### Path 1: Implicit Labels (Steps 2–6 above — highest value)

The core insight: IO-qualified impressions turn every dashboard visit into labeled data.

| Signal | Label | Weight | Rationale |
|--------|-------|--------|-----------|
| Visible + clicked | up (2) | 0.3 | Click is intent, but weaker than explicit upvote |
| Visible + no click + no explicit | down (0) | 0.1 | Genuine "saw it, passed" — noisy per-item but informative in aggregate |
| Visible + has explicit feedback | **skip** | — | Explicit signal dominates |

**Why it works**: Explicit feedback yields ~20–30 labels/day. With impressions, every dashboard load produces ~15–25 additional labeled examples (visible cards minus already-voted). That's a 5–10× increase in training signal volume, even at low weights.

**Risk**: Click ≠ quality. Clickbait titles get clicked but don't satisfy. Mitigated by the 0.3 weight cap — one explicit upvote outweighs 3 clicks.

### Path 2: Aggregated CTR Features (complementary to Path 1)

Use CTR as a **feature** in the ranker rather than a label:

- **Domain CTR**: `SUM(clicks) / SUM(impressions)` per domain. Captures "I always click arxiv papers" without an explicit vote on each one.
- **Acquisition-kind CTR**: Measures whether explore/exploit/uncertainty stories are actually engaging. Feeds back into explore slot sizing.
- **Time-decayed CTR**: Recent CTR vs. historical CTR — captures evolving interests.

These are cheap to compute (SQL aggregates over `telemetry_events`) and add signal orthogonal to embeddings. They'd slot into `build_single_model_feature_batch` as additional numeric features.

### Path 3: Dwell Time (future instrumentation)

Not currently tracked, but highest-signal engagement metric. Would require:

- **JS**: Record `visibilitychange` / `blur` events with timestamps.
- **Compute**: Time-on-page between click and return.
- **Threshold**: >30s = engaged read, <5s = bounce.

This distinguishes "clicked and read the whole thing" from "clicked, saw it was junk, hit back." Much stronger signal than binary click. But needs new JS plumbing and a new telemetry event type.

### Execution Order

| Phase | When | What | Risk |
|-------|------|------|------|
| **Soak** | Now → 6 months | Do nothing. Let telemetry accumulate. Need ~2,000+ clean impressions with meaningful click variance. | — |
| **Phase 2a** | ~6 months | **Path 2**: Domain CTR as a feature. Single SQL query + one new feature column. Low risk, easy to A/B via eval harness. | Low |
| **Phase 2b** | After 2a validated | **Path 1**: Implicit labels with sample weights. Touches training pipeline, needs sample-weight plumbing, more failure modes. Big payoff: 5–10× more training data. | Medium |
| **Phase 3** | Only if click noise hurts | **Path 3**: Dwell time instrumentation. Skip unless evidence that click ≠ quality is degrading model. | Low priority |
