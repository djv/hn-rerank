# Add `Discussion-rich` and `High-engagement` surfaces

## Goal
Add two new post-MMR surfaces that mirror the existing `is_uncertain` and `is_novel`
pattern: a boolean flag on `RankedStory`, a post-rank surfacing pass that pulls
matching items from the `remaining` pool, a templated badge in the dashboard, and
a 5-item budget per surface.

## Why these two
- **Discussion-rich** (comment-count proxy): signals stories worth reading for
  the *comments*, not just the article. Complements `is_novel` (positive
  personalization signal with weak crowd signal) by surfacing stories with active
  debate even when the model score is moderate.
- **High-engagement** (HN-score proxy): pure social proof. Useful during the
  cold-start phase (low feedback count) and as a tiebreaker when the reranker
  hasn't yet calibrated to Daniel's tastes.

We deliberately *skipped* `Underrated`, `Old-but-gold`, `Topic-drift`,
`Duplicate-warning`, and `High-confidence-contrarian` because:
- `Underrated` / `High-confidence-contrarian` overlap heavily with the natural
  dashboard ordering (high model score is already surfaced).
- `Old-but-gold` is partially redundant with `is_novel` (both flag positive
  signal in low-signal regions).
- `Topic-drift` is a strictly weaker `is_novel`.
- `Duplicate-warning` is UX-cost-heavy (annoyance) for low information gain.

## Surfaces

### 1. `is_discussion_rich` — `comment_count` in top decile
- **Definition**: `comment_count >= np.percentile(cand_comment_counts, 90)`
- **Budget**: 5 stories per regeneration (matches `is_novel`)
- **Title attribute**: "Many comments on this story (top 10%)"
- **Badge text**: `💬 Talk-worthy`
- **CSS class**: `.discussion-rich-badge` (warm amber, hsl(35, 60%, 25%) bg)

### 2. `is_high_engagement` — `story.score` in top decile
- **Definition**: `story.score >= np.percentile(cand_scores, 90)`
- **Budget**: 5 stories per regeneration
- **Title attribute**: "Crowd favorite (top 10% by HN points)"
- **Badge text**: `🔥 Trending`
- **CSS class**: `.high-engagement-badge` (warm red-orange, hsl(15, 60%, 25%) bg)

## Data path

`cand_scores` and `cand_comment_counts` are **already computed** in
`rank_stories` at lines 830 and 832 — they get fed into `_augment_features`
as `cand_scores` and `cand_comment_counts`. The arrays are also in scope when
we build the `RankedStory` list at lines 873–884. We just need to:

1. Compute the 90th-percentile threshold once after the arrays exist.
2. Set `is_discussion_rich` / `is_high_engagement` on each `RankedStory`
   when probabilities are available (the `try` block at line 868).
3. Add two new pass-through branches in `run_pipeline` after the existing
   `is_novel` surfacing.

The fallback path at line 890 (no feedback → sort by frontpage) should also
set the flags using the same percentile thresholds; otherwise the badges
disappear during cold start when they're arguably most useful.

## Changes

### `pipeline.py`

**1. `RankedStory` dataclass (line 93)**
- Add `is_discussion_rich: bool = False`
- Add `is_high_engagement: bool = False`

**2. `rank_stories` (around line 820)**
- After `cand_scores` and `cand_comment_counts` are built (lines 830, 832),
  compute thresholds:
  ```python
  discussion_rich_threshold = (
      np.percentile(cand_comment_counts, 90) if len(cand_comment_counts) else 0
  )
  high_engagement_threshold = (
      np.percentile(cand_scores, 90) if len(cand_scores) else 0
  )
  ```
- In the `RankedStory(...)` constructor call (line 874), pass:
  ```python
  is_discussion_rich=int(c.comment_count or 0) >= discussion_rich_threshold
      and (c.comment_count or 0) > 0,
  is_high_engagement=int(c.score) >= high_engagement_threshold,
  ```
- In the no-feedback fallback (line 890–891), compute the same thresholds
  *outside* the `try` (so they exist on cold start) and pass them through.

  Concretely: hoist `discussion_rich_threshold` and `high_engagement_threshold`
  to be computed unconditionally *before* the `try` block, using the arrays
  built in the `try`. This requires moving `cand_scores` /
  `cand_comment_counts` construction outside the `try`, OR computing the
  thresholds twice (once inside the `try`, once in the fallback). Cheapest:
  compute thresholds twice — they're O(n) on 350 elements, ~30µs total.

**3. `run_pipeline` (after line 1091, the `is_novel` block)**
Add two analogous blocks:
```python
# Surface up to 5 discussion-rich stories on top of the normal count
discussion_pool = [
    r for r in ranked
    if r.story.id not in selected_ids and r.is_discussion_rich
]
if discussion_pool:
    discussion_pool.sort(key=lambda r: r.story.comment_count or 0, reverse=True)
    discussion_items = [
        replace(item, is_discussion_rich=True)
        for item in discussion_pool[:5]
    ]
    final.extend(discussion_items)
    selected_ids |= {item.story.id for item in discussion_items}

# Surface up to 5 high-engagement stories on top of the normal count
engagement_pool = [
    r for r in ranked
    if r.story.id not in selected_ids and r.is_high_engagement
]
if engagement_pool:
    engagement_pool.sort(key=lambda r: r.story.score, reverse=True)
    engagement_items = [
        replace(item, is_high_engagement=True)
        for item in engagement_pool[:5]
    ]
    final.extend(engagement_items)
    selected_ids |= {item.story.id for item in engagement_items}
```

The `selected_ids` set is updated *inside* each block so a story can't be
surfaced by two passes (e.g. novel + discussion-rich is exclusive).

**4. Logging**
- Add a single end-of-stage line near the others:
  ```python
  logging.info(f"Surfaced {len(final) - limit} extras (uncertain/novel/discussion/engagement)")
  ```
  Computed *after* the new blocks; cheap and gives the user a single number
  to correlate with the dashboard.

### `templates/index.html`

**1. CSS (after line 186, the `.novel-badge` block)**
Add two parallel rules:
```css
.discussion-rich-badge {
  display: inline-block;
  padding: 0.05rem 0.35rem;
  border-radius: 3px;
  font-weight: 600;
  font-size: 0.7rem;
  background: hsl(35, 60%, 25%); /* Warm amber for active discussion */
  color: hsl(35, 80%, 90%);
  border: 1px solid hsl(35, 60%, 35%);
  line-height: 1.2;
}

.high-engagement-badge {
  display: inline-block;
  padding: 0.05rem 0.35rem;
  border-radius: 3px;
  font-weight: 600;
  font-size: 0.7rem;
  background: hsl(15, 60%, 25%); /* Warm red-orange for crowd favorite */
  color: hsl(15, 80%, 90%);
  border: 1px solid hsl(15, 60%, 35%);
  line-height: 1.2;
}
```

**2. Template (after line 327, the `is_novel` block)**
```html
{% if item.is_discussion_rich %}
<span class="discussion-rich-badge" title="Many comments on this story (top 10%)">💬 Talk-worthy</span>
{% endif %}
{% if item.is_high_engagement %}
<span class="high-engagement-badge" title="Crowd favorite (top 10% by HN points)">🔥 Trending</span>
{% endif %}
```

### `tests/test_pipeline.py`

Add a focused test for the new flag computation:
```python
def test_rank_sets_discussion_rich_and_high_engagement(tmp_path, monkeypatch):
    """Top-10% comment-count stories get is_discussion_rich; top-10% score gets is_high_engagement."""
    from pipeline import Config, rank_stories
    from database import Database

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    config = Config(
        db_path=str(db_file),
        output=str(tmp_path / "index.html"),
        server_port=0,
    )
    # Build 20 candidates: comments 0..19, scores 100, 200, 300, ...
    candidates = []
    now = int(time.time())
    for i in range(20):
        candidates.append(Story(
            id=1000 + i,
            title=f"Story {i}",
            url=None,
            score=100 * (i + 1),
            time=now,
            text_content="body",
            source="hn",
            comment_count=i,
        ))

    ranked = rank_stories(candidates, np.zeros((20, 4), dtype=np.float32), db, config, embedder=None)
    # Top 10% of comments = 2 highest; top 10% of scores = 2 highest
    rich = [r for r in ranked if r.is_discussion_rich]
    trending = [r for r in ranked if r.is_high_engagement]
    assert len(rich) == 2
    assert len(trending) == 2
    assert all(r.story.comment_count >= 18 for r in rich)  # 18, 19
    assert all(r.story.score >= 1800 for r in trending)  # 1800, 1900, 2000 - 90th pct of 100..2000 step 100
```

This requires `Story` and `time` to be importable. Both are already used in
existing tests — confirm before writing.

## Edge cases & invariants
- **Empty candidate set**: percentile on empty array would crash — guard with
  `if len(...)` checks (already in the spec above).
- **All-zero `comment_count`**: `is_discussion_rich` requires `> 0` so we
  don't badge stories with no discussion just because the field is 0.
- **Story already in `final` from MMR**: `selected_ids` filter prevents
  double-surfacing.
- **Cold start (no feedback)**: fallback path at line 890 must set the flags
  too, or the badges disappear exactly when they're most useful. Plan
  addresses this by computing thresholds in the fallback.
- **Percentile choice (90)**: tunable but kept hardcoded for now; matches
  the intent of "small minority, high signal." If we want a knob later,
  it goes in `config.model.*`.

## Out of scope
- No new config fields.
- No changes to `database.py` (no new tables/columns).
- No changes to `server.py`.
- No changes to `eval.py` (these are surface flags, not features for the
  reranker).

## Verification
1. `uv run ruff check .`
2. `uv run pytest tests/ -m "not slow"` — all green, < 30s
3. `uv run python generate.py` — confirm log line shows non-zero extras count
4. Visual check of `public/index.html` — confirm badges render with correct
   color and that a story with both `is_novel` and `is_discussion_rich` shows
   both badges (not exclusive on display; only on surfacing position)
