# Title-Embedding Duplicate Detection

## Problem

Two HN submissions about the same news event from different outlets (e.g., Bloomberg
vs TechCrunch covering "Google paying SpaceX $920M/month") with different URLs and
different exact titles are not caught by existing URL-normalization or exact-title
dedup — the only dedup mechanisms currently active.

Example:
- `48416941` → bloomberg.com — "Google Buying Computing from SpaceX in $920M-a-Month Deal"
- `48423990` → techcrunch.com — "Google will pay SpaceX $920M per month for compute"

The user has upvoted one, yet the other still appears in their ranked dashboard.

## Approach

Use **title embedding cosine similarity** to detect same-story-from-different-outlet
duplicates. The project already has an ONNX embedding model (`get_embeddings()` in
`api/rerank.py:823`) used for ranking. We reuse it on **titles only** (not full
`text_content`) — titles are short and entity-dense, so they naturally cluster same
news across outlets.

## Changes

### 1. `api/config.py` — `AppConfig`

Add a field:

```python
duplicate_title_similarity_threshold: float = 0.80
```

Read from `hn_rerank.toml` root section under the key
`duplicate_title_similarity_threshold`.

### 2. `generate_html.py` — `filter_top_ranked_hn_dupes()`

Currently a no-op at line 97. Enhance to:

- Accept `pos_stories: list[Story]` parameter (already available at call site
  line 1431).
- Compute title embeddings for all `pos_stories` once (list of title strings).
- For each selected `RankResult` in `ranked`, look up the candidate `Story` from
  `cands`, compute its title embedding, compute cosine similarity against all
  pos_story title embeddings.
- If the maximum cosine similarity ≥ `config.duplicate_title_similarity_threshold`
  AND the normalized URL differs from the matched pos_story's URL → filter it out.
- Log all filtered items (story id + title + matched pos_story id + title).

#### Signature (updated)

```python
async def filter_top_ranked_hn_dupes(
    ranked: list[RankResult],
    cands: list[Story],
    exclude_ids: set[int],
    count: int,
    pos_stories: list[Story],
    config: AppConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[RankResult]:
```

#### Call site (line 1820)

Pass `pos_stories` and `config`.

#### Embedding batch strategy

- Compute pos_story title embeddings once: `get_embeddings([s.title for s in pos_stories])`.
- Compute candidate title embeddings for all selected results: batch once, not per-item.
- Compute cosine similarity matrix using `sklearn.metrics.pairwise.cosine_similarity`
  (already imported/used elsewhere in the codebase).

### 3. `hn_rerank.toml`

```toml
duplicate_title_similarity_threshold = 0.80  # 0-1; drop candidates whose title embedding
                                             # cosine sim to an already-upvoted story
                                             # exceeds this (same-news diff-source dedup)
```

### 4. Tests

Add test cases to `tests/`:

- Two stories with same news, different outlets → filtered out
- Two stories about same company, different events → kept
- Empty pos_stories → no filtering
- Single pos_story → dedup works correctly
- Story already excluded by ID → not affected by this logic

## Edge cases

| Case | Behavior |
|------|----------|
| Same URL, different titles | Already caught by URL dedup earlier |
| Same source re-posting | URL dedup catches it |
| Two different stories about same company | Threshold at 0.80 is conservative — requires entity+number overlap |
| Story with empty/None title | Skip (no embedding), keep it |
| pos_stories has 0 or 1 items | Skip dedup, no meaningful comparison |
| Candidate is the pos_story itself (same id, different URL? shouldn't happen) | `exclude_ids` already filters these in the candidate phase |

## Rejected alternatives

1. **Full-text embedding sim** (`max_sim_score`) — already used as a ranking signal.
   High text sim means "similar to things I like", which is desirable. Using it as dedup
   would suppress genuinely good recommendations.
2. **Token Jaccard similarity** — too low for this example (~0.27). Sentence embeddings
   capture semantic structure ("buying computing" ≈ "pay for compute").
3. **LLM-based dedup** — too expensive, no need for an LLM call per candidate.
