# Cross-Encoder

This is legacy reference material for the removed cross-encoder stage.
The live runtime no longer uses CE reranking.

## Purpose

The CE stage is the second-stage semantic reranker inside `api/rerank.py`.

It refines the first-stage HN ordering using an ONNX cross-encoder model.

## Current Config

From `hn_rerank.toml`:

- `cross_encoder.enabled = true`
- `cross_encoder.top_n = 200`
- `cross_encoder.weight = 0.06953`
- `cross_encoder.model_dir = "onnx_ce_model"` (default from config code)

## Scope

Current behavior:

- CE only applies to HN stories
- external stories are not CE-scored
- external stories remain in the downstream ranked pool

This is deliberate. If CE replaced the full pool with only the CE slice,
external quota enforcement in final selection would break.

## Query Construction

Current query construction per interest centroid:

Primary path:

- cluster name
- cluster keywords

These are joined as:

- `name: keywords`

Current behavior intentionally does **not** append the representative story
title when cluster naming data exists.

## Fallback Query Path

If cluster naming data is missing:

- collect titles from the top 10 positive stories in the cluster
- join them with `" | "`
- use that title bundle as the CE query text

Current fallback does **not** use representative story body text.

## Candidate Side

For each CE-scored candidate:

- candidate text is `story.text_content`
- CE scores the candidate against each cluster query
- the max CE score across queries is used

## Blending

For the CE-scored HN slice:

1. first-stage `hybrid_score` values are normalized within the slice
2. CE logits are normalized within the slice
3. the two are blended using `cross_encoder.weight`
4. the blended score is mapped back into the original `hybrid_score` range
5. `hybrid_score` is updated in-place for downstream sorting and display

This is important: the CE stage does not just reorder indices. It also writes
the blended score back into `hybrid_score`.

## Downstream Pool After CE

Current behavior after CE:

- HN pool becomes the reranked CE-scored HN slice
- external stories are appended after that, in ranked order

This means:

- all downstream HN selection happens from the CE-scored HN set
- external quota logic still has access to non-HN stories

## Debug Interpretation

If a displayed story has `cross_encoder_score = 0.0`, that usually means:

- it was not part of the CE-scored HN slice, or
- it is an external story, which currently bypasses CE

This is expected behavior, not necessarily a serialization bug.

## Runtime Cost

CE is one of the main runtime bottlenecks:

- it loops over top-HN candidates
- it runs ONNX inference for each candidate
- large evals with many held-out stories can get stuck here

That is the main reason bigger offline evaluation runs slow down sharply.
