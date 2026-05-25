# Evaluation

This file explains what the repo's evaluation tooling actually measures.

## Main Script

The current offline evaluator is:

- `evaluate_quality.py`

## What `evaluate_quality.py` Measures

The evaluator is an offline benchmark, not a live dashboard metric.

Default flow:

1. load a user's HN signals
2. build positives from upvotes / favorites
3. optionally build negatives from hidden stories when `--classifier` is used
4. split positives into train vs test
5. fetch a candidate pool
6. inject held-out positive test stories if they are missing from the candidate
   pool
7. rank the pool with the current runtime stack
8. compute ranking metrics for the held-out positives

So the script measures:

- how well the ranking pipeline surfaces held-out stories the user previously
  liked

It does not directly measure:

- live click quality
- dashboard feedback quality
- actual production refresh latency
- exact final HTML behavior unless `--final-list` is used

## Important Caveat: Held-Out Positives Are Injected

`evaluate_quality.py` appends test stories into the candidate pool if they were
not fetched naturally.

That means the eval is primarily a ranking benchmark, not a full end-to-end
retrieval benchmark.

Implication:

- good numbers mean "if the relevant story is in the pool, the ranking can find
  it"
- they do not prove retrieval discovered it unaided

## `--classifier`

This flag matters for current serious evals.

With `--classifier`:

- hidden stories are loaded as negatives
- the runtime first-stage model usually has enough positives and negatives to
  use the trained classifier path

Without `--classifier`:

- negatives are typically absent
- the trained model often becomes ineligible
- eval can fall back to a weaker positive-only ranking path

For current pipeline checks, use `--classifier`.

## `--age-matched`

This makes candidate fetch relative to the test-story time basis.

Current implementation:

- uses the newest held-out test-story timestamp as the fetch anchor

This is better than ignoring time entirely, but it is still a coarse proxy for
production recency conditions.

## `--cache-only`

This disables fresh network candidate discovery and uses cached data.

What it is good for:

- faster, repeatable comparisons
- experiments on scoring changes

What it is not:

- a perfect proxy for live mixed-source candidate retrieval

## `--final-list`

Default eval ranks the raw output of `rank_stories`.

With `--final-list`:

- the evaluator applies final display-list policy
- this includes final slate selection rather than raw ranked order

Use it when you want to measure:

- what the final displayed list policy does

Use the default mode when you want to measure:

- raw ranking quality

## Metrics to Trust

The most useful current metrics are:

- `NDCG@k`
- `Precision@k`
- `Recall@k`

Metrics to interpret carefully:

- `MRR`

Why:

- `MRR = 1.0` only means the first relevant item is rank 1
- with many relevant items in the pool, that can look better than the full list
  quality really is

## Current Runtime Bottleneck

On larger eval runs, the slow stages are embedding and feature recomputation.
There is no cross-encoder stage in the live runtime anymore.

This is why "use all available training data" style runs can still take a long
time even after the CE path was removed.

## Common Commands

Current serious cached holdout check:

```bash
uv run python evaluate_quality.py pure_coder --classifier --cache-only --age-matched --candidates 100
```

Evaluate final display-list policy:

```bash
uv run python evaluate_quality.py pure_coder --classifier --cache-only --age-matched --candidates 100 --final-list --count 40
```

Cross-validation:

```bash
uv run python evaluate_quality.py pure_coder --classifier --cache-only --age-matched --candidates 100 --cv 5
```

Model sweeps for the live single-model path:

```bash
uv run python evaluate_quality.py pure_coder --classifier --cache-only --age-matched --final-list --count 40 --model-type svm --svm-kernel rbf --svm-c 3.0 --svm-gamma scale --use-new-features
```

## Current Limitations

- held-out positives are injected
- final HTML behavior can still differ from eval if post-selection dupe
  filtering removes cards
- cached evaluation is easier than live production behavior
- the feedback replay is still approximate, not exact vote-time reconstruction
