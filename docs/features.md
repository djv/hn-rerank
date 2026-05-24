# Features

This file documents the current first-stage and learned-ranker feature surfaces.

## First-Stage Model Features

The first-stage model lives in `api/rerank.py`.

Current config:

- `classifier.scoring_mode = "pairwise_logistic"`
- `classifier.feature_mode = "bottleneck"`

That means the model uses derived features, not raw embedding dimensions.

## Active First-Stage Features

Current active first-stage features from `hn_rerank.toml`:

1. `centroid_feature`
2. `pos_knn_feature`
3. `neg_knn_feature`
4. `log_points`
5. `log_comments`
6. `closest_pos`
7. `closest_neg`

So the live first-stage vector is currently 7 dimensions.

## What Each Feature Means

### `centroid_feature`

- max cosine similarity from the candidate to any positive interest centroid

### `pos_knn_feature`

- median similarity to the top `k_feat` nearest positive examples

### `neg_knn_feature`

- median similarity to the top `k_feat` nearest negative examples

### `log_points`

- `log1p(story.score)`

### `log_comments`

- `log1p(story.comment_count)`

### `closest_pos`

- max similarity to any single positive example

### `closest_neg`

- max similarity to any single negative example

## Optional First-Stage Feature Toggles

Plumbing exists for these optional toggles:

- `use_closest_centroid_feature`
- `use_knn_pos_n1_feature`
- `use_knn_pos_n3_feature`
- `use_knn_pos_n5_feature`
- `use_knn_pos_n10_feature`
- `use_knn_neg_n1_feature`
- `use_knn_neg_n3_feature`
- `use_knn_neg_n5_feature`
- `use_knn_neg_n10_feature`
- `use_comment_ratio_feature`

## Duplicate / Near-Duplicate Features

Current code has some duplicate semantics:

- `closest_centroid` is the same value as `centroid_feature`
- `knn_pos_n1` is effectively a top-1 positive similarity signal similar to
  `closest_pos`
- `knn_neg_n1` is similar to `closest_neg`

Turning these on can improve metrics by changing model weighting, but that is
not the same as adding new information.

## Metadata Features

Current metadata features are HN-centric:

- `log_points`
- `log_comments`
- optional `comment_ratio`

These tend to favor HN stories over external stories because externals often
lack comparable points/comment metadata.

## Learned Ranker Features

The learned final reranker lives in `api/learned_ranker.py`.

Current feature names:

1. `semantic_score`
2. `hybrid_score`
3. `max_cluster_score`
4. `knn_score`
5. `max_sim_score`
6. `cross_encoder_score`
7. `log_points`
8. `log_comments`

These are built from:

- the runtime ranking outputs
- stored story metadata from dashboard feedback

## First-Stage vs Learned-Ranker Feature Difference

First-stage model:

- builds similarity features directly from embeddings and negative/positive
  history

Learned final ranker:

- consumes the outputs of the runtime ranker plus raw popularity metadata

This difference matters because the learned ranker is not an independent model
from raw text. It is learning on top of an already-ranked representation.
