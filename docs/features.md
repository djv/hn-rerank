# Features

This file documents the current runtime feature surface for the feedback-trained
single model.

## Single-Model Runtime Features

The live ranking model is trained in `api/rerank.py` and consumes derived
features, not raw embeddings directly.

Current config path:

- `single_model.model_type = "svm"`
- `single_model.svm_kernel = "rbf"`
- `single_model.svm_c = 3.0`
- `single_model.svm_gamma = "scale"`

The feature vector is assembled from:

1. similarity features from the candidate embedding and user history
2. metadata features from the story itself

## Active Similarity Features

Current active similarity features from `hn_rerank.toml`:

1. `centroid_feature`
2. `pos_knn_feature`
3. `neg_knn_feature`
4. `log_points`
5. `log_comments`
6. `closest_pos`
7. `closest_neg`

So the live similarity block is currently 7 dimensions.

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

## Optional Similarity Feature Toggles

Plumbing exists for these additional toggles:

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

These are mostly experimental refinements of the same similarity signal
family.

## Metadata Features

Current metadata features are HN-centric:

- `log_points`
- `log_comments`
- `comment_ratio` when enabled
- `title_len`
- `text_len`
- `has_url`
- `is_github`
- `is_pdf`
- `comments_count`

`comments_count` uses `Story.comment_count`, not the length of the fetched
comment text list.

These features are mostly lightweight source and quality heuristics. They can
help, but they are not a substitute for better feedback coverage or exact
vote-time feature capture.

## Current Interpretation

The model is not a two-stage stack anymore. The runtime is now:

1. feature extraction
2. single-model scoring
3. slate selection

That makes feature quality and configuration much easier to reason about than
the old CE plus learned-ranker pipeline.
