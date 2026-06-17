# Evaluation Metrics Guide

Based on deep-dives into the evaluation pipeline and cross-validation behavior, the following metrics should be used to judge model performance for the Hacker News reranking dashboard.

## Primary Metrics

These three metrics provide the most robust, actionable signal for dashboard recommendation quality, as they map directly to global model health and the actual user experience.

### 1. Median Rank (The Global Benchmark)
- **What it is:** The exact middle rank of all true positive test stories across the candidate pool.
- **Why it matters:** It proves the global health of the model without being skewed by metric saturation (like NDCG) or ruined by extreme outliers (like Mean Rank). Lower is better. 

### 2. Precision@50 (The "Dashboard Density")
- **What it is:** Out of the top 50 items the model recommends, the percentage that are actually true positives.
- **Why it matters:** This directly mimics the real-world experience of loading the first page of the dashboard. It tells you exactly how clean the recommendations are (e.g., a Precision@50 of 90% means 45 of the 50 slots are filled with highly relevant content). Higher is better.

### 3. Recall@50 (The "Dashboard Coverage")
- **What it is:** Out of *all* the true positive stories sitting in the candidate pool, the percentage that successfully made it into the top 50 slots.
- **Why it matters:** It tells you how much good content you are missing. If you have 250 great stories in the pool, Recall@50 tells you what fraction you actually get to see before you stop scrolling. Higher is better.

## Legacy & Diagnostic Metrics

### NDCG@30 / Hit@30
- **Status:** Legacy / Context-dependent.
- **Context:** These metrics mathematically saturate when evaluating a large test set (e.g. 200+ positives) against a standard candidate pool. `Hit@30` will almost always hit 1.000, and `NDCG@30` easily hits 0.950+ because the model only needs to find the 30 "easiest" positives out of 200 to achieve a perfect score. 
- **Usage:** Useful for quick relative checks or when `max_test` is strictly constrained, but `Median Rank` is a much more honest assessment of the model's global sorting ability.

### Mean Rank
- **Status:** Diagnostic.
- **Context:** Highly sensitive to outliers. If a model brilliantly ranks 100 true positives in the top 100, but completely misses 10 outlier stories and drops them to rank 700, the `Mean Rank` gets dragged down massively. 
- **Usage:** Good for checking if a new model introduces catastrophic blind spots, but `Median Rank` is much more stable for day-to-day tuning.
