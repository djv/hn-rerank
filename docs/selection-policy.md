# Selection Policy

This file documents how the final slate is chosen after ranking.

## Entry Point

- `generate_html.py::select_ranked_results`

Selection happens after:

1. single-model ranking

So the input list is already ordered by the current active final score.

## What Selection Does

Selection applies slate policy. It does not learn and it does not create new
scores.

Current responsibilities:

- split ranked results into external vs HN
- reserve a fixed external quota
- enforce per-source diversity on external picks
- choose the final `count`
- re-sort the chosen slate by active final score

## External Quota

Current target:

- `desired_external = round(count * 0.2) + 5`

With `count = 40`, the target is 13 externals when available.

Quota is clamped by reality:

- if not enough externals are available, it uses fewer
- if not enough HN stories are available, it increases externals as needed

## External Diversity

External stories are selected with a relaxed source-cap pass:

1. max `2` per source
2. if quota is not filled, relax to `3`
3. if still not filled, relax fully

This is only for external stories.

HN selection is simpler:

- just take the top ranked HN items needed to fill the remaining slots

## Final Sort Inside Selection

After combining selected external and selected HN stories, the slate is sorted
by active final score:

- `hybrid_score`

That means quota affects membership, but final within-slate order is still
score-driven.

## Interaction With the Runtime Model

The selection stage does not rescore stories. It only chooses from the
single-model ranked pool and preserves the external quota policy.

## Interaction With Dupe Filtering

Important current limitation:

- HN dupe filtering happens after selection
- if a selected HN story is removed as a dupe, selection is not rerun
- there is no refill from the remaining ranked pool

Consequences:

- final page can have fewer than `count` cards
- selected source mix can differ from rendered source mix

## Practical Debugging Rule

If `scores_debug.json` shows the right source mix but `index.html` shows fewer
cards or fewer HN stories, suspect post-selection dupe filtering first.
