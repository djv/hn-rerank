# UI Verification Demo

This is the current lightweight browser-verification note for dashboard changes.
Historical captured outputs were removed because they drift quickly; regenerate
the dashboard and run the checks below when UI behavior changes.

## Generate Artifacts

```bash
uv run generate_html.py <your-hn-username> --no-tldr
```

Expected generated files:
- `public/index.html`
- `public/clusters.html`

## Static Invariants

```bash
./scripts/check_ui_invariants.sh
```

This checks the generated HTML structure, including story-card counts, RSS badge
presence, HN discussion links, cluster links, and cluster page consistency.

## Showboat Demos

```bash
./scripts/verify_showboat_demos.sh
```

Use this when the change is browser-visible. Update this file when the expected
workflow changes.

## Rodney Smoke Check

```bash
uvx rodney start
uvx rodney open file:///home/dev/hn_rerank/public/index.html
uvx rodney waitstable
uvx rodney title
uvx rodney count '.story-card'
uvx rodney count '.story-card.rss-story'
uvx rodney count '.story-card.rss-story .rss-badge'
uvx rodney count '.story-card .cluster-chip'
uvx rodney open file:///home/dev/hn_rerank/public/clusters.html
uvx rodney waitstable
uvx rodney count '.cluster-card'
uvx rodney count '.cluster-card li'
uvx rodney stop
```

The exact counts depend on `count`, RSS availability, and the current candidate
pool. Treat zero story cards, zero cluster cards, missing RSS badges for RSS
cards, or cluster rows fewer than cluster cards as failures.
