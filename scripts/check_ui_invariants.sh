#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_HTML="$PROJECT_DIR/index.html"
CLUSTERS_HTML="$PROJECT_DIR/clusters.html"
INDEX_URL="file://$INDEX_HTML"
CLUSTERS_URL="file://$CLUSTERS_HTML"

TAKE_SCREENSHOTS=false
if [[ "${1:-}" == "--screenshots" ]]; then
    TAKE_SCREENSHOTS=true
fi

if [[ ! -f "$INDEX_HTML" || ! -f "$CLUSTERS_HTML" ]]; then
    echo "error: index.html and clusters.html are required"
    echo "run: uv run generate_html.py <username> --no-tldr"
    exit 1
fi

# Keep all tool caches/browser state inside the repo by default so the script is
# reproducible in sandboxed and CI-like environments.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$PROJECT_DIR/.cache/uv-tools}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_DIR/.cache}"
export HOME="${HN_RERANK_HOME:-$PROJECT_DIR}"
mkdir -p "$UV_CACHE_DIR" "$UV_TOOL_DIR" "$XDG_CACHE_HOME"

cleanup() {
    uvx rodney stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

uvx rodney stop >/dev/null 2>&1 || true
uvx rodney start >/dev/null

uvx rodney open "$INDEX_URL" >/dev/null
uvx rodney waitstable >/dev/null

total="$(uvx rodney count '.story-card')"
rss="$(uvx rodney count '.story-card.rss-story')"
hn="$(uvx rodney count '.story-card:not(.rss-story)')"
rss_badges="$(uvx rodney count '.story-card.rss-story .rss-badge')"
rss_cluster_chips="$(uvx rodney count '.story-card.rss-story .cluster-chip')"
rss_comment_links="$(uvx rodney count '.story-card.rss-story a[title="Comments"]')"
hn_comment_links="$(uvx rodney count '.story-card:not(.rss-story) a[title="Comments"]')"
clusters_nav_links="$(uvx rodney count 'a[href="clusters.html"]')"

[[ "$total" -gt 0 ]] || { echo "FAIL: no story cards"; exit 1; }
[[ $((rss + hn)) -eq "$total" ]] || {
    echo "FAIL: story totals mismatch (total=$total rss=$rss hn=$hn)"
    exit 1
}
[[ "$rss_badges" -eq "$rss" ]] || {
    echo "FAIL: RSS badge mismatch (badges=$rss_badges rss=$rss)"
    exit 1
}
[[ "$rss_cluster_chips" -eq "$rss" ]] || {
    echo "FAIL: RSS cluster chip mismatch (chips=$rss_cluster_chips rss=$rss)"
    exit 1
}
[[ "$rss_comment_links" -eq 0 ]] || {
    echo "FAIL: RSS cards should not have HN comment links (found=$rss_comment_links)"
    exit 1
}
[[ "$hn_comment_links" -eq "$hn" ]] || {
    echo "FAIL: HN comment link mismatch (links=$hn_comment_links hn=$hn)"
    exit 1
}
[[ "$clusters_nav_links" -ge 1 ]] || {
    echo "FAIL: clusters page link missing from index page"
    exit 1
}

if [[ "$TAKE_SCREENSHOTS" == "true" ]]; then
    uvx rodney screenshot "$PROJECT_DIR/demo_index.png" >/dev/null
fi

uvx rodney open "$CLUSTERS_URL" >/dev/null
uvx rodney waitstable >/dev/null

clusters_title="$(uvx rodney title)"
cluster_cards="$(uvx rodney count '.cluster-card')"
cluster_rows="$(uvx rodney count '.cluster-card li')"
cluster_row_links="$(uvx rodney count '.cluster-card li a')"

[[ "$clusters_title" == "Interest Clusters | "* ]] || {
    echo "FAIL: unexpected clusters title: $clusters_title"
    exit 1
}
[[ "$cluster_cards" -gt 0 ]] || { echo "FAIL: no cluster cards"; exit 1; }
[[ "$cluster_rows" -ge "$cluster_cards" ]] || {
    echo "FAIL: cluster rows < cluster cards (rows=$cluster_rows cards=$cluster_cards)"
    exit 1
}
[[ "$cluster_row_links" -ge "$cluster_cards" ]] || {
    echo "FAIL: cluster row links < cluster cards (links=$cluster_row_links cards=$cluster_cards)"
    exit 1
}

if [[ "$TAKE_SCREENSHOTS" == "true" ]]; then
    uvx rodney screenshot "$PROJECT_DIR/demo_clusters.png" >/dev/null
fi

echo "INVARIANT_TOTAL_STORIES=$total"
echo "INVARIANT_RSS_STORIES=$rss"
echo "INVARIANT_HN_STORIES=$hn"
echo "INVARIANT_CLUSTER_CARDS=$cluster_cards"
echo "INVARIANT_CLUSTER_ROWS=$cluster_rows"
echo "INVARIANTS_STATUS=PASS"
