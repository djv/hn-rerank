# Demo: Rodney Manual Feature Checks for HN Rerank

*2026-02-11T15:43:06Z*

This document performs browser-level feature checks using Rodney, inspired by Simon Willison's Showboat/Rodney workflow.

Coverage:
- Dashboard content counts (total stories + RSS stories)
- RSS-specific rendering behavior (RSS badge + no HN comments icon)
- Cluster labels on RSS cards
- Navigation from index to clusters page
- Basic accessibility tree queries
- Screenshots for visual proof

```bash
export HOME=/home/dev/hn_rerank UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools XDG_CACHE_HOME=/home/dev/hn_rerank/.cache; uvx rodney start
```

```output
Chrome started (PID 3344887)
Debug URL: ws://127.0.0.1:41373/devtools/browser/11ccd600-bd33-4463-98fe-834e9141bac2
```

```bash
export HOME=/home/dev/hn_rerank UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools XDG_CACHE_HOME=/home/dev/hn_rerank/.cache; uvx rodney open file:///home/dev/hn_rerank/index.html && uvx rodney waitstable && uvx rodney title
```

```output
HN Rerank | pure_coder
DOM stable
HN Rerank | pure_coder
```

```bash
export HOME=/home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
export UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools
export XDG_CACHE_HOME=/home/dev/hn_rerank/.cache
total=$(uvx rodney count '.story-card')
rss=$(uvx rodney count '.story-card.rss-story')
hn=$(uvx rodney count '.story-card:not(.rss-story)')
echo TOTAL_STORIES=$total
echo RSS_STORIES=$rss
echo HN_STORIES=$hn

```

```output
TOTAL_STORIES=30
RSS_STORIES=10
HN_STORIES=20
```

```bash
export HOME=/home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
export UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools
export XDG_CACHE_HOME=/home/dev/hn_rerank/.cache
echo RSS_BADGES=$(uvx rodney count '.story-card.rss-story .rss-badge')
echo RSS_CLUSTER_CHIPS=$(uvx rodney count '.story-card.rss-story .cluster-chip')
echo RSS_COMMENTS_LINKS=$(uvx rodney count '.story-card.rss-story a[title="Comments"]')
echo HN_COMMENTS_LINKS=$(uvx rodney count '.story-card:not(.rss-story) a[title="Comments"]')

```

```output
RSS_BADGES=10
RSS_CLUSTER_CHIPS=10
RSS_COMMENTS_LINKS=0
HN_COMMENTS_LINKS=20
```

```bash
export HOME=/home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
export UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools
export XDG_CACHE_HOME=/home/dev/hn_rerank/.cache
uvx rodney open file:///home/dev/hn_rerank/index.html
uvx rodney click 'a[href="clusters.html"]'
uvx rodney waitstable
echo CLUSTERS_TITLE=$(uvx rodney title)
echo CLUSTER_CARDS=$(uvx rodney count '.cluster-card')
echo CLUSTER_STORIES=$(uvx rodney count '.story-item')

```

```output
HN Rerank | pure_coder
Clicked
DOM stable
CLUSTERS_TITLE=Interest Clusters | pure_coder
CLUSTER_CARDS=60
CLUSTER_STORIES=0
```

```bash
export HOME=/home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
export UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools
export XDG_CACHE_HOME=/home/dev/hn_rerank/.cache
uvx rodney open file:///home/dev/hn_rerank/index.html
uvx rodney waitstable
echo "AX_HEADINGS:" 
uvx rodney ax-find --role heading
echo "\nAX_FIRST_STORY_LINK:" 
uvx rodney ax-node '.story-card:first-of-type h2 a'

```

```output
HN Rerank | pure_coder
DOM stable
AX_HEADINGS:
[heading] "HN Rerank" backendNodeId=6103 (level=1)
[heading] "Show HN: Algorithmically finding the longest line of sight on Earth 💬" backendNodeId=6137 (level=2)
[heading] "The Waymo World Model 💬" backendNodeId=6167 (level=2)
[heading] "Experts Have World Models. LLMs Have Word Models 💬" backendNodeId=6197 (level=2)
[heading] "Orchestrate teams of Claude Code sessions 💬" backendNodeId=6227 (level=2)
[heading] "AI fatigue is real and nobody talks about it 💬" backendNodeId=6257 (level=2)
[heading] "Guinea worm on track to be 2nd eradicated human disease; only 10 cases in 2025 💬" backendNodeId=6287 (level=2)
[heading] "Show HN: Showboat and Rodney, so agents can demo what they've built 💬" backendNodeId=6314 (level=2)
[heading] "I am happier writing code by hand 💬" backendNodeId=6344 (level=2)
[heading] "AI Doesn't Reduce Work–It Intensifies It 💬" backendNodeId=6374 (level=2)
[heading] "Ask HN: Ideas for small ways to make the world a better place 💬" backendNodeId=6404 (level=2)
[heading] "Invention of DNA \"page numbers\" opens up possibilities for the bioeconomy 💬" backendNodeId=6434 (level=2)
[heading] "Eight more months of agents 💬" backendNodeId=6464 (level=2)
[heading] "Psychometric Jailbreaks Reveal Internal Conflict in Frontier Models 💬" backendNodeId=6494 (level=2)
[heading] "Show HN: Rowboat – AI coworker that turns your work into a knowledge graph (OSS) 💬" backendNodeId=6521 (level=2)
[heading] "FORTH?\u00a0Really!? 💬" backendNodeId=6551 (level=2)
[heading] "Claude Composer 💬" backendNodeId=6578 (level=2)
[heading] "Curating a Show on My Ineffable Mother, Ursula K. Le Guin 💬" backendNodeId=6608 (level=2)
[heading] "Study: Older Cannabis Users Have Larger Brains, Better Cognition 💬" backendNodeId=6638 (level=2)
[heading] "forecourt networking" backendNodeId=6671 (level=2)
[heading] "Pluralistic: Luxury Kafka (06 Feb 2026)" backendNodeId=6701 (level=2)
[heading] "Study Finds Obvious Truth Everybody Knows" backendNodeId=6731 (level=2)
[heading] "QuitGPT – OpenAI Execs Are Trump's Biggest Donors 💬" backendNodeId=6758 (level=2)
[heading] "Pluralistic: The Nuremberg Caucus (10 Feb 2026)" backendNodeId=6791 (level=2)
[heading] "Fibonacci numbers and time-space tradeoffs" backendNodeId=6821 (level=2)
[heading] "Amazon plunge continues $1T wipeout as AI bubble fears ignite sell-off 💬" backendNodeId=6848 (level=2)
[heading] "CI In a Box" backendNodeId=6881 (level=2)
[heading] "Book Review: Me vs Brain - An Overthinker’s Guide to Life by Hayley Morris ★★★★☆" backendNodeId=6911 (level=2)
[heading] "How to Get New Ideas" backendNodeId=6941 (level=2)
[heading] "Kākāpō mug by Karen James" backendNodeId=6971 (level=2)
[heading] "Package Manager Podcast Episodes" backendNodeId=7001 (level=2)
\nAX_FIRST_STORY_LINK:
role: link
name: Show HN: Algorithmically finding the longest line of sight on Earth
focusable: true
url: https://alltheviews.world/
```

```bash {image}
export HOME=/home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
export UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools
export XDG_CACHE_HOME=/home/dev/hn_rerank/.cache
uvx rodney open file:///home/dev/hn_rerank/index.html
uvx rodney waitstable
uvx rodney screenshot demo_index.png
echo demo_index.png

```

![37f68e0d-2026-02-11](37f68e0d-2026-02-11.png)

```bash {image}
export HOME=/home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
export UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools
export XDG_CACHE_HOME=/home/dev/hn_rerank/.cache
uvx rodney open file:///home/dev/hn_rerank/clusters.html
uvx rodney waitstable
uvx rodney screenshot demo_clusters.png
echo demo_clusters.png

```

![dcc9dfb3-2026-02-11](dcc9dfb3-2026-02-11.png)

```bash
export HOME=/home/dev/hn_rerank UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools XDG_CACHE_HOME=/home/dev/hn_rerank/.cache; uvx rodney stop
```

```output
Chrome stopped
```

Correction: cluster story rows don't use a `.story-item` class. Use `.cluster-card li` to count story rows on the clusters page.

```bash
export HOME=/home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
export UV_TOOL_DIR=/home/dev/hn_rerank/.cache/uv-tools
export XDG_CACHE_HOME=/home/dev/hn_rerank/.cache
uvx rodney start >/dev/null
uvx rodney open file:///home/dev/hn_rerank/clusters.html >/dev/null
uvx rodney waitstable >/dev/null
echo CLUSTER_CARDS=$(uvx rodney count '.cluster-card')
echo CLUSTER_ROWS=$(uvx rodney count '.cluster-card li')
echo CLUSTER_LINKS=$(uvx rodney count '.cluster-card li a')
uvx rodney stop >/dev/null

```

```output
CLUSTER_CARDS=60
CLUSTER_ROWS=280
CLUSTER_LINKS=280
```
