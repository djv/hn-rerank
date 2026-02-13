# Demo: RSS Cluster Labels and 2:1 HN RSS Mix

*2026-02-11T15:38:49Z*

This demo documents the ranking/UI changes implemented in this repo:
- RSS stories now reliably receive cluster labels via nearest-cluster fallback.
- Final story selection now targets a 2:1 HN:RSS ratio (best-effort when one side is scarce).

Below are executable checks that show the code paths and tests proving the behavior.

```bash
nl -ba generate_html.py | sed -n '345,385p'
```

```output
   345	
   346	    if rss_selected < target_rss:
   347	        rss_candidates = [
   348	            r for r in ranked if is_rss_result(r) and r.index not in used_indices
   349	        ]
   350	        hn_selected = [r for r in selected_results if not is_rss_result(r)]
   351	        hn_selected.sort(key=lambda r: r.hybrid_score)
   352	        for new_rss in rss_candidates:
   353	            if rss_selected >= target_rss or not hn_selected:
   354	                break
   355	            to_remove = hn_selected.pop(0)
   356	            selected_results.remove(to_remove)
   357	            used_indices.discard(to_remove.index)
   358	            selected_results.append(new_rss)
   359	            used_indices.add(new_rss.index)
   360	            rss_selected += 1
   361	    elif rss_selected > target_rss:
   362	        hn_candidates = [
   363	            r for r in ranked if not is_rss_result(r) and r.index not in used_indices
   364	        ]
   365	        rss_selected_items = [r for r in selected_results if is_rss_result(r)]
   366	        rss_selected_items.sort(key=lambda r: r.hybrid_score)
   367	        for new_hn in hn_candidates:
   368	            if rss_selected <= target_rss or not rss_selected_items:
   369	                break
   370	            to_remove = rss_selected_items.pop(0)
   371	            selected_results.remove(to_remove)
   372	            used_indices.discard(to_remove.index)
   373	            selected_results.append(new_hn)
   374	            used_indices.add(new_hn.index)
   375	            rss_selected -= 1
   376	
   377	    selected_results.sort(key=lambda x: x.hybrid_score, reverse=True)
   378	    return selected_results
   379	
   380	
   381	STORY_CARD_TEMPLATE: str = """
   382	<div class="story-card group{% if is_rss %} rss-story{% endif %}">
   383	    <div class="flex items-center gap-2 mb-0.5 flex-wrap">
   384	        <span class="px-1.5 py-0.5 rounded bg-hn/10 text-hn text-[10px] font-bold">
   385	            {{ score }}%
```

```bash
nl -ba generate_html.py | sed -n '267,290p'; echo '---'; nl -ba generate_html.py | sed -n '422,437p'; echo '---'; nl -ba generate_html.py | sed -n '930,973p'
```

```output
   267	    if len(cand_emb) == 0:
   268	        return {}
   269	
   270	    sim_to_clusters = cosine_similarity(cand_emb, cluster_centroids)
   271	    cluster_map: dict[int, int] = {}
   272	    for i in range(len(cands)):
   273	        max_sim = float(np.max(sim_to_clusters[i]))
   274	        if max_sim >= threshold or (force_assign_rss and cands[i].id < 0):
   275	            cluster_map[i] = int(np.argmax(sim_to_clusters[i]))
   276	        else:
   277	            cluster_map[i] = -1
   278	    return cluster_map
   279	
   280	
   281	def get_cluster_id_for_result(
   282	    result: RankResult,
   283	    cluster_labels: NDArray[np.int32] | None,
   284	    cand_cluster_map: dict[int, int],
   285	) -> int:
   286	    """Get cluster ID for a result (-1 if none)."""
   287	    if (
   288	        result.best_fav_index != -1
   289	        and result.max_sim_score >= SEMANTIC_MATCH_THRESHOLD
   290	        and cluster_labels is not None
---
   422	        points=story.points,
   423	        time_ago=story.time_ago,
   424	        url=link_url,
   425	        title=story.title,
   426	        hn_url=story.hn_url,
   427	        tldr=story.tldr,
   428	    )
   429	
   430	
   431	def resolve_cluster_name(
   432	    cluster_names: dict[int, str],
   433	    cluster_id: int,
   434	    allow_empty_fallback: bool = False,
   435	) -> str:
   436	    """Return cluster name with stable fallback for unnamed IDs."""
   437	    if cluster_id == -1:
---
   930	            use_classifier=args.use_classifier,
   931	            use_contrastive=args.contrastive,
   932	            knn_k=args.knn,
   933	            progress_callback=rank_cb,
   934	        )
   935	        progress.update(
   936	            r_task, completed=100, description="[green][+] Reranking complete."
   937	        )
   938	
   939	    # Compute cluster assignments for candidates (only if above similarity threshold)
   940	    cand_cluster_map = build_candidate_cluster_map(
   941	        cands,
   942	        cluster_centroids,
   943	        CLUSTER_SIMILARITY_THRESHOLD,
   944	        force_assign_rss=True,
   945	    )
   946	
   947	    seen_urls: set[str] = set()
   948	    seen_titles: set[str] = set()
   949	
   950	    def make_story_display(result: RankResult) -> StoryDisplay | None:
   951	        """Create StoryDisplay from RankResult, handling dedup."""
   952	        s: Story = cands[result.index]
   953	
   954	        url: str | None = s.url
   955	        title: str = s.title
   956	
   957	        norm_url: str = normalize_url(url) if url else f"hn:{s.id}"
   958	        norm_title: str = title.lower().strip() if title else ""
   959	
   960	        if norm_url in seen_urls or norm_title in seen_titles:
   961	            return None
   962	
   963	        if url:
   964	            seen_urls.add(norm_url)
   965	        if title:
   966	            seen_titles.add(norm_title)
   967	
   968	        reason: str = ""
   969	        reason_url: str = ""
   970	        if result.best_fav_index != -1 and result.best_fav_index < len(pos_stories):
   971	            fav_story = pos_stories[result.best_fav_index]
   972	            reason = fav_story.title
   973	            reason_url = f"https://news.ycombinator.com/item?id={fav_story.id}"
```

```bash
nl -ba generate_html.py | sed -n '440,480p'
```

```output
   440	        name = cluster_names[cluster_id].strip()
   441	        if name:
   442	            return name
   443	        if allow_empty_fallback:
   444	            return f"Group {cluster_id + 1}"
   445	        return ""
   446	    return f"Group {cluster_id + 1}"
   447	
   448	
   449	def _similarity_stats(values: NDArray[np.float32]) -> dict[str, float]:
   450	    if values.size == 0:
   451	        return {
   452	            "count": 0,
   453	            "mean": 0.0,
   454	            "median": 0.0,
   455	            "min": 0.0,
   456	            "max": 0.0,
   457	            "p10": 0.0,
   458	            "p90": 0.0,
   459	        }
   460	    return {
   461	        "count": int(values.size),
   462	        "mean": float(np.mean(values)),
   463	        "median": float(np.median(values)),
   464	        "min": float(np.min(values)),
   465	        "max": float(np.max(values)),
   466	        "p10": float(np.percentile(values, 10)),
   467	        "p90": float(np.percentile(values, 90)),
   468	    }
   469	
   470	
   471	def build_cluster_stats(
   472	    embeddings: NDArray[np.float32],
   473	    labels: NDArray[np.int32],
   474	    centroids: NDArray[np.float32],
   475	    cluster_names: dict[int, str],
   476	) -> dict[str, object]:
   477	    if embeddings.size == 0 or labels.size == 0 or centroids.size == 0:
   478	        return {}
   479	    emb_norm = embeddings / np.maximum(
   480	        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9
```

Now run focused tests covering the ratio policy and RSS cluster-label fallback behavior.

```bash
uv run pytest tests/test_html.py -q | sed -E 's/(passed, [0-9]+ warning(s)? in )([0-9]+\.[0-9]+s)/\1<TIME>/'
```

```output
....................                                                     [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/joblib/_multiprocessing_helpers.py:44
  /home/dev/hn_rerank/.venv/lib/python3.12/site-packages/joblib/_multiprocessing_helpers.py:44: UserWarning: [Errno 13] Permission denied.  joblib will operate in serial mode
    warnings.warn("%s.  joblib will operate in serial mode" % (e,))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
20 passed, 1 warning in <TIME>
```
