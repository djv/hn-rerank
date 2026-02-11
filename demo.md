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
   345	    def is_rss_result(res: RankResult) -> bool:
   346	        return cands[res.index].id < 0
   347	
   348	    # Target 2:1 HN:RSS in the final list (best-effort under availability limits).
   349	    desired_rss: int = count // 3
   350	    available_rss = sum(1 for r in ranked if is_rss_result(r))
   351	    available_hn = len(ranked) - available_rss
   352	    min_rss = max(0, count - available_hn)
   353	    max_rss = min(count, available_rss)
   354	    target_rss = min(max(desired_rss, min_rss), max_rss)
   355	
   356	    rss_selected = sum(1 for r in selected_results if is_rss_result(r))
   357	
   358	    if rss_selected < target_rss:
   359	        rss_candidates = [
   360	            r for r in ranked if is_rss_result(r) and r.index not in used_indices
   361	        ]
   362	        hn_selected = [r for r in selected_results if not is_rss_result(r)]
   363	        hn_selected.sort(key=lambda r: r.hybrid_score)
   364	        for new_rss in rss_candidates:
   365	            if rss_selected >= target_rss or not hn_selected:
   366	                break
   367	            to_remove = hn_selected.pop(0)
   368	            selected_results.remove(to_remove)
   369	            used_indices.discard(to_remove.index)
   370	            selected_results.append(new_rss)
   371	            used_indices.add(new_rss.index)
   372	            rss_selected += 1
   373	    elif rss_selected > target_rss:
   374	        hn_candidates = [
   375	            r for r in ranked if not is_rss_result(r) and r.index not in used_indices
   376	        ]
   377	        rss_selected_items = [r for r in selected_results if is_rss_result(r)]
   378	        rss_selected_items.sort(key=lambda r: r.hybrid_score)
   379	        for new_hn in hn_candidates:
   380	            if rss_selected <= target_rss or not rss_selected_items:
   381	                break
   382	            to_remove = rss_selected_items.pop(0)
   383	            selected_results.remove(to_remove)
   384	            used_indices.discard(to_remove.index)
   385	            selected_results.append(new_hn)
```

```bash
nl -ba generate_html.py | sed -n '267,290p'; echo '---'; nl -ba generate_html.py | sed -n '422,437p'; echo '---'; nl -ba generate_html.py | sed -n '930,973p'
```

```output
   267	def build_candidate_cluster_map(
   268	    cands: list[Story],
   269	    cluster_centroids: Optional[NDArray[np.float32]],
   270	    threshold: float,
   271	    force_assign_rss: bool = False,
   272	) -> dict[int, int]:
   273	    """Assign candidates to clusters based on centroid similarity."""
   274	    if cluster_centroids is None or not cands:
   275	        return {}
   276	
   277	    cand_texts = [c.text_content for c in cands]
   278	    cand_emb = rerank.get_cluster_embeddings(cand_texts)
   279	    if len(cand_emb) == 0:
   280	        return {}
   281	
   282	    sim_to_clusters = cosine_similarity(cand_emb, cluster_centroids)
   283	    cluster_map: dict[int, int] = {}
   284	    for i in range(len(cands)):
   285	        max_sim = float(np.max(sim_to_clusters[i]))
   286	        if max_sim >= threshold or (force_assign_rss and cands[i].id < 0):
   287	            cluster_map[i] = int(np.argmax(sim_to_clusters[i]))
   288	        else:
   289	            cluster_map[i] = -1
   290	    return cluster_map
---
   422	_INDEX_TEMPLATE = _JINJA_ENV.from_string(HTML_TEMPLATE)
   423	_CLUSTER_STORY_TEMPLATE = _JINJA_ENV.from_string(CLUSTER_STORY_TEMPLATE)
   424	_CLUSTER_CARD_TEMPLATE = _JINJA_ENV.from_string(CLUSTER_CARD_TEMPLATE)
   425	_CLUSTERS_TEMPLATE = _JINJA_ENV.from_string(CLUSTERS_PAGE_TEMPLATE)
   426	
   427	
   428	def generate_story_html(story: StoryDisplay) -> str:
   429	    link_url = story.url or story.hn_url or "#"
   430	    return _STORY_TEMPLATE.render(
   431	        score=story.match_percent,
   432	        is_rss=story.id < 0,
   433	        cluster_name=story.cluster_name,
   434	        points=story.points,
   435	        time_ago=story.time_ago,
   436	        url=link_url,
   437	        title=story.title,
---
   930	        )
   931	
   932	        # 4. Reranking
   933	        r_task: TaskID = progress.add_task("[*] Reranking stories...", total=100)
   934	
   935	        def rank_cb(curr: int, total: int) -> None:
   936	            progress.update(r_task, total=total, completed=curr)
   937	
   938	        ranked: list[RankResult] = rerank.rank_stories(
   939	            cands,
   940	            p_emb,
   941	            n_emb,
   942	            use_classifier=args.use_classifier,
   943	            use_contrastive=args.contrastive,
   944	            knn_k=args.knn,
   945	            progress_callback=rank_cb,
   946	        )
   947	        progress.update(
   948	            r_task, completed=100, description="[green][+] Reranking complete."
   949	        )
   950	
   951	    # Compute cluster assignments for candidates (only if above similarity threshold)
   952	    cand_cluster_map = build_candidate_cluster_map(
   953	        cands,
   954	        cluster_centroids,
   955	        CLUSTER_SIMILARITY_THRESHOLD,
   956	        force_assign_rss=True,
   957	    )
   958	
   959	    seen_urls: set[str] = set()
   960	    seen_titles: set[str] = set()
   961	
   962	    def make_story_display(result: RankResult) -> Optional[StoryDisplay]:
   963	        """Create StoryDisplay from RankResult, handling dedup."""
   964	        s: Story = cands[result.index]
   965	
   966	        url: Optional[str] = s.url
   967	        title: str = s.title
   968	
   969	        norm_url: str = normalize_url(url) if url else f"hn:{s.id}"
   970	        norm_title: str = title.lower().strip() if title else ""
   971	
   972	        if norm_url in seen_urls or norm_title in seen_titles:
   973	            return None
```

```bash
nl -ba generate_html.py | sed -n '440,480p'
```

```output
   440	    )
   441	
   442	
   443	def resolve_cluster_name(
   444	    cluster_names: dict[int, str],
   445	    cluster_id: int,
   446	    allow_empty_fallback: bool = False,
   447	) -> str:
   448	    """Return cluster name with stable fallback for unnamed IDs."""
   449	    if cluster_id == -1:
   450	        return ""
   451	    if cluster_id in cluster_names:
   452	        name = cluster_names[cluster_id].strip()
   453	        if name:
   454	            return name
   455	        if allow_empty_fallback:
   456	            return f"Group {cluster_id + 1}"
   457	        return ""
   458	    return f"Group {cluster_id + 1}"
   459	
   460	
   461	def _similarity_stats(values: NDArray[np.float32]) -> dict[str, float]:
   462	    if values.size == 0:
   463	        return {
   464	            "count": 0,
   465	            "mean": 0.0,
   466	            "median": 0.0,
   467	            "min": 0.0,
   468	            "max": 0.0,
   469	            "p10": 0.0,
   470	            "p90": 0.0,
   471	        }
   472	    return {
   473	        "count": int(values.size),
   474	        "mean": float(np.mean(values)),
   475	        "median": float(np.median(values)),
   476	        "min": float(np.min(values)),
   477	        "max": float(np.max(values)),
   478	        "p10": float(np.percentile(values, 10)),
   479	        "p90": float(np.percentile(values, 90)),
   480	    }
```

Now run focused tests covering the ratio policy and RSS cluster-label fallback behavior.

```bash
uv run pytest tests/test_html.py -q
```

```output
....................                                                     [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/joblib/_multiprocessing_helpers.py:44
  /home/dev/hn_rerank/.venv/lib/python3.12/site-packages/joblib/_multiprocessing_helpers.py:44: UserWarning: [Errno 13] Permission denied.  joblib will operate in serial mode
    warnings.warn("%s.  joblib will operate in serial mode" % (e,))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
20 passed, 1 warning in 4.91s
```
