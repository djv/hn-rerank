import re

with open('scripts/run_mlp_rf_tuning.py', 'r') as f:
    content = f.read()

# 1. Update run_eval
content = content.replace(
"""    r = data[0]
    print(
        f"  -> NDCG@30={r['ndcg_at_30']:.3f} mean_rank={r['mean_rank']:.1f} "
        f"P@5={r['precision_at_5']:.3f} ({elapsed:.1f}s)",
        flush=True,
    )
    return r""",
"""    r = data[0]
    r['aggregate_score'] = (r.get('recall_at_50', 0) * 1000) - r['median_rank']
    print(
        f"  -> Median={r['median_rank']:.1f} mean={r['mean_rank']:.1f} "
        f"Recall@50={r.get('recall_at_50', 0):.3f} Agg={r['aggregate_score']:.1f} ({elapsed:.1f}s)",
        flush=True,
    )
    return r"""
)

# 2. Update default params in run_mlp and run_rf
content = content.replace("def run_mlp(label, features, overrides=None, raw=False):", "def run_mlp(label, features, overrides=None, raw=True):")
content = content.replace("def run_rf(label, features, overrides=None, raw=False):", "def run_rf(label, features, overrides=None, raw=True):")

# 3. Replace all sort keys and max keys
content = content.replace("r[\"ndcg_at_30\"]", "r[\"aggregate_score\"]")

# 4. Replace print statements for NDCG to Agg
content = re.sub(r"NDCG@30=\{results\[0\]\['aggregate_score'\]:\.3f\}", r"Agg={results[0]['aggregate_score']:.1f}", content)
content = re.sub(r"NDCG@30=\{winner\['aggregate_score'\]:\.3f\}", r"Agg={winner['aggregate_score']:.1f}", content)

# 5. Fix final table
content = content.replace(
"""print(
    f"{'name':<35} {'NDCG@30':>7} {'MRR':>5} {'mean_rank':>7} {'P@5':>5} {'nonhn@0.5':>9}"
)
print("-" * 66)
for r in results:
    name = r.get("name", "?")[:34]
    nonhn = r.get("nonhn_at_0_5_fraction", -1)
    print(
        f"{name:<35} {r['aggregate_score']:>7.3f} {r['mrr']:>5.3f} "
        f"{r['mean_rank']:>7.1f} {r['precision_at_5']:>5.3f} "
        f"{nonhn:>9.2f}"
    )""",
"""print(
    f"{'name':<35} {'Agg Score':>9} {'Median':>6} {'Mean':>6} {'Recall@50':>9} {'NDCG@30':>7}"
)
print("-" * 76)
for r in results:
    name = r.get("name", "?")[:34]
    print(
        f"{name:<35} {r['aggregate_score']:>9.1f} {r['median_rank']:>6.0f} "
        f"{r['mean_rank']:>6.0f} {r.get('recall_at_50', 0):>9.3f} "
        f"{r.get('ndcg_at_30', 0):>7.3f}"
    )"""
)

# 6. Update Phase titles
content = content.replace("(16f,", "(16f+raw,")
content = content.replace("MLP on 16f:", "MLP on 16f+raw:")
content = content.replace("RF sweep (16f, no raw)", "RF sweep (16f+raw)")
content = content.replace("Phase 5: MLP winner on 16f+raw", "Phase 5: MLP winner recap")
content = content.replace("Phase 7: RF winner on 16f+raw", "Phase 7: RF winner recap")

with open('scripts/run_mlp_rf_tuning.py', 'w') as f:
    f.write(content)
