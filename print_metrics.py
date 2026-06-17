import sys
import json

for path in sys.argv[1:]:
    with open(path) as f:
        data = json.load(f)
    for r in data:
        print(
            f"{r['name']:<25} | NDCG@30: {r['ndcg_at_30']:.3f} | Hit@30: {r['hit_at_30']:.3f} | Recall@50: {r.get('recall_at_50', 0.0):.3f}"
        )
