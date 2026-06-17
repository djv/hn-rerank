import json
import glob

def main():
    files = glob.glob("results/*.json")
    results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            if not data:
                continue
            r = data[0]
            agg = (r.get('recall_at_50', 0) * 1000) - r['median_rank']
            r['aggregate_score'] = agg
            results.append(r)
            
    results.sort(key=lambda x: x['aggregate_score'], reverse=True)
    
    print("=== Top 10 Configurations ===")
    print(f"{'Name':<20} {'AggScore':>9} {'MedianRk':>9} {'Recall@50':>9} {'Plateau(%)':>10}")
    print("-" * 62)
    for r in results[:10]:
        plateau = r.get('nonhn_at_0_5_fraction', 0.0) * 100
        print(f"{r['name']:<20} {r['aggregate_score']:>9.1f} {r['median_rank']:>9.1f} {r.get('recall_at_50', 0):>9.3f} {plateau:>9.1f}%")

if __name__ == "__main__":
    main()
