import numpy as np

def compute_plateau_decay(ages_days):
    k = 0.01
    inflection = 365
    weights = 1.0 / (1.0 + np.exp(k * (ages_days - inflection)))
    return weights

ages = np.array([1, 30, 90, 180, 365, 500, 730])
weights = compute_plateau_decay(ages)

print("k=0.01, inflection=365")
for age, w in zip(ages, weights):
    print(f"Age {age:3d} days: {w:.4f}")