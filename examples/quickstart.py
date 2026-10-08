"""GACA quickstart on synthetic data.

Five dense groups in 5 dimensions plus eight planted anomalies far from
everything. GACA is not told how many groups there are. It should recover the
dense groups as heavy Suns and leave each anomaly as a Lone Sun, even though the
random coreset will almost certainly miss most of the anomalies.
K-Means, even when given the true k, has no way to set the anomalies aside and
absorbs them into its clusters.

The data is generated directly on a standardised scale. With real data,
standardise the features first (and project to ~5 dimensions with PCA if you
have many), as described in the README.

    python examples/quickstart.py
"""
import time

import numpy as np
from sklearn.cluster import KMeans

from gaca import GACA

rng = np.random.default_rng(0)
D = 5

# 1. Data: 5 groups of 600 points at 3 * e_i, plus 8 anomalies placed far away
centers = 3.0 * np.eye(D)
X_groups = np.vstack([c + rng.normal(0, 0.3, size=(600, D)) for c in centers])
anomalies = np.vstack([-6.0 * np.eye(D),                  # one per negative axis
                       [6, 6, 0, 0, 0], [0, 0, 6, 6, 6], [-6, -6, -6, -6, -6]])
X = np.vstack([X_groups, anomalies]).astype(float)
is_anomaly = np.r_[np.zeros(len(X_groups), bool), np.ones(len(anomalies), bool)]

# 2. GACA: Solar Genesis on a random 1,000-point coreset, then assign every row.
#    Most anomalies are not in the coreset. They are still isolated, because
#    assignment notices that no Sun pulls them and registers them as Lone Suns.
t0 = time.time()
model = GACA(gamma_clustering=1.0, sample_size=1000, random_state=0).fit(X)
labels = model.assign(X)
elapsed = time.time() - t0

sizes = np.bincount(labels)
order = np.argsort(sizes)[::-1]
lone = model.is_lone(labels)

print(f"GACA found {model.n_suns_} Suns in {elapsed:.1f} s "
      f"({model.n_iters_} genesis iterations), with no k given.")
print("Heaviest Suns (share of rows):",
      ", ".join(f"{sizes[k] / len(X):.1%}" for k in order[:5]))
print(f"Rows in Lone Suns: {lone.sum()} (genuine rows among them: {lone[~is_anomaly].sum()})")
print(f"Planted anomalies isolated as Lone Suns: {lone[is_anomaly].sum()} / {is_anomaly.sum()}")

# 3. K-Means with the TRUE number of groups, for contrast
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
sizes = np.bincount(km.labels_)
absorbed = int((sizes[km.labels_[is_anomaly]] > 50).sum())
print(f"\nK-Means (k=5) cluster sizes: {sorted(sizes.tolist(), reverse=True)}")
print(f"Anomalies absorbed into a cluster of normal points: "
      f"{absorbed} / {is_anomaly.sum()}")
