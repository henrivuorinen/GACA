"""GACA quickstart on synthetic data.

Five dense groups in 5 dimensions plus eight planted anomalies far from
everything. GACA is not told how many groups there are. It should recover the
dense groups as heavy Suns and leave each anomaly as a Lone Sun (mass 1).
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

# 2. GACA: Solar Genesis on a 2,000-point coreset, then assign every row.
#    The coreset is a random sample, so the anomalies are added to it explicitly
#    here to show the Lone Sun behaviour. In real use an anomaly outside the
#    coreset cannot become a Lone Sun; it is caught by the OOD distance check
#    instead (see the README).
core_idx = np.r_[rng.choice(len(X_groups), 1992, replace=False), np.flatnonzero(is_anomaly)]
t0 = time.time()
model = GACA(gamma_clustering=1.0, epsilon=0.05, eta=0.5,
             sample_size=len(core_idx), random_state=0).fit(X[core_idx])
labels = model.assign(X)
elapsed = time.time() - t0

masses = model.sun_masses_
order = np.argsort(masses)[::-1]
lone = np.flatnonzero(masses == 1)

print(f"GACA found {len(masses)} Suns in {elapsed:.1f} s "
      f"({model.n_iters_} genesis iterations), with no k given.")
print("Heaviest Suns (share of coreset mass):",
      ", ".join(f"{masses[k] / masses.sum():.1%}" for k in order[:5]))
print(f"Lone Suns (mass 1): {len(lone)}")
caught = int(np.isin(labels[is_anomaly], lone).sum())
print(f"Planted anomalies isolated as Lone Suns: {caught} / {is_anomaly.sum()}")

# 3. K-Means with the TRUE number of groups, for contrast
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
sizes = np.bincount(km.labels_)
absorbed = int((sizes[km.labels_[is_anomaly]] > 50).sum())
print(f"\nK-Means (k=5) cluster sizes: {sorted(sizes.tolist(), reverse=True)}")
print(f"Anomalies absorbed into a cluster of normal points: "
      f"{absorbed} / {is_anomaly.sum()}")
