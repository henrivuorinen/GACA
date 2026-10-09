"""
GACA on public OpenML datasets from several domains.

Clustering (labels used only for scoring):
    banknote-authentication  finance / forgery      1,372 x 4,   2 classes
    seeds                    agriculture              210 x 7,   3 classes
    ecoli                    biology                  336 x 7,   8 classes
    segment                  image analysis         2,310 x 19,  7 classes
    pendigits                handwriting sensors   10,992 x 16, 10 classes

Anomaly detection (ODDS conventions):
    mammography  medicine: class 1 (malignant, 2.3%) is anomalous
    shuttle      aerospace: class 1 normal, classes 2, 3, 5, 6, 7 anomalous
                 (about 7%), class 4 dropped

The fairness rules of benchmark_suite.py apply: every method sees AutoGACA's
preprocessed space, nothing is tuned per dataset, and K-Means is given the
true number of classes (a reference, not a competitor). Datasets are cached in
data/openml/.

    python experiments/benchmark_openml.py      # a few minutes
"""
import os
import time
import warnings

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             normalized_mutual_info_score, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor

from gaca import AutoGACA

warnings.filterwarnings("ignore")
CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'openml')

CLUSTERING = [('banknote-authentication', 1462, 'finance'), ('seeds', 1499, 'agriculture'),
              ('ecoli', 39, 'biology'), ('segment', 36, 'image analysis'),
              ('pendigits', 32, 'handwriting')]
ANOMALY = [('mammography', 310, 'medicine'), ('shuttle', 40685, 'aerospace')]


def load(did):
    b = fetch_openml(data_id=did, as_frame=True, data_home=CACHE, parser='auto')
    return b.data, b.target.astype(str).to_numpy()


def anomaly_labels(name, X, y):
    if name == 'mammography':
        return X, y == '1'
    if name == 'shuttle':                            # ODDS: drop class 4
        keep = y != '4'
        return X[keep].reset_index(drop=True), (y[keep] != '1')
    raise ValueError(name)


def clustering():
    print("== Clustering: ARI (NMI) against the dataset's classes ==")
    cols = ['GACA', 'GACA linked', 'GACA best level', 'HDBSCAN', 'K-Means (true k)']
    print(f"{'dataset':<26}{'domain':<16}" + "".join(f"{c:>19}" for c in cols) + f"{'GACA s':>8}")
    summary = {c: [] for c in cols}
    for name, did, domain in CLUSTERING:
        X, y = load(did)
        t = time.perf_counter()
        auto = AutoGACA().fit(X)
        t = time.perf_counter() - t
        Z, k = auto.Z_, len(np.unique(y))
        best = max(auto.levels_, key=lambda lv: adjusted_rand_score(y, lv['labels']))['labels']
        labels = {'GACA': auto.labels_, 'GACA linked': auto.linked_labels_,
                  'GACA best level': best,
                  'HDBSCAN': HDBSCAN(min_cluster_size=max(5, len(Z) // 100)).fit_predict(Z),
                  'K-Means (true k)': KMeans(k, n_init=10, random_state=0).fit_predict(Z)}
        cells = []
        for c in cols:
            ari = adjusted_rand_score(y, labels[c])
            nmi = normalized_mutual_info_score(y, labels[c])
            summary[c].append(ari)
            cells.append(f"{ari:.2f} ({nmi:.2f})")
        print(f"{name:<26}{domain:<16}" + "".join(f"{c:>19}" for c in cells) + f"{t:8.1f}",
              flush=True)
    print(f"{'mean ARI':<42}" + "".join(f"{np.mean(summary[c]):>19.2f}" for c in cols))


def anomalies():
    print("\n== Anomaly detection: average precision (ROC AUC); flagged share and recall "
          "at each method's own threshold ==")
    methods = ['GACA', 'Isolation Forest', 'LOF']
    for name, did, domain in ANOMALY:
        X, y = load(did)
        X, a = anomaly_labels(name, X, y)
        auto = AutoGACA(link_view=False).fit(X)
        Z = auto.Z_
        iso = IsolationForest(random_state=0).fit(Z)
        lof = LocalOutlierFactor(n_neighbors=20)
        lof_flag = lof.fit_predict(Z) < 0
        res = {'GACA': (auto.anomaly_score_, auto.anomaly_),
               'Isolation Forest': (-iso.score_samples(Z), iso.predict(Z) < 0),
               'LOF': (-lof.negative_outlier_factor_, lof_flag)}
        print(f"{name} ({domain}): {len(a):,} rows, {100 * a.mean():.1f}% anomalous")
        for m in methods:
            sc, flag = res[m]
            print(f"   {m:<18} AP {average_precision_score(a, sc):.3f}  AUC {roc_auc_score(a, sc):.3f}"
                  f"   flagged {100 * flag.mean():5.1f}%  recall {(flag & a).sum() / a.sum():.2f}"
                  f"  precision {(flag & a).sum() / max(flag.sum(), 1):.2f}", flush=True)


if __name__ == "__main__":
    clustering()
    anomalies()
