"""
Regression check for within-group scaling (AutoGACA(separate_modes=...)).

Runs the clustering and anomaly benchmarks of benchmark_suite.py with the
option off and on, and prints the change per dataset.

    python experiments/compare_separate_modes.py
"""
import os
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_suite import anomaly_datasets, clustering_datasets  # noqa: E402

from gaca import AutoGACA  # noqa: E402

warnings.filterwarnings("ignore")


def best_level(auto, y):
    return max(adjusted_rand_score(y, lv['labels']) for lv in auto.levels_)


def main():
    print("== clustering (ARI): main view / linked view / best hierarchy level ==")
    print(f"{'data':26} {'off':>18} {'on':>18}  {'time off/on':>12}")
    for name, X, y in clustering_datasets():
        res = {}
        for sm in (False, True):
            t = time.perf_counter()
            a = AutoGACA(separate_modes=sm).fit(X)
            res[sm] = (adjusted_rand_score(y, a.labels_), adjusted_rand_score(y, a.linked_labels_),
                       best_level(a, y), time.perf_counter() - t)
        f = lambda r: f"{r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f}"
        print(f"{name:26} {f(res[False]):>18} {f(res[True]):>18}  {res[False][3]:5.1f}/{res[True][3]:5.1f}s",
              flush=True)

    print("\n== anomalies: AP / AUC / caught / genuine flagged ==")
    print(f"{'data':34} {'off':>28} {'on':>28}")
    for name, X, y in anomaly_datasets():
        res = {}
        for sm in (False, True):
            a = AutoGACA(separate_modes=sm, link_view=False).fit(X)
            res[sm] = (average_precision_score(y, a.anomaly_score_), roc_auc_score(y, a.anomaly_score_),
                       (a.anomaly_ & y).sum() / max(y.sum(), 1), a.anomaly_[~y].mean())
        f = lambda r: f"{r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f} / {100 * r[3]:.1f}%"
        print(f"{name:34} {f(res[False]):>28} {f(res[True]):>28}", flush=True)


if __name__ == "__main__":
    main()
