"""
GACA against established methods: clustering, anomaly detection and scale.

    python experiments/benchmark_suite.py clustering
    python experiments/benchmark_suite.py anomaly
    python experiments/benchmark_suite.py scale

Fairness rules:

* Every method sees the same input: the space AutoGACA's preprocessing
  produces (column selection, robust scaling, PCA above 10 columns). The
  comparison is between algorithms, not preprocessing pipelines.
* Methods run with their default or a standard documented setting; nothing is
  tuned per dataset with the labels. K-Means is given the true number of
  clusters and is a reference, not a competitor.
* Rows a method leaves unclustered (HDBSCAN noise, GACA anomalies) count as one
  extra label in the ARI, and the share of clustered rows is reported next to
  it, so a method cannot score well by discarding hard rows.

The SDSS rows need data/sdss_objects.csv (python examples/sdss/fetch_sdss.py);
they are skipped if the file is missing.
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris, load_wine,
                              make_blobs, make_circles, make_moons)
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
                             normalized_mutual_info_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(__file__))
from improvements_benchmark import blobs as uneven_blobs  # noqa: E402
from improvements_benchmark import company, inject  # noqa: E402

from gaca import AutoGACA  # noqa: E402

SDSS = os.path.join(os.path.dirname(__file__), '..', 'data', 'sdss_objects.csv')


# --------------------------------------------------------------------------- data

def noisy_subspace(rng, n=6000, d=30, d_sig=5):
    """Six clusters in a 5-D subspace, plus 25 noise dimensions, rotated."""
    sizes = (np.array([.3, .2, .15, .15, .1, .1]) * n).astype(int)
    C = rng.normal(0, 2.5, (6, d_sig))
    S = np.vstack([C[i] + rng.normal(0, 1, (s, d_sig)) for i, s in enumerate(sizes)])
    X = np.hstack([S, rng.normal(0, 1, (len(S), d - d_sig))])
    X = X @ np.linalg.qr(rng.normal(size=(d, d)))[0]
    return X, np.repeat(np.arange(6), sizes)


def clustering_datasets():
    rng = np.random.default_rng(0)
    out = []
    for name, f in [('iris', load_iris), ('wine', load_wine),
                    ('breast cancer', load_breast_cancer), ('digits', load_digits)]:
        b = f()
        out.append((name, b.data, b.target))
    X, y = uneven_blobs(np.random.default_rng(0))
    out.append(('uneven blobs 5-D', X, y))
    out.append(('6 clusters in 30-D noise', *noisy_subspace(rng)))
    X, y = make_blobs(6000, centers=[[0, 0], [6, 0], [3, 6]], cluster_std=[0.4, 1.2, 2.0],
                      random_state=0)
    out.append(('varied density', X, y))
    X, y = make_blobs(6000, centers=4, cluster_std=0.8, random_state=3)
    out.append(('anisotropic', X @ np.array([[0.6, -0.6], [-0.4, 0.8]]), y))
    out.append(('two moons', *make_moons(4000, noise=0.06, random_state=0)))
    out.append(('circles', *make_circles(4000, noise=0.04, factor=0.45, random_state=0)))
    if os.path.exists(SDSS):
        import pandas as pd
        df = pd.read_csv(SDSS)
        X = df.drop(columns=['ra', 'dec', 'class', 'subClass'])
        out.append(('SDSS objects (class)', X, df['class'].astype('category').cat.codes.to_numpy()))
    return out


# --------------------------------------------------------------------------- clustering

def _score(y, labels):
    covered = labels >= 0
    return dict(ari=adjusted_rand_score(y, labels), nmi=normalized_mutual_info_score(y, labels),
                covered=covered.mean(), k=len(set(labels[covered].tolist())))


def run_clustering():
    rows = []
    for name, X, y in clustering_datasets():
        t0 = time.perf_counter()
        auto = AutoGACA().fit(X)
        t_auto = time.perf_counter() - t0
        Z = auto.Z_                                      # the shared preprocessed space
        n, k_true = len(Z), len(np.unique(y))

        res = {'GACA (auto)': (auto.labels_, t_auto)}
        t0 = time.perf_counter()
        linked = AutoGACA(link_tau=0.6).fit(X)
        res['GACA (auto + link)'] = (linked.labels_, time.perf_counter() - t0)
        for label, mcs in [('HDBSCAN (default)', 5), ('HDBSCAN (min size 1%)', max(5, n // 100))]:
            t0 = time.perf_counter()
            lab = HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(Z)
            res[label] = (lab, time.perf_counter() - t0)
        t0 = time.perf_counter()
        lab = KMeans(k_true, n_init=10, random_state=0).fit_predict(Z)
        res['K-Means (true k) ref'] = (lab, time.perf_counter() - t0)

        for method, (lab, t) in res.items():
            rows.append(dict(data=name, n=n, k_true=k_true, method=method, t=t, **_score(y, lab)))
        print(f"{name:26} " + "  ".join(f"{m.split(' (')[0][:7]}{'+' if 'link' in m else ''}"
                                        f"{'1%' if '1%' in m else ''}:{_score(y, l)['ari']:.2f}"
                                        for m, (l, _) in res.items()), flush=True)
    return rows


# --------------------------------------------------------------------------- anomalies

def anomaly_datasets():
    """(name, X, is_anomaly). Planted groups, rare classes, and a public benchmark."""
    out = []
    for name, gen in [('blobs + planted', uneven_blobs), ('company-like + planted', company)]:
        rng = np.random.default_rng(1)
        Xg, _ = gen(rng)
        X, groups = inject(Xg, rng)
        y = np.zeros(len(X), bool)
        for g in groups:
            y[g['rows']] = True
        out.append((name, X, y))

    b = load_breast_cancer()                  # malignant thinned to 5% (classic set-up)
    rng = np.random.default_rng(0)
    mal = np.flatnonzero(b.target == 0)
    ben = np.flatnonzero(b.target == 1)
    keep = np.r_[ben, rng.choice(mal, int(0.05 * len(ben) / 0.95), replace=False)]
    out.append(('breast cancer (5% malignant)', b.data[keep], b.target[keep] == 0))

    if os.path.exists(SDSS):
        import pandas as pd
        df = pd.read_csv(SDSS)
        feats = df.drop(columns=['ra', 'dec', 'class', 'subClass'])
        q = np.flatnonzero(df['class'] == 'QSO')
        rest = np.flatnonzero(df['class'] != 'QSO')
        keep = np.r_[rest, rng.choice(q, int(0.01 * len(rest) / 0.99), replace=False)]
        out.append(('SDSS: quasars thinned to 1%', feats.iloc[keep].reset_index(drop=True),
                    (df['class'].to_numpy() == 'QSO')[keep]))
        wd = df['subClass'].fillna('').str.startswith('WD').to_numpy()
        out.append(('SDSS: white dwarfs (natural ~1%)', feats, wd))

    try:
        from sklearn.datasets import fetch_kddcup99
        import pandas as pd
        kdd = fetch_kddcup99(subset='SA', percent10=True, random_state=0, as_frame=True)
        X = kdd.data.apply(pd.to_numeric, errors='coerce')
        X = X.loc[:, X.notna().mean() > 0.95]
        y = (kdd.target.astype(str) != "b'normal.'").to_numpy() & (kdd.target.astype(str) != 'normal.').to_numpy()
        out.append(('KDD Cup 99 network attacks (SA)', X, y))
    except Exception as err:                  # offline: skip the public benchmark
        print(f"(KDD Cup 99 skipped: {err.__class__.__name__})")
    return out


def run_anomaly():
    rows = []
    for name, X, y in anomaly_datasets():
        t0 = time.perf_counter()
        auto = AutoGACA().fit(X)
        t_auto = time.perf_counter() - t0
        Z = auto.Z_
        scores = {'GACA': (auto.anomaly_score_, auto.anomaly_, t_auto)}
        t0 = time.perf_counter()
        iso = IsolationForest(random_state=0).fit(Z)
        scores['Isolation Forest'] = (-iso.score_samples(Z), iso.predict(Z) < 0, time.perf_counter() - t0)
        t0 = time.perf_counter()
        lof = LocalOutlierFactor(n_neighbors=20)
        flag = lof.fit_predict(Z) < 0
        scores['LOF'] = (-lof.negative_outlier_factor_, flag, time.perf_counter() - t0)
        t0 = time.perf_counter()
        hdb = HDBSCAN(min_cluster_size=5, copy=True).fit(Z)
        # scikit-learn's HDBSCAN has no GLOSH score; 1 - membership strength is
        # its built-in equivalent (noise points have strength 0).
        scores['HDBSCAN (1 - membership)'] = (1.0 - hdb.probabilities_, hdb.labels_ < 0,
                                              time.perf_counter() - t0)

        for method, (sc, flag, t) in scores.items():
            tp = (flag & y).sum()
            rows.append(dict(data=name, n=len(y), rate=y.mean(), method=method, t=t,
                             auc=roc_auc_score(y, sc), ap=average_precision_score(y, sc),
                             flagged=flag.mean(), precision=tp / max(flag.sum(), 1),
                             recall=tp / max(y.sum(), 1)))
        print(f"{name:34} n={len(y):>6} anomalies {100 * y.mean():.1f}% | " + "  ".join(
            f"{m.split(' (')[0]}: AUC {r['auc']:.2f} AP {r['ap']:.2f}"
            for m, r in ((r['method'], r) for r in rows[-4:])), flush=True)
    return rows


# --------------------------------------------------------------------------- scale

SCALE_METHODS = ['GACA (auto)', 'GACA (fixed gamma)', 'HDBSCAN', 'Isolation Forest', 'LOF',
                 'K-Means']


def _scale_data(n, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (8, 5))
    lab = rng.integers(0, 8, n)
    return centers[lab] + rng.normal(0, 1, (n, 5))


def scale_one(method, n):
    """Run one method on n rows in this process; print seconds and peak memory."""
    import json
    import resource
    X = _scale_data(n)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    if method == 'GACA (auto)':
        AutoGACA(scale='none').fit(X)
    elif method == 'GACA (fixed gamma)':
        from gaca import GACA
        model = GACA(gamma_clustering=0.5, random_state=0).fit(X)
        for s in range(0, n, 200_000):               # streamed in batches
            model.assign(X[s:s + 200_000])
    elif method == 'HDBSCAN':
        HDBSCAN(min_cluster_size=max(5, n // 100)).fit(X)
    elif method == 'Isolation Forest':
        IsolationForest(random_state=0).fit(X).score_samples(X)
    elif method == 'LOF':
        LocalOutlierFactor(n_neighbors=20).fit(X)
    elif method == 'K-Means':
        KMeans(8, n_init=1, random_state=0).fit(X)
    t = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform != 'darwin' else 1024    # macOS reports bytes, Linux KiB
    print(json.dumps(dict(t=t, mem_mb=peak / scale / 1024, data_mb=X.nbytes / 2 ** 20)))


def run_scale(sizes, timeout):
    import json
    import subprocess
    rows = []
    skip = set()
    for n in sizes:
        for m in SCALE_METHODS:
            if m in skip:
                rows.append(dict(data=f"{n:,}", method=m, t=np.nan, mem_mb=np.nan))
                continue
            try:
                out = subprocess.run([sys.executable, __file__, 'scale-one', m, str(n)],
                                     capture_output=True, text=True, timeout=timeout)
                r = json.loads(out.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                r = dict(t=np.inf, mem_mb=np.nan)
                skip.add(m)                          # larger sizes would time out too
            rows.append(dict(data=f"{n:,}", method=m, **r))
            print(f"n={n:>9,} {m:20} {r['t']:8.1f} s  {r.get('mem_mb', np.nan):8.0f} MB", flush=True)
    return rows


def print_table(rows, cols, fmt):
    import pandas as pd
    df = pd.DataFrame(rows)
    for c in cols:
        piv = df.pivot_table(index='data', columns='method', values=c, sort=False)
        print(f"\n== {c} ==")
        print(piv.map(lambda v: fmt[c].format(v)).to_string())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('part', choices=['clustering', 'anomaly', 'scale', 'scale-one'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--timeout', type=float, default=600)
    args = ap.parse_args()
    if args.part == 'scale-one':
        scale_one(args.args[0], int(args.args[1]))
        return
    if args.part == 'clustering':
        rows = run_clustering()
        df = print_table(rows, ['ari', 'covered', 'k', 't'],
                         dict(ari='{:.3f}', covered='{:.0%}', k='{:.0f}', t='{:.1f}'))
        df.to_csv('benchmark_clustering_suite.csv', index=False)
    elif args.part == 'anomaly':
        rows = run_anomaly()
        df = print_table(rows, ['auc', 'ap', 'precision', 'recall', 'flagged', 't'],
                         dict(auc='{:.3f}', ap='{:.3f}', precision='{:.2f}', recall='{:.2f}',
                              flagged='{:.1%}', t='{:.1f}'))
        df.to_csv('benchmark_anomaly_suite.csv', index=False)
    elif args.part == 'scale':
        sizes = [int(a) for a in args.args] or [10_000, 100_000, 1_000_000]
        rows = run_scale(sizes, args.timeout)
        df = print_table(rows, ['t', 'mem_mb'], dict(t='{:.1f}', mem_mb='{:.0f}'))
        df.to_csv('benchmark_scale_suite.csv', index=False)


if __name__ == "__main__":
    main()
