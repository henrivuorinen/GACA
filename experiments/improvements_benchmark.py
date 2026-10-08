"""
Thesis pipeline vs the improved defaults, on synthetic data with planted anomalies.

Two datasets of 20,000 standardised rows, each with the nine planted anomaly
groups of Sec. 6.4 (5, 10 and 20 standard deviations x multiplicities 1, 3, 10):

    blobs     five Gaussian groups in 5-D with uneven sizes and widths
    company   a heavy-tailed, partly discretised stand-in for the thesis table
              (log employees, log reviews, age, lat/long around eight cities)

Methods (all on the same uniform coreset):

    thesis          Newtonian assignment M/d^2; anomaly = assigned to a mass-1 Sun
    thesis+ood95    plus the 95th-percentile radius rule of Sec. 6.5
    pull            GACA defaults: pull assignment with Lone-Sun registration
    pull+link g5    pull at gamma = 5 with saddle linking (link_tau=0.6):
                    a 5x finer resolution, with linking joining the pieces

Columns: share of anomaly groups detected (every copy flagged), the same split
by injection distance, share of genuine rows flagged (false positives), ARI on
genuine rows against the generating groups, and R^2 of the Mixture of Experts
on a fresh test set with new anomalies.

    python experiments/improvements_benchmark.py [--seeds 4] [--coreset 2000]
"""
import argparse
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.preprocessing import StandardScaler

from gaca import GACA

DISTANCES = [5.0, 10.0, 20.0]
MULTIPLICITIES = [1, 3, 10]
EPS = 0.05
SIZE_MID = np.array([5, 30, 125, 350, 750, 3000, 7500, 15000])


def inject(Xs, rng):
    """Append the nine anomaly groups; returns the data and each group's rows."""
    d = Xs.shape[1]
    dirs, thresh, tries = [], 0.5, 0
    while len(dirs) < len(DISTANCES) * len(MULTIPLICITIES):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        if all(np.dot(v, u) < thresh for u in dirs):
            dirs.append(v)
        tries += 1
        if tries > 500:
            thresh, tries = min(thresh + 0.1, 0.9), 0
    extra, groups, cur, k = [], [], len(Xs), 0
    for dist in DISTANCES:
        for mult in MULTIPLICITIES:
            jitter = rng.normal(size=(mult, d))
            jitter /= np.linalg.norm(jitter, axis=1, keepdims=True)
            jitter *= (EPS / 4) * rng.random((mult, 1))
            extra.append(dirs[k] * dist + jitter)
            groups.append(dict(distance=dist, rows=np.arange(cur, cur + mult)))
            cur += mult
            k += 1
    return np.vstack([Xs] + extra), groups


def blobs(rng, n=20000):
    shares = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
    widths = [0.8, 0.5, 0.6, 0.3, 0.4]
    sizes = (shares * n).astype(int)
    X = np.vstack([3.0 * np.eye(5)[i] + rng.normal(0, widths[i], (sizes[i], 5))
                   for i in range(5)])
    return StandardScaler().fit_transform(X), np.repeat(np.arange(5), sizes)


def company(rng, n=20000):
    band = rng.random(n) < 0.4
    emp = np.where(band, SIZE_MID[np.minimum(rng.geometric(0.55, n) - 1, 7)],
                   rng.lognormal(1.5, 1.2, n))
    rev = np.where(rng.random(n) < 0.35, 0.0, rng.lognormal(2.0, 1.5, n))
    founded = np.where(rng.random(n) < 0.2, 2015, 2026 - np.floor(rng.exponential(15, n)))
    cities = rng.normal(0, 4, (8, 2)) + [60, 24]
    city = rng.choice(8, n, p=rng.dirichlet(np.ones(8) * 0.7))
    spread = rng.choice([1, 4], (n, 1), p=[0.85, 0.15])
    ll = cities[city] + rng.normal(0, 0.3, (n, 2)) * spread
    X = np.c_[np.log1p(emp), np.log1p(rev), 2026 - founded, ll]
    return StandardScaler().fit_transform(X), city


def make(gen, seed):
    """Train and test sets sharing the genuine structure, with fresh anomalies in
    the test set, and a per-group linear target (anomalies get wild targets)."""
    rng = np.random.default_rng(seed)
    Xg, c = gen(rng)
    coefs = rng.normal(0, 1, (c.max() + 1, Xg.shape[1]))
    out = []
    for _ in range(2):
        X, groups = inject(Xg, rng)
        y = np.r_[np.einsum('ij,ij->i', Xg, coefs[c]) + rng.normal(0, 0.3, len(Xg)),
                  rng.normal(0, 20, len(X) - len(Xg))]
        out.append((X, y, groups))
    return out, c


def ood95(model, X_train, X):
    """The Sec. 6.5 rule: farther from the Sun than its 95th-percentile radius."""
    lab_tr = model.assign(X_train)
    d_tr = np.linalg.norm(X_train - model.suns_[lab_tr], axis=1)
    r95 = np.array([np.percentile(d_tr[lab_tr == k], 95) if np.any(lab_tr == k) else 0.0
                    for k in range(len(model.suns_))])
    lab = model.assign(X)
    return np.linalg.norm(X - model.suns_[lab], axis=1) > r95[lab]


def run(gen, seed, n_c, gamma):
    (train, test), c = make(gen, seed)
    X, y, groups = train
    X_te, y_te, _ = test
    genuine = np.arange(len(X)) < len(c)
    rows = []
    configs = [('thesis', dict(assignment='newton')),
               ('thesis+ood95', dict(assignment='newton')),
               ('pull', dict()),
               ('pull+link g5', dict(link_tau=0.6, gamma_clustering=5.0))]
    for name, kw in configs:
        t0 = time.perf_counter()
        kw = {'gamma_clustering': gamma, **kw}
        model = GACA(sample_size=n_c, random_state=seed, **kw)
        model.fit(X, y)
        labels = model.assign(X)
        flag = model.is_lone(labels)
        if name == 'thesis+ood95':
            flag = flag | ood95(model, X, X)
        pred = model.predict(X_te)
        if name == 'thesis+ood95':                     # OOD rows go to the global model
            rej = ood95(model, X, X_te)
            pred[rej] = model.global_model_.predict(X_te[rej])
        det = np.array([flag[g['rows']].all() for g in groups])
        dist = np.array([g['distance'] for g in groups])
        rows.append(dict(
            method=name, det=det.mean(),
            **{f'd{int(D)}': det[dist == D].mean() for D in DISTANCES},
            fp=flag[genuine].mean(),
            ari=adjusted_rand_score(c, labels[genuine]),
            r2=r2_score(y_te[genuine], pred[genuine]),
            t=time.perf_counter() - t0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--coreset', type=int, default=2000)
    ap.add_argument('--gamma', type=float, default=1.0)
    args = ap.parse_args()

    cols = ['det', 'd5', 'd10', 'd20', 'fp', 'ari', 'r2', 't']
    print(f"coreset {args.coreset}, gamma {args.gamma}, {args.seeds} seeds\n")
    print(f"{'data':8} {'method':13} " + " ".join(f"{c:>6}" for c in cols))
    for dname, gen in [('blobs', blobs), ('company', company)]:
        allrows = [r for s in range(args.seeds) for r in run(gen, s, args.coreset, args.gamma)]
        for m in dict.fromkeys(r['method'] for r in allrows):
            rs = [r for r in allrows if r['method'] == m]
            vals = [np.mean([r[c] for r in rs]) for c in cols]
            vals[4] *= 100                               # false positives in %
            print(f"{dname:8} {m:13} " + " ".join(f"{v:6.3f}" for v in vals))
    print("\nfp is in % of genuine rows; ARI on company is against the city, "
          "which the features only partly determine.")


if __name__ == "__main__":
    main()
