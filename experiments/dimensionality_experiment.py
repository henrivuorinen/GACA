"""
Dimensionality experiment for GACA (thesis section 4.3.2, the Curse of
Dimensionality). Empirically demonstrates why the latent projection phi_d is
mandatory in high dimensions.

For a fixed sample of N rows it sweeps the number of raw feature dimensions D and
measures, at each D:

  (a) Distance concentration  -- the relative contrast (D_max - D_min)/D_min
      averaged over random query points (thesis Eq. 4.3). It collapses toward 0
      as D grows, so "near" and "far" become indistinguishable.
  (b) Barnes-Hut degradation  -- the mean fraction of GACANode tree nodes that the
      Multipole Acceptance Criterion actually visits per query. It rises toward 1
      (i.e. brute force) as bounding volumes overlap, so the O(N log N) tree
      collapses to O(N) per query.
  (c) Cluster resolution      -- the number of effective Suns GACA recovers, its
      silhouette, and the genesis wall-time. High D degrades separation.

Finally it applies PCA to reduce the full-D data back to a small latent dimension
and re-measures (a)-(c), showing the projection restores both tree efficiency and
meaningful clustering -- the empirical justification for phi_d.

Usage:
    python dimensionality_experiment.py [wide_sample.csv]
Produces: dimensionality_results.csv, dimensionality.pdf, dimensionality.png
"""
import sys
import os
import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from gaca import GACANode
from gaca import GACA

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- experiment configuration ---
N_SAMPLE = 20000          # rows used for the experiment (dimensionality, not scale)
CORESET = 5000            # genesis coreset size
GAMMA, EPS, ETA, THETA, N_ITER = 1.0, 0.05, 0.5, 0.5, 20
DELTA_MAC = 1e-9
D_LIST = [2, 5, 10, 15, 20, 30, 40, 50]   # capped to available columns at runtime
PCA_TARGET = 5
MIN_COVERAGE = 0.10   # drop numeric columns sparser than this (dead once imputed)

# heavy-tailed non-negative columns to log1p-compress
LOG_HINT = {
    'EMPLOYEES_COUNT', 'FOLLOWERS', 'PHYSICAL_LOCATIONS', 'GMAPS_REVIEW_COUNT',
    'TOTAL_GMAPS_REVIEW_COUNT', 'GMAPS_REVIEWS_ONE_STAR', 'GMAPS_REVIEWS_TWO_STAR',
    'GMAPS_REVIEWS_THREE_STAR', 'GMAPS_REVIEWS_FOUR_STAR', 'GMAPS_REVIEWS_FIVE_STAR',
    'REVENUE', 'REVENUE_ESTIMATES', 'TOTAL_FUNDING', 'NUM_FUNDING_ROUNDS',
    'NUM_INVESTMENTS', 'LATEST_VISITORS', 'EMPLOYEE_COUNTRY_COUNT',
}
# informative continuous features placed first, so small D is meaningful; the
# remaining (sparser count) features fill in as D grows
CORE_FIRST = ['REVENUE', 'REVENUE_ESTIMATES', 'EMPLOYEES_COUNT',
              'EMPLOYEE_COUNTRY_COUNT', 'FOLLOWERS', 'HEADCOUNT_GROWTH_12MONTH',
              'TOTAL_FUNDING', 'LATEST_VISITORS', 'COMPANY_AGE',
              'LATITUDE', 'LONGITUDE',
              'TOTAL_GMAPS_REVIEW_COUNT', 'GMAPS_REVIEWS_AVERAGE']
# numeric columns that are identifiers / metadata, not features (plus any *_ID)
DROP_COLS = {'PRIMARY_SOURCE_ID', 'SOURCE_PRIORITY', 'RANK_WITHIN_SOURCE'}


def load_numeric(path):
    df = pd.read_csv(path, low_memory=False)
    # coerce numeric-looking string columns
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().mean() > 0.5:
                df[c] = coerced
    num = df.select_dtypes(include=[np.number]).copy()
    # drop identifiers / metadata and any *_ID column (e.g. COMPANY_ID)
    drop = [c for c in num.columns
            if c in DROP_COLS or c.upper().endswith('_ID') or c.upper() == 'ID']
    num = num.drop(columns=drop, errors='ignore')
    # drop ultra-sparse columns (dead once imputed) and constant columns
    cov = num.notna().mean()
    num = num[[c for c in num.columns if cov[c] >= MIN_COVERAGE]]
    num = num.loc[:, num.nunique(dropna=True) > 1]       # drop constant
    # order: informative core first, then the remainder in file order
    cols = [c for c in CORE_FIRST if c in num.columns]
    cols += [c for c in num.columns if c not in cols]
    num = num[cols]
    # log-compress heavy-tailed non-negative columns
    for c in num.columns:
        col = num[c]
        med = col.median(skipna=True)
        skewed = col.min(skipna=True) >= 0 and med is not None and \
            col.quantile(0.99) > 10 * max(med, 1e-9)
        if c in LOG_HINT or skewed:
            num[c] = np.log1p(col.clip(lower=0))
    num = num.fillna(num.median(numeric_only=True))
    return num.values.astype(float), list(num.columns)


def distance_cv(X, n_pairs=40000, rng=None):
    """Coefficient of variation of squared pairwise distances, CV = std/mean.

    This is the distance-concentration measure of thesis Proposition 3.1
    (CV of R^2 = sqrt(2/d) for isotropic Gaussian data), which vanishes as the
    dimensionality grows. Unlike a (D_max - D_min)/D_min ratio it is robust to
    near-duplicate points, which are common in real data and would otherwise send
    D_min to zero and the ratio to infinity.
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    n = len(X)
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)
    keep = a != b
    r2 = np.sum((X[a[keep]] - X[b[keep]]) ** 2, axis=1)
    m = r2.mean()
    return float(r2.std() / m) if m > 0 else 0.0


def count_nodes(node):
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def _count_visit(p, q, node, ctr, theta, delta):
    if node is None:
        return
    ctr[0] += 1
    if node.is_leaf():
        return
    dist = np.sqrt(np.sum((p - node.com) ** 2))
    accept = (q not in node.index_set) and (node.radius / max(dist, delta) < theta)
    if accept:
        return
    _count_visit(p, q, node.left, ctr, theta, delta)
    _count_visit(p, q, node.right, ctr, theta, delta)


def tree_visit_fraction(Xc, n_q=500, rng=None):
    rng = rng or np.random.default_rng(RANDOM_SEED)
    n = len(Xc)
    root = GACANode(Xc, np.ones(n), np.arange(n), leaf_size=1)
    total = count_nodes(root)
    fracs = []
    for i in rng.choice(n, min(n_q, n), replace=False):
        ctr = [0]
        _count_visit(Xc[i], int(i), root, ctr, THETA, DELTA_MAC)
        fracs.append(ctr[0] / total)
    return float(np.mean(fracs))


def cluster_metrics(X):
    gaca = GACA(gamma_clustering=GAMMA, n_iterations=N_ITER, epsilon=EPS,
                               theta=THETA, sample_size=CORESET, eta=ETA,
                               random_state=RANDOM_SEED)
    t = time.time()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        gaca.fit(X)
        labels = gaca._assign_to_suns(X)
    genesis_s = time.time() - t
    uniq, counts = np.unique(labels, return_counts=True)
    eff, lone = len(uniq), int(np.sum(counts == 1))
    if eff > 1:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(X), min(5000, len(X)), replace=False)
        sil = float(silhouette_score(X[idx], labels[idx]))
    else:
        sil = float('nan')
    return len(gaca.suns_), eff, lone, sil, genesis_s


def run_one(Xstd, label, coreset_idx):
    cv = distance_cv(Xstd)
    tvf = tree_visit_fraction(Xstd[coreset_idx])
    n_suns, eff, lone, sil, gs = cluster_metrics(Xstd)
    return dict(label=label, D=Xstd.shape[1], dist_cv=cv, tree_visit_frac=tvf,
                n_suns=n_suns, effective=eff, lone=lone, silhouette=sil, genesis_s=gs)


def main(path):
    X, cols = load_numeric(path)
    print(f"Loaded {X.shape[0]} rows x {X.shape[1]} numeric feature columns")
    print("Columns (in D-order):", cols)

    rng = np.random.default_rng(RANDOM_SEED)
    if len(X) > N_SAMPLE:
        X = X[rng.choice(len(X), N_SAMPLE, replace=False)]
    # Reproduce exactly the draw GACA.fit makes, so the tree whose
    # visit fraction is measured is the tree Solar Genesis actually builds. Drawing
    # it from the already-advanced `rng` gave a different subset from the one the
    # model used, so panels (b) and (c) described different coresets.
    coreset_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(X), min(CORESET, len(X)), replace=False)

    d_max = X.shape[1]
    d_list = sorted(set([d for d in D_LIST if d <= d_max] + [d_max]))

    print(f"\n{'label':<22} | {'D':>3} | {'dist_cv':>8} | {'treevisit':>9} | "
          f"{'eff':>3} | {'lone':>4} | {'silhou':>7} | {'genesis':>8}")
    print("-" * 86)
    results = []
    for D in d_list:
        Xd = StandardScaler().fit_transform(X[:, :D])
        r = run_one(Xd, f"D={D}", coreset_idx)
        results.append(r)
        print(f"{r['label']:<22} | {r['D']:>3} | {r['dist_cv']:>8.3f} | "
              f"{r['tree_visit_frac']:>9.3f} | {r['effective']:>3} | {r['lone']:>4} | "
              f"{r['silhouette']:>7.3f} | {r['genesis_s']:>7.2f}s")

    # PCA rescue from the full dimensionality
    Xmax = StandardScaler().fit_transform(X[:, :d_max])
    Xp = PCA(n_components=min(PCA_TARGET, d_max), random_state=RANDOM_SEED).fit_transform(Xmax)
    Xp = StandardScaler().fit_transform(Xp)
    rp = run_one(Xp, f"PCA(d={Xp.shape[1]})", coreset_idx)
    results.append(rp)
    print(f"{rp['label']:<22} | {rp['D']:>3} | {rp['dist_cv']:>8.3f} | "
          f"{rp['tree_visit_frac']:>9.3f} | {rp['effective']:>3} | {rp['lone']:>4} | "
          f"{rp['silhouette']:>7.3f} | {rp['genesis_s']:>7.2f}s   <- PCA rescue from D={d_max}")

    pd.DataFrame(results).to_csv('dimensionality_results.csv', index=False)
    print("\nSaved dimensionality_results.csv")

    # ---------- figure ----------
    sweep = [r for r in results if r['label'].startswith('D=')]
    Ds = [r['D'] for r in sweep]
    BLUE, GREEN, GREY = '#0072B2', '#009E73', '#555555'
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                         'grid.alpha': 0.4, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(6.2, 7.6), sharex=True,
                                     gridspec_kw={'hspace': 0.15})
    a1.plot(Ds, [r['genesis_s'] for r in sweep], color=BLUE, lw=2, marker='o', ms=6)
    a1.set_yscale('log')
    a1.set_ylabel('Genesis time (s)')
    a1.set_title('(a) Runtime explodes', loc='left', fontsize=10, pad=4)
    a2.plot(Ds, [r['tree_visit_frac'] for r in sweep], color=BLUE, lw=2, marker='s', ms=6)
    a2.set_ylabel('Fraction of tree\nnodes visited')
    a2.set_title('(b) Barnes--Hut degradation', loc='left', fontsize=10, pad=4)
    a3.plot(Ds, [r['silhouette'] for r in sweep], color=BLUE, lw=2, marker='^', ms=7)
    a3.set_ylabel('Silhouette')
    a3.set_title('(c) Cluster cohesion', loc='left', fontsize=10, pad=4)
    a3.set_xlabel('Number of raw dimensions $D$')
    a3.axhline(0, color=GREY, lw=0.8)      # silhouette = 0: no structure
    # PCA-rescue reference lines (green dotted) on all three panels
    a1.axhline(rp['genesis_s'], color=GREEN, ls=':', lw=1.5)
    a2.axhline(rp['tree_visit_frac'], color=GREEN, ls=':', lw=1.5)
    a3.axhline(rp['silhouette'], color=GREEN, ls=':', lw=1.5)
    a1.text(Ds[-1], rp['genesis_s'], 'PCA rescue ', color=GREEN, va='bottom',
            ha='right', fontsize=8)
    for ax in (a1, a2, a3):        # theoretical D~15 degradation threshold
        ax.axvline(15, color=GREY, ls='--', lw=1)
    fig.savefig('dimensionality.pdf', bbox_inches='tight')
    fig.savefig('dimensionality.png', dpi=200, bbox_inches='tight')
    print("Wrote dimensionality.pdf / dimensionality.png")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/200k_wide_sample.csv')
