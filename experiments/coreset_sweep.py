"""
Coreset-size sensitivity experiment for GACA (justifies the coreset hyperparameter).

The Solar Genesis simulation runs on a random coreset of size n_c; n_c is a design
choice, not something the algorithm derives. This experiment sweeps n_c at fixed
data and shows the two things needed to justify the choice:

  1. Genesis COST grows with n_c (this is the price of a larger coreset, and the
     reason the coreset exists at all).
  2. The discovered structure CONVERGES: the number of Suns and, most importantly,
     the largest-Sun mass share on a fixed held-out evaluation set flatten out, so
     beyond a modest n_c a larger coreset buys almost nothing. That is the empirical
     justification that a fixed n_c (e.g. 5000) is "enough".

A fixed evaluation sample (disjoint from every coreset) is assigned with each
coreset's Suns, so the topology metrics are directly comparable across n_c.

Usage:
    python coreset_sweep.py [file.csv] [chunksize]
Produces: coreset_results.csv, coreset_sweep.pdf/.png
"""
import sys
import time
import contextlib
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from gaca import solar_genesis
from run_scalability_test import engineer, iter_chunks, assign  # reuse streaming helpers

RANDOM_SEED = 42
SEEDS = [42, 43, 44]   # each seed redraws the pool, the eval set and every coreset
GAMMA, EPS, ETA, THETA, N_ITER = 1.0, 0.05, 0.5, 0.5, 20
CORESET_SIZES = [1000, 2000, 5000, 10000, 20000, 40000]   # edit to taste
EVAL = 50000        # fixed held-out evaluation sample, disjoint from the coresets
SIL_SAMPLE = 5000   # silhouette is O(n^2), so it is measured on a fixed sample


def run_seed(seed, path, chunksize, n_total):
    """One full sweep of CORESET_SIZES at a given seed."""
    rng = np.random.default_rng(seed)

    # Proportional random sample of a pool big enough for the largest coreset plus
    # the evaluation set (streamed, low memory).
    need = max(CORESET_SIZES) + EVAL
    p = min(1.0, need * 1.3 / n_total)   # 1.3x headroom for sampling variance
    print(f"  seed {seed}: sampling a pool of ~{need:,} rows (p={p:.4f})...")
    pool = []
    for ch in iter_chunks(path, chunksize):
        feats = engineer(ch)
        mask = rng.random(len(feats)) < p
        if mask.any():
            pool.append(feats[mask])
    pool = np.vstack(pool)
    pool = pool[rng.permutation(len(pool))]

    eval_raw = pool[:EVAL]
    coreset_pool = pool[EVAL:]
    sizes = [k for k in CORESET_SIZES if k <= len(coreset_pool)]
    if not sizes:
        sizes = [len(coreset_pool)]

    sil_idx = rng.choice(len(eval_raw), min(SIL_SAMPLE, len(eval_raw)), replace=False)

    rows = []
    for k in sizes:
        # Each coreset is an INDEPENDENT draw from the pool. Taking nested prefixes
        # (coreset_pool[:k]) makes the k=1000 sample a subset of the k=40000 one, so
        # the sweep understates exactly the sampling variability it is meant to bound.
        core_raw = coreset_pool[rng.choice(len(coreset_pool), k, replace=False)]

        # The scaler is fitted on the CORESET only, matching thesis section 4.1
        # ("estimated from the training or coreset construction data"). Fitting it
        # on the pool, which contains the eval set, leaked eval statistics into the
        # standardisation.
        scaler = StandardScaler().fit(core_raw)
        core = scaler.transform(core_raw)
        eval_set = scaler.transform(eval_raw)

        t0 = time.time()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            suns, masses, n_iters = solar_genesis(
                core, method='barnes_hut', gamma=GAMMA, n_iterations=N_ITER, theta=THETA, epsilon=EPS,
                eta=ETA, return_iters=True)
        gen_s = time.time() - t0
        labels = assign(eval_set, suns, masses)
        counts = np.bincount(labels, minlength=len(suns))
        n_suns = len(suns)
        eff = int((counts > 0).sum())
        largest_share = counts.max() / counts.sum()
        sil = (float(silhouette_score(eval_set[sil_idx], labels[sil_idx]))
               if len(np.unique(labels[sil_idx])) > 1 else float('nan'))
        rows.append(dict(seed=seed, coreset=k, genesis_s=gen_s, suns=n_suns,
                         effective=eff, largest_share=largest_share, silhouette=sil,
                         genesis_iters=n_iters))
        print(f"  seed {seed} | {k:>6} | {gen_s:>8.2f}s | suns {n_suns:>3} | "
              f"eff {eff:>3} | share {largest_share:>6.4f} | sil {sil:>6.3f}")
    return rows


def main(path, chunksize=50000):
    # Pass 1: count usable rows (shared by every seed).
    print(f"Pass 1: counting rows in {path}...")
    n_total = sum(len(engineer(ch)) for ch in iter_chunks(path, chunksize))
    print(f"  usable rows N = {n_total:,}")

    print(f"\nSweeping coreset sizes {CORESET_SIZES} over seeds {SEEDS}")
    all_rows = []
    for seed in SEEDS:
        all_rows += run_seed(seed, path, chunksize, n_total)

    allseeds = pd.DataFrame(all_rows)
    allseeds.to_csv('coreset_results_allseeds.csv', index=False)

    num = ['genesis_s', 'suns', 'effective', 'largest_share', 'silhouette',
           'genesis_iters']
    mean = allseeds.groupby('coreset', as_index=False)[num].mean()
    sd = allseeds.groupby('coreset')[num].std().add_suffix('_sd').reset_index()
    results_df = mean.merge(sd, on='coreset')
    results_df['n_seeds'] = len(SEEDS)
    results_df.to_csv('coreset_results.csv', index=False)

    print(f"\n--- seed-averaged (n={len(SEEDS)}), mean +/- sd ---")
    print(f"{'coreset':>8} | {'genesis_s':>16} | {'suns':>12} | {'eff':>12} | "
          f"{'largest_share':>17} | {'silhouette':>15}")
    for _, r in results_df.iterrows():
        print(f"{int(r['coreset']):>8} | {r['genesis_s']:>7.2f}+-{r['genesis_s_sd']:<7.2f} | "
              f"{r['suns']:>5.1f}+-{r['suns_sd']:<5.1f} | "
              f"{r['effective']:>5.1f}+-{r['effective_sd']:<5.1f} | "
              f"{r['largest_share']:>8.4f}+-{r['largest_share_sd']:<7.4f} | "
              f"{r['silhouette']:>7.3f}+-{r['silhouette_sd']:<6.3f}")
    print("\nSaved coreset_results.csv and coreset_results_allseeds.csv")
    results = results_df.to_dict('records')

    # ---------- figure: cost vs. convergence (mean +/- sd over seeds) ----------
    ks = [int(r['coreset']) for r in results]
    BLUE, GREEN, GREY = '#0072B2', '#009E73', '#555555'
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                         'grid.alpha': 0.4, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(6.2, 7.4), sharex=True,
                                     gridspec_kw={'hspace': 0.15})

    def series(col):
        return ([r[col] for r in results], [r[f'{col}_sd'] for r in results])

    for ax, col, color, marker, ylab, title in (
            (a1, 'genesis_s', BLUE, 'o', 'Genesis time (s)',
             '(a) Cost grows with coreset size'),
            (a2, 'largest_share', GREEN, 's', 'Largest-Sun\nmass share',
             '(b) Structure converges'),
            (a3, 'effective', BLUE, '^', 'Effective Suns',
             '(c) Effective-Sun count converges')):
        vals, errs = series(col)
        ax.errorbar(ks, vals, yerr=errs, color=color, lw=2, marker=marker, ms=6,
                    capsize=3, elinewidth=1)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc='left', fontsize=10, pad=4)

    a3.set_xlabel(f'Coreset size $n_c$  (mean $\\pm$ sd over {len(SEEDS)} seeds)')
    a3.set_xscale('log')
    a3.set_xticks(ks)
    a3.set_xticklabels([f'{k // 1000}k' if k >= 1000 else str(k) for k in ks])
    fig.savefig('coreset_sweep.pdf', bbox_inches='tight')
    fig.savefig('coreset_sweep.png', dpi=200, bbox_inches='tight')
    print("Wrote coreset_sweep.pdf / .png")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/scalability_slim_1M.csv'
    cs = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    main(path, cs)
