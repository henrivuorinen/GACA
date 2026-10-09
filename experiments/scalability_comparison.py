"""
Scalability comparison (RQ1): GACA vs MiniBatch K-Means vs Mean-Shift vs HDBSCAN.

Runs each algorithm at an increasing ladder of N and records wall-clock time. All
algorithms cluster the SAME standardized feature space, in memory, for a fair
compute-time comparison. Mean-Shift and HDBSCAN are O(N^2); they are run up to a
cap, plus ONE probe at the first rung beyond it so that "did not finish" is a
measured outcome (a real MemoryError or a real wall-clock number, recorded in the
*_status columns) rather than an untested claim. That truncation, versus the two
linear curves, is the result.

Rows are shuffled once before the ladder is built, so each rung is a random sample
of the population rather than a prefix of the file.

Note on memory: this script compares TIME. GACA's separate O(1)-memory / out-of-core
property (peak RAM flat in N) is demonstrated by run_scalability_test.py; K-Means,
Mean-Shift and HDBSCAN must all hold the full dataset (and for the O(N^2) methods,
pairwise structures) in RAM.

Usage:
    python scalability_comparison.py [file.csv]
Produces: scalability_comparison.csv, scalability_comparison.pdf/.png
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
from sklearn.cluster import MiniBatchKMeans, MeanShift

from gaca import solar_genesis
from run_scalability_test import engineer, iter_chunks, assign

try:
    import hdbscan
    HAVE_HDBSCAN = True
except ImportError:
    HAVE_HDBSCAN = False

RANDOM_SEED = 42
GAMMA, EPS, ETA, THETA, N_ITER = 1.0, 0.05, 0.5, 0.5, 20
CORESET = 5000
N_LADDER = [10_000, 20_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
MS_CAP = 20_000      # exact Mean-Shift is O(N^2); beyond this it is impractically slow
HDB_CAP = 3_000_000    # HDBSCAN builds O(N^2) structures; cap to avoid memory blow-up
HDB_MIN_CLUSTER_SIZE = 15   # kept identical to benchmark_clustering.py

# Mean-Shift bandwidth matched to GACA's kernel scale via gamma = 1/(2 h^2), so
# both methods run at the same kernel width. It was previously hard-coded to 1.0
# while GACA ran at gamma = 1.0, i.e. h = 0.707.
MS_BANDWIDTH = float(1.0 / np.sqrt(2.0 * GAMMA))

# One attempt is made at the first ladder rung ABOVE each cap, so that "did not
# finish" is a measured outcome (a real MemoryError, or a real wall-clock number)
# rather than an untested assertion. The thesis claims HDBSCAN exhausted memory;
# without this, nothing in the code ever tested that.
PROBE_BEYOND_CAP = True


def load_features(path, max_n, chunksize=100_000):
    rows, got = [], 0
    for ch in iter_chunks(path, chunksize):
        f = engineer(ch)
        rows.append(f)
        got += len(f)
        if got >= max_n:
            break
    return np.vstack(rows)[:max_n]


def timed_fit(estimator, X):
    """Fit and return (seconds, status). Failures are recorded, not raised."""
    t = time.time()
    try:
        estimator.fit(X)
        return time.time() - t, 'ok'
    except MemoryError:
        return np.nan, f'MemoryError after {time.time() - t:.1f}s'
    except Exception as e:                                  # noqa: BLE001
        return np.nan, f'{type(e).__name__} after {time.time() - t:.1f}s: {e}'


def gaca_time(X):
    """Coreset genesis + full assignment (in-memory), returns (seconds, n_suns)."""
    rng = np.random.default_rng(RANDOM_SEED)
    core = X[rng.choice(len(X), min(CORESET, len(X)), replace=False)]
    t = time.time()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        suns, masses = solar_genesis(core, method='barnes_hut', gamma=GAMMA, n_iterations=N_ITER,
                                          theta=THETA, epsilon=EPS, eta=ETA)
        assign(X, suns, masses)
    return time.time() - t, len(suns)


def main(path):
    print(f"Loading up to {max(N_LADDER):,} rows from {path}...")
    X_all = StandardScaler().fit_transform(load_features(path, max(N_LADDER)))

    # Shuffle once before building the ladder. Without this every rung is a
    # PREFIX of the file, so a file ordered by country / id / date makes the small
    # rungs biased slices rather than samples of the same population, and the
    # apparent N-invariance of the recovered Sun count is an artefact of nesting.
    X_all = X_all[np.random.default_rng(RANDOM_SEED).permutation(len(X_all))]

    avail = len(X_all)
    ladder = [n for n in N_LADDER if n <= avail] or [avail]
    print(f"  available rows: {avail:,}; ladder: {ladder}")
    print(f"  Mean-Shift bandwidth = {MS_BANDWIDTH:.3f} (matched to gamma={GAMMA})")

    # first rung past each cap, probed once so DNF is measured rather than assumed
    ms_probe = next((n for n in ladder if n > MS_CAP), None) if PROBE_BEYOND_CAP else None
    hdb_probe = next((n for n in ladder if n > HDB_CAP), None) if PROBE_BEYOND_CAP else None

    print(f"\n{'N':>10} | {'GACA':>8} | {'MiniK-Means':>11} | {'Mean-Shift':>10} | {'HDBSCAN':>8}")
    print("-" * 62)
    rows = []
    for n in ladder:
        X = X_all[:n]

        t_gaca, K = gaca_time(X)

        t = time.time()
        MiniBatchKMeans(n_clusters=max(K, 2), random_state=RANDOM_SEED, n_init=3).fit(X)
        t_km = time.time() - t

        t_ms, ms_status = np.nan, 'not attempted (past cap)'
        if n <= MS_CAP or n == ms_probe:
            t_ms, ms_status = timed_fit(
                MeanShift(bandwidth=MS_BANDWIDTH, bin_seeding=False), X)
            if n == ms_probe:
                ms_status += ' [probe past cap]'

        t_hdb, hdb_status = np.nan, 'not attempted (past cap)'
        if HAVE_HDBSCAN and (n <= HDB_CAP or n == hdb_probe):
            t_hdb, hdb_status = timed_fit(
                hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER_SIZE,
                                core_dist_n_jobs=1), X)
            if n == hdb_probe:
                hdb_status += ' [probe past cap]'
        elif not HAVE_HDBSCAN:
            hdb_status = 'hdbscan not installed'

        rows.append(dict(N=n, gaca_s=t_gaca, minikmeans_s=t_km,
                         meanshift_s=t_ms, hdbscan_s=t_hdb, gaca_suns=K,
                         meanshift_status=ms_status, hdbscan_status=hdb_status,
                         ms_bandwidth=MS_BANDWIDTH,
                         hdb_min_cluster_size=HDB_MIN_CLUSTER_SIZE))
        fmt = lambda v: f"{v:>8.2f}" if v == v else f"{'DNF':>8}"   # NaN -> DNF
        print(f"{n:>10} | {t_gaca:>8.2f} | {t_km:>11.2f} | {fmt(t_ms):>10} | {fmt(t_hdb)}")
        if n in (ms_probe, hdb_probe):
            print(f"           probe: Mean-Shift={ms_status} | HDBSCAN={hdb_status}")

        # Write after every rung. The largest rungs can take hours, and losing a
        # completed 62-minute measurement to a Ctrl-C on the next one is not an
        # acceptable failure mode.
        pd.DataFrame(rows).to_csv('scalability_comparison.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv('scalability_comparison.csv', index=False)
    print("\nSaved scalability_comparison.csv")
    print("\nStatus of the O(N^2) baselines (cite these, do not assert a failure "
          "the code never tested):")
    for _, r in df.iterrows():
        if r['meanshift_status'] != 'not attempted (past cap)' or \
           r['hdbscan_status'] != 'not attempted (past cap)':
            print(f"  N={int(r['N']):>9,}: Mean-Shift {r['meanshift_status']}; "
                  f"HDBSCAN {r['hdbscan_status']}")

    # ---------- figure: wall-clock vs N (log-log) ----------
    BLUE, ORANGE, VERM, GREEN = '#0072B2', '#E69F00', '#D55E00', '#009E73'
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                         'grid.alpha': 0.4, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    Ns = df['N'].values

    def plot_series(col, color, marker, label):
        m = df[col].notna().values
        if m.any():
            ax.plot(Ns[m], df[col].values[m], color=color, lw=2, marker=marker,
                    ms=6, label=label)

    plot_series('gaca_s', BLUE, 'o', 'GACA (streaming, no $k$)')
    plot_series('minikmeans_s', ORANGE, '^', 'MiniBatch K-Means')
    plot_series('meanshift_s', VERM, 's', 'Mean-Shift ($O(N^2)$)')
    plot_series('hdbscan_s', GREEN, 'd', 'HDBSCAN ($O(N^2)$)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of records $N$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Clustering runtime vs. dataset size', loc='left', fontsize=11)
    ax.legend(frameon=False, loc='upper left')
    ax.text(0.98, 0.02, 'Mean-Shift / HDBSCAN: $O(N^2)$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#555555')
    fig.savefig('scalability_comparison.pdf', bbox_inches='tight')
    fig.savefig('scalability_comparison.png', dpi=200, bbox_inches='tight')
    print("Wrote scalability_comparison.pdf / .png")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/scalability_slim_10M.csv')
