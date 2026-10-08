"""
Out-of-core scalability benchmark for GACA (Chapter 6 / RQ1).

Genuinely streams the dataset from disk in fixed-size chunks so that peak memory
is bounded by the chunk size, not by N. A single streaming pass yields all three
RQ1 deliverables:

  1. Linear runtime      -- cumulative accretion time vs. cumulative N (the
                            Solar Genesis cost is a fixed constant on the coreset).
  2. Constant memory     -- peak RSS during accretion, independent of N.
  3. Topology stability  -- the largest-Sun mass share and effective-Sun count
                            recorded at cumulative-N checkpoints, showing that the
                            cluster proportions converge and are N-invariant (so a
                            1M run and a 100M run describe the same structure).

Passes:
  A. stream once to count usable rows N and reservoir-free proportional coreset;
  B. fit preprocessing + run Solar Genesis on the coreset -> Suns;
  C. stream again, assign each chunk (O(1) memory), timing + tallying at checkpoints.

Usage:
    python run_scalability_test.py <file.csv> [chunksize]
Produces: scalability_results.csv, scalability_plot.pdf/.png
"""
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from gaca import solar_genesis
from process_data import (SURFACE_FEATURES, SIZE_MAP, REFERENCE_YEAR,
                          FOUNDED_FALLBACK)

RANDOM_SEED = 42
CORESET = 5000
GAMMA, EPS, ETA, THETA, N_ITER = 1.0, 0.05, 0.5, 0.5, 20
FEATURES = SURFACE_FEATURES


def _rss_mb():
    """Resident set size in MB.

    tracemalloc was used here previously, but it only traces Python's own
    allocator: the NumPy and pandas blocks that actually dominate a streaming
    batch are invisible to it, so the number it reported was not peak memory even
    though the docstring called it RSS.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kB
        return peak / 1e6 if sys.platform == 'darwin' else peak / 1e3


def engineer(df):
    """Chunk-local feature engineering, mirroring load_and_clean_data. Returns the
    FEATURES matrix for rows with valid coordinates (no global statistics needed,
    so it is safe to apply per chunk).

    The FOUNDED fallback and reference year come from process_data so that the
    streaming pipeline of Chapter 6 derives identical features to the in-memory
    pipeline of Chapter 5.
    """
    founded = pd.to_numeric(df.get('FOUNDED'), errors='coerce').fillna(FOUNDED_FALLBACK)
    age = (REFERENCE_YEAR - founded).clip(lower=0)
    emp = pd.to_numeric(df.get('EMPLOYEES_COUNT'), errors='coerce')
    emp = emp.fillna(df.get('SIZE').map(SIZE_MAP) if 'SIZE' in df else np.nan).fillna(1)
    log_emp = np.log1p(emp.clip(lower=0))
    rev = pd.to_numeric(df.get('TOTAL_GMAPS_REVIEW_COUNT'), errors='coerce').fillna(0)
    log_reviews = np.log1p(rev.clip(lower=0))
    lat = pd.to_numeric(df.get('LATITUDE'), errors='coerce')
    lon = pd.to_numeric(df.get('LONGITUDE'), errors='coerce')
    out = pd.DataFrame({'log_emp': log_emp, 'log_reviews': log_reviews, 'age': age,
                        'LATITUDE': lat, 'LONGITUDE': lon}).dropna(subset=['LATITUDE', 'LONGITUDE'])
    return out[FEATURES].values.astype(float)


def iter_chunks(path, chunksize):
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    else:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            yield chunk


def assign(X, suns, masses):
    """Newtonian nearest-well assignment (Phase 4), vectorised."""
    d2 = np.maximum(np.sum(X**2, 1)[:, None] + np.sum(suns**2, 1)[None, :]
                    - 2 * X @ suns.T, 1e-9)
    return np.argmax(masses / d2, axis=1)


def main(path, chunksize=50000):
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Pass A + B: count N and draw a uniform coreset in one stream ---
    #
    # Exact reservoir sampling (Vitter's Algorithm R): after n rows have been seen
    # every row is in the reservoir with probability CORESET/n, independent of
    # where in the file it appeared.
    #
    # The previous scheme buffered up to 4*CORESET rows and then subsampled down to
    # CORESET, repeatedly. That gives a surviving row only a 1-in-4 chance per
    # compaction, and a 10M-row file triggers hundreds of compactions, so a row
    # from the first chunk survived with probability ~(1/4)^n. The "coreset" was in
    # practice a sample of the last few tens of thousands of rows of the file. Any
    # ordering in the file (by country, id, insertion date) therefore leaked
    # straight into the Suns, which is fatal for the N-invariance claim this
    # experiment exists to support.
    print(f"Pass A/B: streaming {path} to count rows and sample coreset...")
    n_total = 0
    reservoir = None
    n_replacements = 0
    for chunk in iter_chunks(path, chunksize):
        feats = engineer(chunk)
        if not len(feats):
            continue
        if reservoir is None:
            take = min(CORESET, len(feats))
            reservoir = feats[:take].copy()
            n_total += take
            feats = feats[take:]
        if len(feats):
            # row i of this block is global row (n_total + i + 1)
            probs = CORESET / np.arange(n_total + 1, n_total + len(feats) + 1)
            accepted = np.flatnonzero(rng.random(len(feats)) < probs)
            for i in accepted:
                reservoir[rng.integers(len(reservoir))] = feats[i]
            n_replacements += len(accepted)
            n_total += len(feats)
    coreset_raw = reservoir
    print(f"  usable rows N = {n_total:,}; coreset = {len(coreset_raw)} "
          f"(uniform; {n_replacements:,} reservoir replacements)")

    scaler = StandardScaler().fit(coreset_raw)
    print("Solar Genesis on coreset...")
    t_gen = time.time()
    suns, masses, n_iters = solar_genesis(
        scaler.transform(coreset_raw), method='barnes_hut', gamma=GAMMA, n_iterations=N_ITER,
        theta=THETA, epsilon=EPS, eta=ETA, return_iters=True)
    genesis_s = time.time() - t_gen
    K = len(suns)
    print(f"  {K} Suns in {n_iters} iterations ({genesis_s:.2f}s).")

    # --- Pass C: stream + assign, timing/memory/topology at checkpoints ---
    #
    # t0 is outside the loop and iter_chunks is a generator, so the reported
    # accretion time INCLUDES reading and parsing the file, not just the assignment
    # arithmetic. That is the honest end-to-end number and the thesis should say so.
    print("Pass C: streaming accretion...")
    counts = np.zeros(K)
    cum_n = 0
    rows = []
    rss_baseline = _rss_mb()
    peak_rss = rss_baseline
    t0 = time.time()
    for chunk in iter_chunks(path, chunksize):
        X = scaler.transform(engineer(chunk))
        labels = assign(X, suns, masses)
        counts += np.bincount(labels, minlength=K)
        cum_n += len(X)
        elapsed = time.time() - t0
        peak_rss = max(peak_rss, _rss_mb())
        eff = int(np.sum(counts > 0))
        largest_share = counts.max() / counts.sum()
        rows.append(dict(N=cum_n, accretion_s=elapsed, peak_mb=peak_rss,
                         effective_suns=eff, largest_share=largest_share,
                         genesis_s=genesis_s, genesis_iters=n_iters,
                         suns=K, chunksize=chunksize))
    df = pd.DataFrame(rows)
    df.to_csv('scalability_results.csv', index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nGenesis (fixed): coreset {len(coreset_raw)} -> {K} Suns, "
          f"{n_iters} iterations, {genesis_s:.2f}s")
    print(f"Peak process RSS during accretion: {df['peak_mb'].max():.1f} MB "
          f"(baseline before streaming was {rss_baseline:.1f} MB; should be flat in N)")
    print(f"Accretion (incl. CSV parse): {df['accretion_s'].iloc[-1]:.2f}s for "
          f"{cum_n:,} rows = {df['accretion_s'].iloc[-1] / cum_n * 1e6:.3f} us/row")
    print(f"Largest-Sun share: {df['largest_share'].iloc[0]:.3f} -> "
          f"{df['largest_share'].iloc[-1]:.3f} (should converge)")

    # --- figure: linear-time + memory + topology stability ---
    BLUE, GREEN, GREY = '#0072B2', '#009E73', '#555555'
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                         'grid.alpha': 0.4, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True,
                                 gridspec_kw={'hspace': 0.15})
    a1.plot(df['N'], df['accretion_s'], color=BLUE, lw=2, marker='o', ms=5, label='Measured')
    # linear reference through the last point
    a1.plot(df['N'], df['accretion_s'].iloc[-1] * df['N'] / df['N'].iloc[-1],
            color=GREY, ls='--', lw=1, label='Linear $O(N)$')
    a1.set_ylabel('Cumulative accretion\ntime (s)')
    a1.set_title('(a) Runtime scales linearly', loc='left', fontsize=10, pad=4)
    a1.legend(frameon=False, loc='upper left')
    a2.plot(df['N'], df['largest_share'], color=GREEN, lw=2, marker='^', ms=6)
    a2.set_ylabel('Largest-Sun\nmass share')
    a2.set_title('(b) Cluster topology is N-invariant', loc='left', fontsize=10, pad=4)
    a2.set_xlabel('Records processed $N$')
    fig.savefig('scalability_plot.pdf', bbox_inches='tight')
    fig.savefig('scalability_plot.png', dpi=200, bbox_inches='tight')
    print("Wrote scalability_plot.pdf / .png")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/scalability_slim_1M.csv'
    cs = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    main(path, cs)
