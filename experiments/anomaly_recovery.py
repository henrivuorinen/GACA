"""
Does a uniform coreset recover the anomalies GACA claims to isolate?

RQ2 asserts that GACA detaches extreme points as singleton "Lone Suns" instead
of absorbing them into the mainstream. The scalable architecture asserts that
the structure can be discovered from a uniform coreset of n_c << N points.
Section 6.3 already notes, in general terms, that a uniformly sampled coreset is
not a worst-case optimal coreset and that rare modes can be missed with
probability set by their population mass. It never connects that to the Lone Sun
claim, and it never measures the size of the effect.

The tension is concrete. Solar Genesis only ever sees the coreset. A point that
is not drawn into it cannot become a Sun of its own: it is streamed through
Phase 4 and assigned to whichever existing Sun pulls hardest. So the anomaly
isolation property should apply to anomalies that are represented in the
coreset, and not automatically to every anomaly in the dataset.

This script tests that directly by planting anomalies whose ground truth is
known, then asking how often each is recovered.

METHOD
------
Into the real standardized company data we inject nine groups of synthetic
anomalies. A group is defined by two parameters:

    distance d   how far from the data centroid the group sits, in standard
                 deviations, along a randomly chosen direction
    multiplicity m   how many near-identical copies of the anomaly exist in the
                 population (m = 1 is a unique outlier, m = 10 a small rare
                 niche)

All nine groups are injected into the same run, in mutually distinct random
directions, so that a single Solar Genesis pass tests every combination. The
copies within a group are placed within epsilon/4 of one another, so that a
group which is recovered at all should be recovered as a single Sun of exactly
m members.

For each (seed, coreset size, group) we record whether any of the group's copies
were drawn into the coreset, whether the group ends up isolated in a Sun
containing no genuine data, and whether the out-of-distribution rule of
Section 6.4 would flag it even when it was not isolated.

WHAT TO COMPARE AGAINST
-----------------------
Under uniform sampling, the probability that at least one of m copies lands in a
coreset of size n_c drawn from N points is

    P(sampled) = 1 - C(N - m, n_c) / C(N, n_c)  ~=  1 - (1 - n_c/N)^m

If the isolation rate tracks this curve, the limitation is exactly the sampling
one and nothing more; the anomaly story is then sound conditional on the coreset
and the honest statement of scope is available. If the isolation rate falls
below it, coreset membership is necessary but not sufficient and something in
the dynamics is also losing anomalies.

Usage:
    python anomaly_recovery.py            # 5 seeds x 5 coreset sizes
    python anomaly_recovery.py --quick    # 2 seeds x 3 coreset sizes
"""
import argparse
import contextlib
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from characterize_suns import load
from gaca import solar_genesis
from run_scalability_test import assign

DATA = 'data/20k_sample_data.csv'

SEEDS = [42, 43, 44, 45, 46]
CORESET_SIZES = [1000, 2000, 5000, 10000, 20000]

QUICK_SEEDS = [42, 43]
QUICK_CORESETS = [1000, 5000, 20000]

GAMMA, EPSILON, ETA, THETA, N_ITER = 1.0, 0.05, 0.5, 0.5, 20

# Group definitions: (distance in sigma, multiplicity)
DISTANCES = [5.0, 10.0, 20.0]
MULTIPLICITIES = [1, 3, 10]

OOD_PERCENTILE = 95   # matches the rejection rule of Section 6.4


def inject(Xs, rng):
    """Append the nine anomaly groups to standardized data Xs.

    Returns the augmented matrix and a list of groups, each holding the row
    indices of its copies together with its (distance, multiplicity) label.
    """
    d = Xs.shape[1]
    extra, groups = [], []
    cursor = len(Xs)

    # One direction per group, drawn on the unit sphere and kept mutually
    # separated so that two groups cannot condense into each other.
    dirs, thresh, tries = [], 0.5, 0
    while len(dirs) < len(DISTANCES) * len(MULTIPLICITIES):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        if all(np.dot(v, u) < thresh for u in dirs):  # at least 60 degrees apart
            dirs.append(v)
        tries += 1
        if tries > 500:        # relax rather than spin; 5-D has limited room
            thresh, tries = min(thresh + 0.1, 0.9), 0

    k = 0
    for dist in DISTANCES:
        for mult in MULTIPLICITIES:
            centre = dirs[k] * dist
            k += 1
            # Copies sit inside epsilon/4 of one another, so a recovered group
            # is one Sun rather than several.
            jitter = rng.normal(size=(mult, d))
            jitter /= np.linalg.norm(jitter, axis=1, keepdims=True)
            jitter *= (EPSILON / 4) * rng.random((mult, 1))
            pts = centre[None, :] + jitter
            extra.append(pts)
            groups.append({'distance': dist, 'multiplicity': mult,
                           'rows': np.arange(cursor, cursor + mult)})
            cursor += mult

    return np.vstack([Xs] + extra), groups


def run_seed(seed, Xs_real, coresets):
    rng = np.random.default_rng(seed)
    X, groups = inject(Xs_real, rng)
    n_real = len(Xs_real)
    N = len(X)
    injected = np.zeros(N, dtype=bool)
    for g in groups:
        injected[g['rows']] = True

    rows = []
    for n_c in coresets:
        core_idx = rng.choice(N, min(n_c, N), replace=False)
        in_core = np.zeros(N, dtype=bool)
        in_core[core_idx] = True

        t0 = time.perf_counter()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            suns, masses = solar_genesis(
                X[core_idx], gamma=GAMMA, n_iterations=N_ITER,
                theta=THETA, epsilon=EPSILON, eta=ETA)
            labels = assign(X, suns, masses)
        genesis_s = time.perf_counter() - t0

        # Per-Sun radius at the OOD percentile, computed over genuine data only,
        # exactly as Section 6.4 defines it over assigned training points.
        dist_to_sun = np.linalg.norm(X - suns[labels], axis=1)
        radius = {}
        for s in np.unique(labels):
            m = (labels == s) & ~injected
            radius[s] = (np.percentile(dist_to_sun[m], OOD_PERCENTILE)
                         if m.any() else 0.0)

        sizes = np.bincount(labels, minlength=len(suns))
        real_per_sun = np.bincount(labels[~injected], minlength=len(suns))

        for g in groups:
            r = g['rows']
            n_sampled = int(in_core[r].sum())
            lbls = labels[r]
            same_sun = len(np.unique(lbls)) == 1
            sun = int(lbls[0])
            # Isolated: every copy in one Sun, and that Sun holds no real data.
            isolated = bool(same_sun and real_per_sun[sun] == 0)
            ood = float(np.mean(dist_to_sun[r] > radius.get(sun, 0.0)))

            rows.append({
                'seed': seed, 'coreset': n_c, 'N': N,
                'distance': g['distance'], 'multiplicity': g['multiplicity'],
                'n_sampled': n_sampled, 'sampled': n_sampled > 0,
                'isolated': isolated, 'same_sun': same_sun,
                'sun_size': int(sizes[sun]),
                'real_in_sun': int(real_per_sun[sun]),
                'ood_flagged': ood,
                'p_sampled_theory': 1.0 - (1.0 - min(n_c, N) / N) ** g['multiplicity'],
                'n_suns': len(suns), 'genesis_s': genesis_s,
            })

        got = sum(1 for r_ in rows[-len(groups):] if r_['isolated'])
        print(f"  seed {seed} | n_c {n_c:>6} | suns {len(suns):>3} | "
              f"isolated {got}/{len(groups)} | {genesis_s:>5.1f}s")

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    seeds = QUICK_SEEDS if args.quick else SEEDS
    coresets = QUICK_CORESETS if args.quick else CORESET_SIZES

    print("=== CORESET vs ANOMALY RECOVERY ===")
    df_real, X_real = load(DATA, 50_000)
    Xs_real = StandardScaler().fit_transform(X_real)

    n_groups = len(DISTANCES) * len(MULTIPLICITIES)
    print(f"\nN = {len(Xs_real):,} real rows, d = {Xs_real.shape[1]}")
    print(f"gamma={GAMMA}, epsilon={EPSILON}, eta={ETA}, n_iter={N_ITER}")
    print(f"{n_groups} anomaly groups per run: distances {DISTANCES} sigma "
          f"x multiplicities {MULTIPLICITIES}")
    print(f"seeds={seeds}, coresets={coresets}\n")

    t0 = time.perf_counter()
    all_rows = []
    for seed in seeds:
        all_rows += run_seed(seed, Xs_real, coresets)

    allseeds = pd.DataFrame(all_rows)
    allseeds.to_csv('anomaly_recovery_allseeds.csv', index=False)

    num = ['sampled', 'isolated', 'same_sun', 'sun_size', 'real_in_sun',
           'ood_flagged', 'p_sampled_theory', 'n_suns', 'n_sampled']
    keys = ['coreset', 'distance', 'multiplicity']
    mean = allseeds.groupby(keys, as_index=False)[num].mean()
    sd = allseeds.groupby(keys)[num].std().add_suffix('_sd').reset_index()
    summary = mean.merge(sd, on=keys)
    summary['n_seeds'] = len(seeds)
    summary.to_csv('anomaly_recovery.csv', index=False)

    # --- headline: isolation rate against the sampling prediction -----------
    print(f"\n--- isolation rate vs uniform-sampling prediction "
          f"(n={len(seeds)} seeds, all distances pooled) ---")
    print(f"{'n_c':>7} | {'mult':>5} | {'P(sampled) theory':>18} | "
          f"{'sampled':>9} | {'isolated':>9} | {'OOD flagged':>12}")
    pooled = (allseeds.groupby(['coreset', 'multiplicity'], as_index=False)
              [['p_sampled_theory', 'sampled', 'isolated', 'ood_flagged']].mean())
    for _, r in pooled.iterrows():
        print(f"{int(r['coreset']):>7} | {int(r['multiplicity']):>5} | "
              f"{r['p_sampled_theory']:>18.3f} | {r['sampled']:>9.3f} | "
              f"{r['isolated']:>9.3f} | {r['ood_flagged']:>12.3f}")

    # --- is coreset membership sufficient, or only necessary? ---------------
    print("\n--- conditional on having been sampled ---")
    s = allseeds[allseeds.sampled]
    if len(s):
        print(f"  isolated | sampled            = {s.isolated.mean():.3f}  "
              f"(n = {len(s)})")
    ns = allseeds[~allseeds.sampled]
    if len(ns):
        print(f"  isolated | not sampled        = {ns.isolated.mean():.3f}  "
              f"(n = {len(ns)})")
        print(f"  OOD flagged | not sampled     = {ns.ood_flagged.mean():.3f}"
              "   <- does the rejection rule catch what the coreset missed?")

    print("\n--- by distance (does being further out help?) ---")
    for dist, b in allseeds.groupby('distance'):
        print(f"  {dist:>5.1f} sigma | sampled {b.sampled.mean():.3f} | "
              f"isolated {b.isolated.mean():.3f} | "
              f"OOD {b.ood_flagged.mean():.3f}")

    print(f"\nElapsed {time.perf_counter() - t0:.0f}s")
    print("Saved anomaly_recovery_allseeds.csv and anomaly_recovery.csv")


if __name__ == "__main__":
    main()
