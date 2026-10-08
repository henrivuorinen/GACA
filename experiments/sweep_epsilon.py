"""
Condensation-radius (epsilon) sweep for GACA.

Thesis section 5.1 sweeps the bandwidth gamma over eight values and names three
regimes, but epsilon is held at 0.05 throughout with no justification on record.
GACA is a two-parameter system presented as a one-parameter one, and a reader is
entitled to ask what the second parameter does. This script answers that.

The grid is (gamma, epsilon) rather than epsilon alone, because the two are
coupled: Corollary 3.2 gives the no-merge condition s_t - 2*Delta_t > epsilon
with Delta_t carrying exp(-gamma * s_t^2), so epsilon sets when particles count
as having met and gamma sets how fast they approach. Sweeping epsilon at a
single gamma could not distinguish "epsilon does nothing" from "epsilon does
nothing at this particular gamma".

Three gammas are used, one per regime identified in section 5.1:
    0.5  equilibrium band, low end
    1.0  the operating point adopted for every benchmark in the thesis
    2.0  onset of high-entropy fragmentation

Hypothesis under test: gamma decides what structure exists, epsilon only decides
the resolution at which it is recorded. If that is right, the Sun count should
fall as epsilon grows (coarser recording) while the largest-Sun share and the
downstream R2 stay flat. If instead R2 or the largest share move with epsilon,
the thesis needs to sweep it properly rather than assert a fixed value.

Everything is repeated over SEEDS and reported as mean +/- sd, matching
sweep_gamma.py. Two CSVs are written:

    eps_sweep_allseeds.csv   one row per (seed, gamma, epsilon)
    eps_sweep.csv            seed-averaged, plus *_sd columns

The MoE scorer and the stranded-row metric are imported from sweep_gamma rather
than reimplemented, so Table 5.1 and this table are directly comparable.

Usage:
    python sweep_epsilon.py            # full grid, 3 x 5 x 3 = 45 genesis runs
    python sweep_epsilon.py --quick    # 2 x 3 x 2 = 12 runs, for a smoke test
"""
import argparse
import contextlib
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from process_data import load_and_clean_data, SURFACE_FEATURES, TARGET
from gaca import GACA
# Same scorer as the bandwidth sweep, so the two tables can be read together.
from sweep_gamma import moe_r2, stranded_fraction, MIN_EXPERT, SIL_SAMPLE

DATA = 'data/20k_sample_data.csv'

SEEDS = [42, 43, 44]
GAMMAS = [0.5, 1.0, 2.0]
EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.20]

CORESET = 5000
N_ITER = 20
ETA = 0.5

QUICK_SEEDS = [42, 43]
QUICK_GAMMAS = [1.0, 2.0]
QUICK_EPSILONS = [0.02, 0.05, 0.20]


def run_seed(seed, X, y, gammas, epsilons):
    rng = np.random.default_rng(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fixed across the whole grid for this seed, so silhouettes are comparable
    # between cells and not just within a column.
    sil_idx = rng.choice(len(X_train), min(SIL_SAMPLE, len(X_train)),
                         replace=False)

    global_r2 = r2_score(
        y_test, Ridge(alpha=1.0).fit(X_train, y_train).predict(X_test))

    rows = []
    for gamma in gammas:
        for eps in epsilons:
            gaca = GACA(assignment='newton', method='barnes_hut', gamma_clustering=gamma,
                                       n_iterations=N_ITER,
                                       epsilon=eps, sample_size=CORESET,
                                       random_state=seed, eta=ETA,
                                       min_expert_size=MIN_EXPERT)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                gaca.fit(X_train)
                train_labels = gaca._assign_to_suns(X_train)
                test_labels = gaca._assign_to_suns(X_test)
            genesis_s = time.perf_counter() - t0

            suns_found = len(gaca.suns_)
            uniq, counts = np.unique(train_labels, return_counts=True)
            effective = len(uniq)
            lone = int(np.sum(counts == 1))
            largest = int(counts.max())
            median = float(np.median(counts))
            largest_share = float(largest / counts.sum())
            # How hard condensation worked: coreset points per surviving Sun.
            compression = float(CORESET / suns_found) if suns_found else np.nan

            sub = train_labels[sil_idx]
            sil = (float(silhouette_score(X_train[sil_idx], sub))
                   if len(np.unique(sub)) > 1 else float('nan'))

            gaca_r2 = moe_r2(train_labels, test_labels,
                            X_train, y_train, X_test, y_test)

            if effective > 1:
                km = KMeans(n_clusters=effective, random_state=seed, n_init=10)
                km_r2 = moe_r2(km.fit_predict(X_train), km.predict(X_test),
                               X_train, y_train, X_test, y_test)
            else:
                km_r2 = float('nan')

            rows.append({
                'seed': seed, 'gamma': gamma, 'epsilon': eps,
                'suns_found': suns_found, 'effective': effective, 'lone': lone,
                'largest': largest, 'largest_share': largest_share,
                'median': median, 'compression': compression,
                'silhouette': sil, 'gaca_r2': gaca_r2, 'km_r2': km_r2,
                'global_r2': global_r2, 'genesis_s': genesis_s,
                'n_iters': gaca.n_iters_,
                'stranded_frac': stranded_fraction(train_labels, test_labels),
                'eta': ETA, 'coreset': CORESET,
            })
            print(f"  seed {seed} | gamma {gamma:<4} eps {eps:<5} | "
                  f"suns {suns_found:<3} eff {effective:<3} lone {lone:<3} | "
                  f"share {largest_share:>5.3f} | sil {sil:>6.3f} | "
                  f"GACA {gaca_r2:>7.4f} | KM {km_r2:>7.4f} | "
                  f"{genesis_s:>5.1f}s")
    return rows


def report(summary, gammas, epsilons):
    """Print one block per gamma: epsilon down the rows."""
    for gamma in gammas:
        block = summary[summary.gamma == gamma]
        print(f"\n--- gamma = {gamma} ---")
        print(f"{'eps':>6} | {'suns':>12} | {'effective':>12} | {'lone':>11} | "
              f"{'largest share':>15} | {'silhouette':>15} | {'GACA R2':>16}")
        for _, r in block.iterrows():
            print(f"{r['epsilon']:>6.2f} | "
                  f"{r['suns_found']:>6.1f}+-{r['suns_found_sd']:<5.1f} | "
                  f"{r['effective']:>6.1f}+-{r['effective_sd']:<5.1f} | "
                  f"{r['lone']:>5.1f}+-{r['lone_sd']:<5.1f} | "
                  f"{r['largest_share']:>7.3f}+-{r['largest_share_sd']:<7.3f} | "
                  f"{r['silhouette']:>7.3f}+-{r['silhouette_sd']:<7.3f} | "
                  f"{r['gaca_r2']:>7.4f}+-{r['gaca_r2_sd']:<7.4f}")

    # The question the thesis actually needs answered: across the epsilon range,
    # how much does each quantity move relative to how much it moves with gamma?
    print("\n--- range across epsilon, per gamma (max - min of the seed means) ---")
    print(f"{'gamma':>6} | {'suns':>9} | {'effective':>10} | "
          f"{'largest share':>14} | {'GACA R2':>10}")
    for gamma in gammas:
        b = summary[summary.gamma == gamma]
        print(f"{gamma:>6.2f} | {b.suns_found.max() - b.suns_found.min():>9.1f} | "
              f"{b.effective.max() - b.effective.min():>10.1f} | "
              f"{b.largest_share.max() - b.largest_share.min():>14.4f} | "
              f"{b.gaca_r2.max() - b.gaca_r2.min():>10.4f}")

    g = summary.groupby('gamma')[['suns_found', 'gaca_r2']].mean()
    if len(g) > 1:
        print(f"\nFor contrast, moving gamma from {g.index.min()} to "
              f"{g.index.max()} changes mean suns by "
              f"{g.suns_found.max() - g.suns_found.min():.1f} and mean GACA R2 "
              f"by {g.gaca_r2.max() - g.gaca_r2.min():.4f}.")
        print("If the epsilon ranges above are small next to that, the thesis "
              "claim that gamma sets the structure and epsilon only sets the "
              "recording resolution is supported.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='small grid for a smoke test')
    args = ap.parse_args()

    seeds = QUICK_SEEDS if args.quick else SEEDS
    gammas = QUICK_GAMMAS if args.quick else GAMMAS
    epsilons = QUICK_EPSILONS if args.quick else EPSILONS

    print("=== CONDENSATION RADIUS (EPSILON) SWEEP ===")
    _, df = load_and_clean_data(DATA, drop_missing_target=True)
    X = df[SURFACE_FEATURES].values
    y = df[TARGET].values

    n_runs = len(seeds) * len(gammas) * len(epsilons)
    print(f"\nN = {len(X):,} rows; features = {SURFACE_FEATURES}")
    print(f"gammas={gammas}")
    print(f"epsilons={epsilons}")
    print(f"seeds={seeds}, coreset={CORESET}, eta={ETA}, n_iter={N_ITER}, "
          f"min_expert={MIN_EXPERT}")
    print(f"{n_runs} genesis runs total\n")

    t0 = time.perf_counter()
    all_rows = []
    for seed in seeds:
        all_rows += run_seed(seed, X, y, gammas, epsilons)

    allseeds = pd.DataFrame(all_rows)
    allseeds.to_csv('eps_sweep_allseeds.csv', index=False)

    num = ['suns_found', 'effective', 'lone', 'largest', 'largest_share',
           'median', 'compression', 'silhouette', 'gaca_r2', 'km_r2',
           'global_r2', 'genesis_s', 'n_iters', 'stranded_frac']
    keys = ['gamma', 'epsilon']
    mean = allseeds.groupby(keys, as_index=False)[num].mean()
    sd = allseeds.groupby(keys)[num].std().add_suffix('_sd').reset_index()
    summary = mean.merge(sd, on=keys)
    summary['n_seeds'] = len(seeds)
    summary['eta'] = ETA
    summary['coreset'] = CORESET
    summary.to_csv('eps_sweep.csv', index=False)

    print(f"\n--- seed-averaged (n={len(seeds)}), mean +/- sd ---")
    report(summary, gammas, epsilons)

    print(f"\nGlobal Ridge R2 = {summary['global_r2'].iloc[0]:.4f}")
    print(f"Elapsed {time.perf_counter() - t0:.0f}s")
    print("Saved eps_sweep_allseeds.csv and eps_sweep.csv")


if __name__ == "__main__":
    main()
