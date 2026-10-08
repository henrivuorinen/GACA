"""
Bandwidth (gamma) sweep for GACA.

For each gamma this reports, on the real 20k company dataset:
  - Suns found      : modes produced by Solar Genesis on the coreset
  - Effective Suns  : how many of those Suns actually receive >=1 training point
  - Lone Suns       : Suns that receive exactly 1 training point (anomaly isolation, RQ2)
  - Largest / Median: assigned-cluster-size distribution over the training set
  - Silhouette      : internal cohesion of the assignment (on a fixed sample)
  - GACA R2         : Mixture-of-Experts test R2 using GACA assignments (RQ3)
  - KMeans R2       : same MoE with K-Means at k = Effective Suns (matched baseline)
  - Global R2       : single global Ridge (constant reference)

This regenerates the "phase transition" analysis of thesis section 5.1.

Everything is repeated over SEEDS and reported as mean +/- sd: a single seed
cannot support a claim about a trend. Two CSVs are written:

  sweep_results_eta{ETA}_allseeds.csv   one row per (seed, gamma)
  sweep_results_eta{ETA}.csv            seed-averaged, plus *_sd columns

The second keeps the original column names, so plot_phase_transition.py consumes
it unchanged.

Matched-baseline convention: K-Means is matched at k = EFFECTIVE Suns, i.e. the
number of experts GACA actually deploys. final_benchmark_test.py uses the same
convention, so the two tables of Chapter 5 are directly comparable.
"""
import os
import contextlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from process_data import load_and_clean_data, SURFACE_FEATURES, TARGET
from gaca import GACA

# --- knobs to explore ---
SEEDS = [42, 43, 44, 45, 46]
GAMMAS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
EPSILON = 0.05
CORESET = 5000
N_ITER = 20
ETA = 0.5           # blurring mean-shift damping (eqs. 3.9-3.11)
MIN_EXPERT = 10     # Suns with fewer training points get a constant predictor
SIL_SAMPLE = 5000   # silhouette is O(n^2), so it is measured on a fixed sample


def moe_r2(labels_train, labels_eval, X_train, y_train, X_eval, y_eval,
           min_expert=MIN_EXPERT):
    """Fit one Ridge per cluster on train, predict eval, return R2.

    Rows whose cluster holds no training point are predicted by a global Ridge
    rather than left at 0.0. That zero default used to penalise GACA specifically:
    GACA routinely produces Suns that receive no training point, whereas K-Means
    populates every one of its k cells, so only the GACA column paid the cost of
    predicting 0 against a target whose mean is above 4.
    """
    fallback = Ridge(alpha=1.0).fit(X_train, y_train)
    preds = fallback.predict(X_eval)  # covers clusters unseen in training
    for c in np.unique(labels_train):
        m_tr, m_ev = labels_train == c, labels_eval == c
        if not m_ev.any():
            continue
        if m_tr.sum() < min_expert:  # the constant fallback Ch.4 already claims
            preds[m_ev] = y_train[m_tr].mean()
        else:
            preds[m_ev] = Ridge(alpha=1.0).fit(X_train[m_tr], y_train[m_tr]).predict(X_eval[m_ev])
    return r2_score(y_eval, preds)


def stranded_fraction(labels_train, labels_eval):
    """Share of eval rows routed to a cluster that holds no training point.

    Reported so the size of the effect the zero-default bug used to cause is on
    the record rather than assumed to be negligible.
    """
    return float(np.mean(~np.isin(labels_eval, np.unique(labels_train))))


def run_seed(seed, X, y):
    rng = np.random.default_rng(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # sample indices for silhouette (it is O(n^2)); seeded, and fixed across gammas
    sil_idx = rng.choice(len(X_train), min(SIL_SAMPLE, len(X_train)), replace=False)

    global_r2 = r2_score(y_test, Ridge(alpha=1.0).fit(X_train, y_train).predict(X_test))

    rows = []
    for gamma in GAMMAS:
        gaca = GACA(assignment='newton', method='barnes_hut', gamma_clustering=gamma, n_iterations=N_ITER,
                                   epsilon=EPSILON, sample_size=CORESET,
                                   random_state=seed, eta=ETA,
                                   min_expert_size=MIN_EXPERT)
        # silence the per-iteration genesis prints
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            gaca.fit(X_train)
            train_labels = gaca._assign_to_suns(X_train)
            test_labels = gaca._assign_to_suns(X_test)

        suns_found = len(gaca.suns_)
        uniq, counts = np.unique(train_labels, return_counts=True)
        effective = len(uniq)
        lone = int(np.sum(counts == 1))
        largest = int(counts.max())
        median = float(np.median(counts))

        # Needs >=2 distinct labels inside the sampled subset, not merely in the
        # full training set.
        sub_labels = train_labels[sil_idx]
        sil = (float(silhouette_score(X_train[sil_idx], sub_labels))
               if len(np.unique(sub_labels)) > 1 else float('nan'))

        gaca_r2 = moe_r2(train_labels, test_labels, X_train, y_train, X_test, y_test)

        # matched K-Means baseline at k = effective Suns
        if effective > 1:
            km = KMeans(n_clusters=effective, random_state=seed, n_init=10)
            km_train = km.fit_predict(X_train)
            km_test = km.predict(X_test)
            km_r2 = moe_r2(km_train, km_test, X_train, y_train, X_test, y_test)
        else:
            km_r2 = float('nan')

        rows.append({
            'seed': seed, 'gamma': gamma, 'suns_found': suns_found,
            'effective': effective, 'lone': lone, 'largest': largest,
            'median': median, 'silhouette': sil, 'gaca_r2': gaca_r2, 'km_r2': km_r2,
            'global_r2': global_r2, 'n_iters': gaca.n_iters_,
            'stranded_frac': stranded_fraction(train_labels, test_labels),
            'eta': ETA, 'epsilon': EPSILON, 'coreset': CORESET,
        })
        print(f"  seed {seed} | gamma {gamma:<4} | suns {suns_found:<3} "
              f"eff {effective:<3} lone {lone:<3} | sil {sil:>6.3f} | "
              f"GACA {gaca_r2:>7.4f} | KM {km_r2:>7.4f} | iters {gaca.n_iters_}")
    return rows


def main():
    _, df = load_and_clean_data('data/20k_sample_data.csv', drop_missing_target=True)
    X = df[SURFACE_FEATURES].values
    y = df[TARGET].values

    print(f"\nN = {len(X):,} rows; features = {SURFACE_FEATURES}")
    print(f"Coreset={CORESET}, epsilon={EPSILON}, n_iter={N_ITER}, eta={ETA}, "
          f"min_expert={MIN_EXPERT}, seeds={SEEDS}\n")

    all_rows = []
    for seed in SEEDS:
        all_rows += run_seed(seed, X, y)

    allseeds = pd.DataFrame(all_rows)
    all_out = f'sweep_results_eta{ETA}_allseeds.csv'
    allseeds.to_csv(all_out, index=False)

    # --- seed-averaged table (column names preserved for the plot script) ---
    num = ['suns_found', 'effective', 'lone', 'largest', 'median', 'silhouette',
           'gaca_r2', 'km_r2', 'global_r2', 'n_iters', 'stranded_frac']
    mean = allseeds.groupby('gamma', as_index=False)[num].mean()
    sd = allseeds.groupby('gamma')[num].std().add_suffix('_sd').reset_index()
    summary = mean.merge(sd, on='gamma')
    summary['n_seeds'] = len(SEEDS)
    summary['eta'] = ETA
    summary['epsilon'] = EPSILON
    summary['coreset'] = CORESET
    out = f'sweep_results_eta{ETA}.csv'
    summary.to_csv(out, index=False)

    print(f"\n--- seed-averaged (n={len(SEEDS)}), mean +/- sd ---")
    print(f"{'gamma':>6} | {'suns':>12} | {'eff':>12} | {'lone':>11} | "
          f"{'silhouette':>15} | {'GACA R2':>16} | {'KM R2':>16}")
    for _, r in summary.iterrows():
        print(f"{r['gamma']:>6.2f} | {r['suns_found']:>6.1f}+-{r['suns_found_sd']:<5.1f} | "
              f"{r['effective']:>6.1f}+-{r['effective_sd']:<5.1f} | "
              f"{r['lone']:>5.1f}+-{r['lone_sd']:<5.1f} | "
              f"{r['silhouette']:>7.3f}+-{r['silhouette_sd']:<7.3f} | "
              f"{r['gaca_r2']:>7.4f}+-{r['gaca_r2_sd']:<7.4f} | "
              f"{r['km_r2']:>7.4f}+-{r['km_r2_sd']:<7.4f}")
    print(f"\nGlobal Ridge R2 = {summary['global_r2'].iloc[0]:.4f} "
          f"+- {summary['global_r2_sd'].iloc[0]:.4f}")
    print(f"Mean stranded test rows (no trained expert): "
          f"{allseeds['stranded_frac'].mean() * 100:.3f}%")
    print(f"Saved {out} (used by plot_phase_transition.py) and {all_out}")


if __name__ == "__main__":
    main()
