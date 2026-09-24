"""
Robustness benchmark under feature corruption (thesis Table 5.2 / RQ3).
Compares six predictive frameworks on the 20k company sample as an
increasing share of the test features is corrupted:
  Global          single global Ridge (reference)
  K-Means         one Ridge per K-Means cell, matched at k = EFFECTIVE GACA Suns
  GACA            one Ridge per Sun, hard Newtonian routing
  GACA+OOD        as GACA, but a point beyond its Sun's training radius is rejected
                  to the global model
  LGBM            LightGBM baseline
  LGBM+GACA       LightGBM with GACA maximum gravitational pull as an extra feature

Runs over SEEDS and reports mean +/- sd. Writes robustness_results.csv (one row
per seed x sigma x model) and robustness_summary.csv (seed-averaged).

The dataset is a CLI argument so that the same benchmark can be run at two
scales and the two compared. Chapter 5 is a small-to-medium-scale chapter and
Table 5.1 is built on the 20k sample, so the default is 20k and the headline
table stays on one population. The large run is a replication, not the primary
result: agreement between them across a 50x change in sample size is the point.

Usage:
    python final_benchmark_test.py
        -> data/20k_sample_data.csv, writes robustness_{results,summary}.csv
    python final_benchmark_test.py --data scalability_slim_1M.csv --suffix _1M
        -> writes robustness_{results,summary}_1M.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

from process_data import load_and_clean_data, SURFACE_FEATURES, TARGET
from gaca import GACA

# --- GLOBAL SETTINGS ---
SEEDS = [42, 43, 44, 45, 46]
GAMMA, EPSILON, ETA, N_ITER, CORESET = 1.0, 0.05, 0.5, 20, 5000
MIN_EXPERT = 10            # matches sweep_gamma.MIN_EXPERT
NOISE_LEVELS = [0.25, 0.5, 1.0, 2.0]
NOISE_ROW_FRACTION = 0.10  # share of test rows corrupted
OOD_QUANTILE = 0.95        # per-Sun training radius quantile (thesis sec. 6.6)

def get_pulls(X_target, suns, sun_masses):
    """Newtonian pull M_k / d^2 of every Sun on every row (Phase 4)."""
    sq_X = np.sum(X_target ** 2, axis=1)[:, np.newaxis]
    sq_S = np.sum(suns ** 2, axis=1)[np.newaxis, :]
    dot_XS = np.dot(X_target, suns.T)
    dists_sq = np.maximum(sq_X + sq_S - 2 * dot_XS, 1e-9)
    return sun_masses / dists_sq, dists_sq

def build_experts(labels_train, X_train, y_train, n_clusters, min_expert=MIN_EXPERT):
    """One expert per cluster: a Ridge if the cluster is big enough, otherwise a
    constant. Clusters with no training point get no entry and fall through to the
    global model at predict time."""
    experts = {}
    for c in range(n_clusters):
        m = labels_train == c
        n = int(m.sum())
        if n == 0:
            continue
        elif n < min_expert:
            experts[c] = ('constant', float(np.mean(y_train[m])))
        else:
            experts[c] = ('model', Ridge(alpha=1.0).fit(X_train[m], y_train[m]))
    return experts

def predict_moe(labels, X_query, experts, global_model):
    """Route each row to its expert; anything unroutable uses the global model."""
    preds = global_model.predict(X_query)
    for c, (kind, obj) in experts.items():
        m = labels == c
        if not np.any(m):
            continue
        preds[m] = obj if kind == 'constant' else obj.predict(X_query[m])
    return preds

def run_seed(seed, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    # Standardize predictor features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------- Solar Genesis ----------
    gaca = GACA(gamma_clustering=GAMMA, n_iterations=N_ITER,
                               epsilon=EPSILON, sample_size=CORESET,
                               random_state=seed, eta=ETA,
                               min_expert_size=MIN_EXPERT)
    gaca.fit(X_train)
    n_suns = len(gaca.suns_)
    gaca_train_pulls, gaca_train_d2 = get_pulls(X_train, gaca.suns_, gaca.sun_masses_)
    gaca_train_labels = np.argmax(gaca_train_pulls, axis=1)
    effective = int(len(np.unique(gaca_train_labels)))

    # ---------- Experts & Baselines ----------
    global_model = Ridge(alpha=1.0).fit(X_train, y_train)
    gaca_experts = build_experts(gaca_train_labels, X_train, y_train, n_suns)

    # K-Means matched at k = EFFECTIVE Suns
    k_matched = max(effective, 2)
    kmeans = KMeans(n_clusters=k_matched, random_state=seed, n_init=10)
    km_train_labels = kmeans.fit_predict(X_train)
    km_experts = build_experts(km_train_labels, X_train, y_train, k_matched)

    # Per-Sun OOD radius
    ood_r2 = np.zeros(n_suns)
    for k in range(n_suns):
        m = gaca_train_labels == k
        if np.any(m):
            ood_r2[k] = np.quantile(gaca_train_d2[m, k], OOD_QUANTILE)

    # --- LIGHTGBM MODELS ---
    # Standard LightGBM
    lgbm_model = LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)
    lgbm_model.fit(X_train, y_train)

    # LightGBM + GACA Features (Appending Max Gravitational Pull as a feature)
    X_train_enriched = np.column_stack([X_train, np.max(gaca_train_pulls, axis=1)])
    lgbm_gaca_model = LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)
    lgbm_gaca_model.fit(X_train_enriched, y_train)

    # ---------- evaluation ----------
    def evaluate(Xq):
        pulls, d2 = get_pulls(Xq, gaca.suns_, gaca.sun_masses_)
        gaca_labels = np.argmax(pulls, axis=1)
        km_labels = kmeans.predict(Xq)

        # Expert predictions
        gaca_pred = predict_moe(gaca_labels, Xq, gaca_experts, global_model)

        # OOD variant: reject rows outside their Sun's training radius
        own_d2 = d2[np.arange(len(Xq)), gaca_labels]
        rejected = own_d2 > ood_r2[gaca_labels]
        gaca_ood = gaca_pred.copy()
        if np.any(rejected):
            gaca_ood[rejected] = global_model.predict(Xq[rejected])

        # LGBM Predictions
        lgbm_pred = lgbm_model.predict(Xq)

        # LGBM + GACA Features Prediction
        Xq_enriched = np.column_stack([Xq, np.max(pulls, axis=1)])
        lgbm_gaca_pred = lgbm_gaca_model.predict(Xq_enriched)

        return {
            'Global': r2_score(y_test, global_model.predict(Xq)),
            'KMeans': r2_score(y_test, predict_moe(km_labels, Xq, km_experts, global_model)),
            'GACA': r2_score(y_test, gaca_pred),
            'GACA+OOD': r2_score(y_test, gaca_ood),
            'LGBM': r2_score(y_test, lgbm_pred),
            'LGBM+GACA': r2_score(y_test, lgbm_gaca_pred),
            'rejected_frac': float(np.mean(rejected)),
        }

    # One corruption pattern, drawn once, then rescaled by sigma.
    rng = np.random.default_rng(seed)
    n_noise = int(NOISE_ROW_FRACTION * len(X_test))
    noise_idx = rng.choice(len(X_test), n_noise, replace=False)
    Z = rng.normal(0.0, 1.0, size=(n_noise, X_test.shape[1]))

    rows = []
    models_to_eval = ('Global', 'KMeans', 'GACA', 'GACA+OOD', 'LGBM', 'LGBM+GACA')

    for sigma in [0.0] + NOISE_LEVELS:
        Xn = np.copy(X_test)
        if sigma > 0:
            Xn[noise_idx] += sigma * Z
        res = evaluate(Xn)
        for model in models_to_eval:
            rows.append(dict(seed=seed, sigma=sigma, model=model, r2=res[model],
                             suns=n_suns, effective=effective, k_matched=k_matched,
                             rejected_frac=res['rejected_frac'],
                             n_iters=gaca.n_iters_))

        # Console printing for progress monitoring
        print(f"  seed {seed} | sigma {sigma:<5} | " +
              " | ".join(f"{m} {res[m]:>8.4f}" for m in models_to_eval) +
              f" | rejected {res['rejected_frac'] * 100:.1f}%")

    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data/20k_sample_data.csv',
                    help='CSV to evaluate on (default: the 20k sample, which is '
                         'the population Table 5.1 uses)')
    ap.add_argument('--suffix', default='',
                    help='appended to the output filenames, e.g. "_1M", so a '
                         'replication run cannot overwrite the primary result')
    args = ap.parse_args()

    print("=== ROBUSTNESS BENCHMARK (RQ3) ===")
    _, df = load_and_clean_data(args.data, drop_missing_target=True)
    X = df[SURFACE_FEATURES].values
    y = df[TARGET].values

    print(f"source = {args.data}")
    print(f"N = {len(X):,} rows; features = {SURFACE_FEATURES}")
    print(f"gamma={GAMMA}, epsilon={EPSILON}, eta={ETA}, coreset={CORESET}, "
          f"min_expert={MIN_EXPERT}, seeds={SEEDS}")
    print(f"{int(NOISE_ROW_FRACTION * 100)}% of test rows corrupted per sigma\n")

    all_rows = []
    for seed in SEEDS:
        all_rows += run_seed(seed, X, y)

    allseeds = pd.DataFrame(all_rows)
    allseeds['source'] = args.data
    allseeds['n_rows'] = len(X)
    out_all = f'robustness_results{args.suffix}.csv'
    out_sum = f'robustness_summary{args.suffix}.csv'
    allseeds.to_csv(out_all, index=False)

    summary = (allseeds.groupby(['sigma', 'model'], as_index=False)
               .agg(r2_mean=('r2', 'mean'), r2_sd=('r2', 'std'),
                    effective=('effective', 'mean'), k_matched=('k_matched', 'mean'),
                    n_seeds=('r2', 'size')))
    summary.to_csv(out_sum, index=False)

    wide = summary.pivot(index='sigma', columns='model', values='r2_mean')
    wide_sd = summary.pivot(index='sigma', columns='model', values='r2_sd')

    order = ['Global', 'KMeans', 'GACA', 'GACA+OOD', 'LGBM', 'LGBM+GACA']

    print(f"\n--- seed-averaged test R2 (n={len(SEEDS)}), mean +/- sd ---")
    print(f"{'sigma':>7} | " + " | ".join(f"{m:>17}" for m in order))

    for sigma in wide.index:
        cells = [f"{wide.loc[sigma, m]:>8.4f}+-{wide_sd.loc[sigma, m]:<7.4f}" for m in order]
        label = 'clean' if sigma == 0 else f"{sigma}"
        print(f"{label:>7} | " + " | ".join(cells))

    print(f"\nMean effective Suns = {summary['effective'].mean():.1f} "
          f"(K-Means matched at k = {summary['k_matched'].mean():.1f})")
    print(f"Saved {out_all} and {out_sum}")

if __name__ == "__main__":
    main()