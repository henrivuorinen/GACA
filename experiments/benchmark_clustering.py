"""
Unsupervised clustering benchmark at medium scale (thesis Table 5.4).

Compares GACA against K-Means, HDBSCAN and its density-based ancestor Mean-Shift
on the same 20k company sample and the same standardised feature space.

Fixes relative to the earlier version, all of which affected the published table:

  * The silhouette / Calinski-Harabasz evaluation sample is drawn ONCE, seeded,
    and shared by every algorithm. It used to be re-drawn inside the per-algorithm
    loop from an unseeded global RNG, so each algorithm was scored on a different
    subsample and none of it was reproducible.
  * Mean-Shift's bandwidth is derived from GACA's gamma via the thesis identity
    gamma = 1 / (2 h^2), so the two methods run at the SAME kernel scale. It was
    previously fixed at 1.0 while GACA ran at gamma = 1.0, i.e. h = 0.707.
  * HDBSCAN's cluster count excludes the noise label -1, and its silhouette is
    reported both with noise treated as a cluster and with noise excluded.
  * Every algorithm is timed over fit AND assignment. GACA's timer used to stop
    before assignment while the baselines' fit_predict included theirs.
  * Clustering uses the five surface features, the same space as Tables 5.1/5.2,
    so Chapter 5 is internally consistent. The regression target is no longer part
    of the clustering feature vector.

Produces: clustering_benchmark.csv
"""
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans, MeanShift

try:
    import hdbscan
except ImportError:
    hdbscan = None
    print("Please install hdbscan: pip install hdbscan")

from gaca import GACA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from process_data import load_and_clean_data, SURFACE_FEATURES

RANDOM_SEED = 42
GAMMA, EPSILON, ETA, N_ITER, CORESET = 1.0, 0.05, 0.5, 20, 5000
HDB_MIN_CLUSTER_SIZE = 15   # kept identical in scalability_comparison.py
EVAL_SAMPLE = 5000          # silhouette / CH are O(n^2), so score on a sample

# Mean-Shift bandwidth matched to GACA's kernel scale: gamma = 1/(2 h^2).
MS_BANDWIDTH = float(1.0 / np.sqrt(2.0 * GAMMA))


def run_clustering_benchmark(X, sample_name="Company Sample"):
    """Compare GACA vs K-Means vs HDBSCAN vs Mean-Shift on the same dataset."""
    rng = np.random.default_rng(RANDOM_SEED)
    results = {}

    # --- 1. RUN GACA (Physics-Based Model) ---
    print("\n--- Running GACA (Barnes-Hut) ---")
    gaca = GACA(gamma_clustering=GAMMA, n_iterations=N_ITER,
                               epsilon=EPSILON, sample_size=CORESET,
                               random_state=RANDOM_SEED, eta=ETA)
    start = time.time()
    gaca.fit(X)
    gaca_labels = gaca._assign_to_suns(X)        # assignment is part of the cost
    gaca_time = time.time() - start
    n_suns = len(gaca.suns_)
    effective = int(len(np.unique(gaca_labels)))
    results['GACA'] = {'time': gaca_time, 'clusters': effective, 'labels': gaca_labels,
                       'note': f'{n_suns} Suns, {effective} effective, '
                               f'{gaca.n_iters_} iterations'}

    # --- 2. RUN K-MEANS (Industry Standard), matched at k = effective Suns ---
    print(f"--- Running K-Means (k={effective}) ---")
    kmeans = KMeans(n_clusters=effective, random_state=RANDOM_SEED, n_init=10)
    start = time.time()
    kmeans_labels = kmeans.fit_predict(X)
    results['K-Means'] = {'time': time.time() - start, 'clusters': effective,
                          'labels': kmeans_labels, 'note': f'k fixed at {effective}'}

    # --- 3. RUN HDBSCAN (Density-Based Standard) ---
    if hdbscan is not None:
        print("--- Running HDBSCAN ---")
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER_SIZE,
                                        gen_min_span_tree=True)
            start = time.time()
            hdbscan_labels = clusterer.fit_predict(X)
            elapsed = time.time() - start
            # -1 is the noise label, not a cluster
            n_clusters = int(len(np.unique(hdbscan_labels[hdbscan_labels >= 0])))
            noise_frac = float(np.mean(hdbscan_labels < 0))
            results['HDBSCAN'] = {
                'time': elapsed, 'clusters': n_clusters, 'labels': hdbscan_labels,
                'note': f'min_cluster_size={HDB_MIN_CLUSTER_SIZE}, '
                        f'{noise_frac * 100:.1f}% noise'}
        except Exception as e:
            print(f"HDBSCAN Failed (Likely Memory Error): {e}")

    # --- 4. RUN MEAN-SHIFT (Theoretical Ancestor to GACA) ---
    print(f"--- Running Mean-Shift (bandwidth={MS_BANDWIDTH:.3f}, "
          f"matched to gamma={GAMMA}; may take a while) ---")
    start = time.time()
    meanshift = MeanShift(bandwidth=MS_BANDWIDTH, bin_seeding=False)
    ms_labels = meanshift.fit_predict(X)
    results['Mean-Shift'] = {'time': time.time() - start,
                             'clusters': int(len(np.unique(ms_labels))),
                             'labels': ms_labels,
                             'note': f'bandwidth={MS_BANDWIDTH:.3f} '
                                     f'(= 1/sqrt(2*gamma))'}

    # --- 5. EVALUATE METRICS on ONE shared, seeded sample ---
    print("\nCalculating Silhouette Scores (shared evaluation sample)...")
    score_idx = rng.choice(len(X), min(EVAL_SAMPLE, len(X)), replace=False)
    Xs = X[score_idx]

    summary = []
    for name, data in results.items():
        labels = np.asarray(data['labels'])[score_idx]
        if len(np.unique(labels)) > 1:
            s_score = silhouette_score(Xs, labels)
            ch_score = calinski_harabasz_score(Xs, labels)
        else:
            s_score = ch_score = float('nan')

        # For HDBSCAN also report the metrics with noise points excluded, since
        # treating -1 as a single genuine cluster depresses both indices.
        s_denoised = ch_denoised = float('nan')
        if name == 'HDBSCAN':
            keep = labels >= 0
            if keep.sum() > 1 and len(np.unique(labels[keep])) > 1:
                s_denoised = silhouette_score(Xs[keep], labels[keep])
                ch_denoised = calinski_harabasz_score(Xs[keep], labels[keep])

        summary.append({
            'Algorithm': name,
            'Time (s)': round(data['time'], 4),
            'Clusters Found': data['clusters'],
            'Silhouette': round(s_score, 4),
            'Calinski-Harabasz': round(ch_score, 1),
            'Silhouette (noise excl.)': (round(s_denoised, 4)
                                         if s_denoised == s_denoised else ''),
            'CH (noise excl.)': (round(ch_denoised, 1)
                                 if ch_denoised == ch_denoised else ''),
            'Notes': data['note'],
        })

    print("\n--- BENCHMARK SUMMARY ---")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    df_summary.to_csv('clustering_benchmark.csv', index=False)
    print(f"\nSaved clustering_benchmark.csv "
          f"(N={len(X):,}, D={X.shape[1]}, eval sample={len(score_idx):,}, "
          f"seed={RANDOM_SEED})")
    return df_summary


# --- Run on Real Data ---
if __name__ == "__main__":
    try:
        print("Loading real company data for benchmark...")
        # Five surface features, the same space used by Tables 5.1 and 5.2.
        X_real, df_real = load_and_clean_data('data/20k_sample_data.csv',
                                              features=SURFACE_FEATURES)
        run_clustering_benchmark(X_real, sample_name="20k Companies")
    except FileNotFoundError:
        print("Could not find 'data/20k_sample_data.csv'. Please check the file path.")
