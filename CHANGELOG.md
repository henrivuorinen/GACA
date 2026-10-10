# Changelog

Versions follow [semantic versioning](https://semver.org): bug fixes raise the
last number, new features the middle one, and changes that break existing code
the first.

## Unreleased

### Changed
- **Better automatic level choice.** Among equally long plateaus, the finest
  one whose stability is within 0.1 of the most stable is chosen. Mean ARI:
  development sets 0.63 → 0.73, held-out OpenML sets 0.30 → 0.41.
- **Rare groups in the anomaly score.** Clusters holding less than 5% of the
  rows pull with a weight proportional to their share (`rare_share`), so
  small, dense, isolated groups score as anomalous. Flags are unchanged. KDD
  Cup ROC AUC: 0.41 → 0.79; held-out shuttle: 0.80 → 0.98.

### Added
- **`--truth COLUMN` and `--baselines`:** score the clustering against known
  labels (kept out of the clustering) in a report section. Optionally K-Means
  and HDBSCAN are scored on the same preprocessed data. In Python:
  `AutoGACA.compare()`.
- **`--fit-first`:** with `--chunksize`, fit on the start of a time-ordered
  file, so drift is measured against it. The report charts drift per chunk.
- **A third SDSS example file** (`sdss_timeline.csv`, original survey then
  BOSS) with a real change for trying drift monitoring.
- `experiments/benchmark_openml.py`: GACA, HDBSCAN, K-Means, Isolation Forest
  and LOF on seven public OpenML datasets from finance, agriculture, biology,
  image analysis, handwriting, medicine and aerospace. Results, including the
  weak spots, are in the README.
- **Drift monitoring.** `AutoGACA.drift()` compares a batch of new rows with
  the fitted data (cluster-mix shift, anomaly rate, new anomaly groups) and
  flags changes that are both large and significant. With `--chunksize` the
  CLI writes `drift.csv`. In tests there were no false alarms on 30 unchanged
  batches, and every real shift was caught.
- **Cluster rules.** Each cluster gets a short rule in the original column
  units, with precision and recall, in the report, the summary and
  `summary.json`. On SDSS objects the quasar cluster is
  `redshift > 0.745 and u ≤ 20.9` (99% / 97%).
- `resolution=` / `--resolution D`: keep groups at least D apart separate. γ
  is derived from the resolution law (`resolution_to_gamma`, `critical_gap`).
  On SDSS galaxy positions, `--resolution 3.3` (Mpc) recovers the Coma
  cluster as the largest group.
- `docs/theory.md`: the mathematics behind GACA's behaviour and design
  choices. Its new resolution law has two parts:
  - **Proved, for two Suns:** the exact gap map, inevitable collapse, bounds
    on the merge time, and a critical gap of h√(2 ln T)(1 + o(1)).
  - **Approximated, for clusters with a spread:** a mean-field model checked
    against simulations.

  `experiments/resolution_law.py` verifies the numbers.

## 1.3.0

### Added
- **Linked view.** AutoGACA also clusters with saddle linking as an alternative,
  stored in `gaca_linked_cluster` and shown in the report when it disagrees
  with the main view. Turn it off with `link_view=False` or `--no-link-view`.
- `n_jobs` / `--jobs`: the bandwidth sweep and the hierarchy fits run in
  parallel threads. This is about 2x faster, with identical results.

### Changed
- **Within-group scaling.** A column whose values form clearly separate groups
  is scaled by the spread inside the groups rather than by its interquartile
  range, which spans the gaps and squeezed the column. This improved most
  benchmark sets (breast cancer 0.36 to 0.56; nested groups 0.66 to 0.98).
  `separate_modes=False` turns it off.
- The PyPI page shows a short description (`PYPI.md`) instead of the full
  README, which stays on GitHub.

## 1.2.0

First release on PyPI (`pip install gaca`).

### Added
- **AutoGACA and the `gaca` command.** Clusters any table without settings:
  - picks usable columns and recognises identifiers and placeholder codes such
    as `-9999`
  - imputes missing values, log-transforms count-like columns, scales robustly,
    and projects with PCA above `max_dims` columns
  - chooses the bandwidth from the most stable plateau of a sweep

  Writes a labelled CSV, an HTML report and a JSON summary. `--chunksize`
  streams files larger than memory.
- **Cluster hierarchy.** Every stable resolution is kept and nested into a tree
  (`gaca_level_1 … gaca_level_K`, path-style names such as `0.2.1`), shown in
  the report.
- **Pull-based assignment and Lone-Sun registration.** Anomalies are detected
  for every row, not only for rows that happened to be in the coreset. Copies
  of one anomaly share a group.
- **Saddle linking** (`link_tau`) for curved and elongated clusters.
- `--bandwidth`, to give the kernel width in data units (e.g. Mpc).
- `experiments/benchmark_suite.py`: comparison with HDBSCAN, Isolation Forest,
  LOF and K-Means on clustering, anomaly detection and scale.
- `examples/sdss/`: two public Sloan Digital Sky Survey tables, with a guide.

### Changed
- Solar Genesis evaluates the Gaussian sum exactly in vectorised blocks
  (`method='exact'`, the default). This gives the same Suns as Barnes-Hut and
  is about 100 times faster in this implementation.
- The condensation radius defaults to `epsilon='auto'` (0.05/√γ, capped at half
  the point spacing).
- The Lone-Sun threshold `kappa` is relative to the typical pull.
- A Lone Sun is defined by its share of the coreset rather than by mass 1.
- pandas is a core dependency.

### Fixed
- A fixed condensation radius larger than the point spacing chained dense
  structures into one particle in the first iteration.
- On data without cluster structure, the bandwidth selection could report
  spurious clusters. A split now has to agree across subsamples.
- Anomaly scores let the copies of a sampled anomaly vouch for each other.

The thesis behaviour remains available as
`GACA(assignment='newton', method='barnes_hut', epsilon=0.05)`, and every
script in `experiments/` uses it.

## 1.0.0

Reference implementation from the master's thesis.
