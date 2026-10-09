# Changelog

Versions follow [semantic versioning](https://semver.org): bug fixes raise the
last number, new features the middle one, and changes that break existing code
the first.

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
