# GACA: Gravitational Accretion Clustering

Clustering and anomaly detection for large, unlabelled tables. You don't
specify the number of clusters: it emerges from a physics-inspired simulation
in which data points act as particles that attract, drift and merge. Points
that nothing attracts are reported as anomalies.

![Solar Genesis: particles condensing into clusters](https://raw.githubusercontent.com/henrivuorinen/GACA/main/figures/accretion_evolution.png)

## Install

```bash
pip install gaca
```

## Use

Point it at a CSV file. Preprocessing and settings are chosen from the data:

```bash
gaca run data.csv --exclude id,name
```

This writes `data_gaca/` with:

- `labels.csv`: your rows, plus a cluster, an anomaly flag and an anomaly
  score for each
- `report.html`: what each cluster looks like in terms of your columns, the
  anomalies and why they are unusual, and the cluster hierarchy
- `summary.json`: the settings that were chosen

Or from Python:

```python
from gaca import AutoGACA

auto = AutoGACA().fit(df)          # a DataFrame, an array, or a path to a CSV
print(auto.summary())
auto.result_                       # per-row cluster, anomaly flag and score
auto.report("report.html", data=df)
```

## What it does for you

- **Picks and prepares columns.** It skips identifiers and text, treats
  placeholders like `-9999` as missing, log-transforms count-like columns,
  scales robustly, and reduces many columns with PCA.
- **Chooses the resolution** where the clustering is stable across repeated
  runs. It keeps every other stable resolution as a cluster hierarchy, and it
  reports one cluster when the data has no structure.
- **Finds anomalies in every row**, with few false alarms, and groups copies of
  the same anomaly together.
- **Offers a linked view** for curved or elongated clusters, shown when it
  disagrees with the main view.
- **Scales to large files.** It fits on a sample and assigns the rest in
  batches (`--chunksize`), with memory that stays flat as the data grows.

## Learn more

- [Full documentation, benchmarks and examples](https://github.com/henrivuorinen/GACA#readme),
  including a comparison with HDBSCAN, Isolation Forest and LOF, and a walk-through
  on public Sloan Digital Sky Survey data
- [Changelog](https://github.com/henrivuorinen/GACA/blob/main/CHANGELOG.md)
- [Source and issues](https://github.com/henrivuorinen/GACA)

GACA comes from the master's thesis *The Gravitational Accretion Clustering
Algorithm (GACA): Scalable, Physics-Inspired Clustering for Large Datasets*
(H. J. Vuorinen, Aalto University, 2026). MIT licence.
