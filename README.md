# GACA: Gravitational Accretion Clustering Algorithm

GACA is a density-based clustering method that treats every data point as a
particle with mass. Nearby mass attracts, particles drift towards dense regions,
and particles that meet merge and keep their combined mass. What is left are the
**Suns**: the cluster centres. You never specify the number of clusters. It
emerges from the dynamics and is controlled by one bandwidth parameter, γ.

GACA is designed for large tables. The expensive simulation runs once on a small
random sample (the coreset), and the full dataset is then streamed through the
resulting Suns in batches. In the thesis experiments this clustered **10 million
records in about 23 seconds on a laptop, with constant memory**.

This repository contains the reference implementation from the master's thesis
*The Gravitational Accretion Clustering Algorithm (GACA): Scalable,
Physics-Inspired Clustering for Large Datasets*, plus several improvements made
after it (see [Improvements since the thesis](#improvements-since-the-thesis)).
The thesis behaviour is still available with
`GACA(assignment='newton', method='barnes_hut')`, and every script in
`experiments/` uses it.

![Solar Genesis: 3,000 particles condensing into 7 Suns](https://raw.githubusercontent.com/henrivuorinen/GACA/main/figures/accretion_evolution.png)

## How it works

**Phase 1: Solar Genesis (structure discovery on a coreset).** A uniform sample of
`sample_size` rows (5,000 by default) starts as particles of mass 1. Each
iteration:

1. **Transport.** Every particle moves a damped step (η) towards the mass-weighted
   Gaussian average of its surroundings. This is exactly a blurring mean-shift
   step: the displacement equals `(1 / 2γ) ∇ log KDE(x)`.
2. **Condensation.** Particles closer than ε are linked, and each connected
   component is replaced by one particle at its centre of mass, carrying the
   summed mass. Mass is conserved exactly.

The Gaussian sum is evaluated exactly in vectorised blocks (`method='exact'`,
the default). The thesis used a Barnes-Hut tree (`method='barnes_hut'`), which is
asymptotically cheaper but in interpreted Python is 90 to 140 times slower on
the coresets GACA uses, and only approximate. Genesis stops at a stable plateau
or at the iteration cap. The surviving particles are the Suns, and every
coreset row remembers which Sun it condensed into.

**Phase 2: Particle Accretion (assignment of every row).** Each row is assigned
to the Sun whose coreset members exert the strongest Gaussian pull on it,
`Σ_{j ∈ Sun k} exp(−γ‖x − z_j‖²)`, summed over its 32 nearest coreset members.
This follows the shape the dynamics found rather than cutting space into
Voronoi cells. It is O(N log n_c) time and works batch by batch, so memory
depends on the batch size rather than on N. (`assignment='newton'` restores
the thesis rule, `M_k / d²`.)

**Lone Suns (anomalies).** A point far from everything feels essentially no
Gaussian pull, never moves and never merges. It survives as a Sun of mass 1: a
Lone Sun. Assignment applies the same test to every streamed row: if the total
pull on it is below `kappa_`, the row would not have moved had it been in the
coreset (thesis Prop. 3.8), so it becomes a new Lone Sun, or joins an earlier
Lone Sun that pulls it. Anomalies therefore no longer need to be in the coreset
to be isolated.

**Saddle linking (optional, `link_tau`).** At a fine bandwidth a curved or
elongated cluster splits into a chain of Suns. With `link_tau` set, two Suns
are joined when the density along the best edge between their members stays
above `link_tau` times the lower of their two peaks, so the decision rests on
the mass lying between the Suns rather than the distance between them. Lone
Suns are never joined.

## Installation

```bash
pip install gaca
```

That installs the package and the `gaca` command (numpy, scipy, scikit-learn
and pandas are pulled in). Python 3.10 or newer.

For the latest development version, or to run the tests and experiments:

```bash
pip install "git+https://github.com/henrivuorinen/GACA.git"   # latest from GitHub

git clone https://github.com/henrivuorinen/GACA.git              # or work on the code
cd GACA
pip install -e ".[dev]"            # + pytest, build, twine
pip install -e ".[experiments]"    # + matplotlib, hdbscan, lightgbm, ... for experiments/
```

## Any table: AutoGACA and the `gaca` command

The algorithm only needs points in a space, but someone has to decide how to
turn a table into that space and which bandwidth to use. `AutoGACA` makes those
decisions from the data and records each one, so a dataset can be clustered
without knowing anything about it in advance:

```bash
gaca run galaxies.csv                         # writes galaxies_gaca/
gaca run galaxies.csv --exclude objid,ra,dec  # leave columns out
gaca run huge.csv --chunksize 200000          # files larger than memory
```

The output folder holds `labels.csv` (the input rows plus `gaca_cluster`,
`gaca_anomaly`, `gaca_anomaly_group` and `gaca_anomaly_score`), a self-contained
`report.html` and `summary.json`. The report describes each cluster in terms of
the original columns, lists the anomalies and the columns that make them
unusual, shows how the resolution was chosen and which other resolutions were
also stable, and lists every preprocessing decision.

From Python:

```python
from gaca import AutoGACA

auto = AutoGACA().fit(df)           # DataFrame, array, or path to a CSV
print(auto.summary())
auto.result_                        # per-row cluster, anomaly flag, group, score
auto.report("report.html", data=df)
new = auto.assign(df_new)           # same preprocessing, same clusters
```

What it decides, and how:

| Step | Rule |
|---|---|
| Columns | Numeric columns are used. Text, identifiers (unique integers named like `id` or strictly increasing), constant columns and columns more than 50% missing are left out. `columns=` / `exclude=` override. |
| Missing values | Median of the column. |
| Transforms | `log1p` for count- or flux-like columns: non-negative, skewed, and spanning more than an order of magnitude. Other skewed columns are left alone, since a log would also pull genuine outliers back towards the data. |
| Scaling | Median and interquartile range, so outliers do not compress everything else. A column whose values form clearly separate groups (a 1-D Gaussian mixture, groups of at least 5% of rows, Ashman's D > 2) is scaled by the spread inside the groups instead, since its interquartile range spans the gaps and would squeeze it. `scale='none'` (`--scale none`) keeps the values as they are, for data whose distances already mean something, such as positions in physical units. |
| Dimensions | Unchanged up to `max_dims` columns (default 10). Above that, PCA to between 5 and `max_dims` components, the number set by the noise floor of the singular values (Gavish and Donoho). |
| γ | Swept as c / (median squared distance between rows). At each value GACA is fitted on four subsamples. γ is the middle of the most stable stretch of the longest plateau in the cluster count. A split only counts if the subsamples agree (adjusted Rand index ≥ 0.8). Otherwise the answer is one group. |
| Anomalies | Rows that no cluster pulls, plus clusters smaller than `min_cluster_size` (10 rows). Score = −log10 of the pull relative to a typical row; above 3 is anomalous. |
| Hierarchy | Every stable resolution of the sweep is kept, fitted on the same coreset and nested into a tree: each fine cluster sits under the coarse cluster holding most of its particles, and each row is assigned within its own branch. `gaca_cluster` is the level the rule above picks; `gaca_level_1 … gaca_level_K` hold all levels as path-style names (`0`, `0.2`, `0.2.1`). `hierarchy=False` turns it off. |
| Linked view | A second clustering with saddle linking (its own automatic γ), better for curved or elongated groups and worse for overlapping ones. It is in `gaca_linked_cluster`, and the report shows it when it disagrees with the main view. `link_view=False` / `--no-link-view` skips it. |
| Speed | The sweep's fits and the hierarchy levels run in parallel threads (`n_jobs`, `--jobs`; up to 8 by default), with identical results. |

How it did on data it was not tuned on (ARI against known labels; *best γ* is
the best that any γ in the sweep achieved):

| Data | Rows × columns | AutoGACA | Best γ | Notes |
|---|---|---|---|---|
| Wine | 178 × 13 | 0.79 | 0.82 | 3 clusters, PCA to 5-D |
| Iris | 150 × 4 | 0.54 | 0.55 | two of the species overlap; density methods see 2 groups |
| Breast cancer | 569 × 30 | 0.36 | 0.50 | |
| Blobs + planted anomalies | 20,042 × 5 | 0.99 | 0.99 | 42/42 anomalies, 0.1% false positives |
| 6 clusters in 30-D, 25 columns of noise | 6,000 × 30 | 0.90 | 0.89 | PCA to 5-D |
| Two moons, `link_tau=0.6` | 4,000 × 2 | 1.00 | | |
| Concentric circles, `link_tau=0.6` | 4,000 × 2 | 0.93 | | |
| Handwritten digits | 1,797 × 64 | 0.17 | 0.24 | not density-separated in a few dimensions |
| One Gaussian (no clusters) | 5,000 × 2 to 10 | 1 cluster | | correctly reports no structure and no anomalies |

`link_tau` is not on by default: it joins curved clusters but also merges
clusters that overlap (the 30-D case drops to 0.15 with it). Instead, AutoGACA
computes the linked clustering as an alternative view. When the two disagree,
one of them was the good one on every benchmark set (moons 1.00, circles 0.96,
anisotropic 0.99 and digits 0.39 from the linked view; wine 0.81, breast cancer
0.36 and the 30-D case 0.90 from the main view). The report shows both, with
guidance on which suits which shape of group.

### The cluster hierarchy

Real data often has structure at several scales, and one automatic resolution
then has to pick between a coarse and a fine view that are almost equally
stable. AutoGACA keeps all of them. On the SDSS objects (no labels used), the
tree runs from 3 clusters (the main population, quasars, cool M dwarfs) to 10
(distant luminous galaxies, white dwarfs, low-redshift quasars and warmer M
dwarfs separate out) to 17, where the quasars form a ladder of 99 to 100% pure
clusters by redshift (z ≈ 0.9, 1.2, 1.6 and 2.0). The report shows the tree
with what sets each branch apart from its parent.

Two details make the tree reliable:

- The bandwidth sweep reaches down to the local scale (a kernel about three
  nearest-neighbour distances wide), not just to a fixed fraction of the
  typical distance. Otherwise groups made of subgroups never get resolved.
- Between equally long, equally stable plateaus the finer one is used for
  `gaca_cluster`, since the coarser ones remain in the hierarchy.

Robust scaling divides each column by its spread, and for a column whose
values form far-apart groups the interquartile range spans the gaps, which
compresses the column and merged the subgroups along it. Such columns are
therefore scaled by the spread inside their groups. In the test above (two
groups of three subgroups), this raised the best level from ARI 0.66 to 0.98.
Across the benchmark it improved breast cancer from 0.36 to 0.56, digits (best
level) from 0.36 to 0.56, iris (best level) from 0.54 to 0.75, and the 30-D
linked view from 0.15 to 0.87. It lowered the main view on the anisotropic set
(0.67 to 0.32; the linked view stays at 0.99) and the SDSS linked view (0.61
to 0.53). `separate_modes=False` restores plain robust scaling.

To try it on real astronomy data, `examples/sdss/` downloads two public Sloan
Digital Sky Survey tables (object properties, and galaxy positions around the
Coma cluster) and walks through both runs; see
[examples/sdss/README.md](https://github.com/henrivuorinen/GACA/blob/main/examples/sdss/README.md).

### Why `max_dims` can be 10 or more

The thesis put the ceiling at about 15 dimensions, for two reasons: the
Barnes-Hut tree degrades, and the Gaussian kernel loses contrast. The first is
gone with the exact backend. The second was made worse in the thesis experiment
by keeping γ = 1 at every dimension: in 45-D, squared distances are around 90,
every pull is about e⁻⁹⁰, and every point becomes a singleton. With γ scaled to
the data, six clusters were recovered at ARI ≥ 0.98 from 10 to 50 dimensions
when every column carries signal. When most columns are noise the limit is
real: raw data held up to about 15 dimensions (ARI 0.86), fell to 0.17 at 30,
and PCA restored it (0.89 at 5 components). Hence the default: use the columns
as they are up to 10, and project above that. Raise `max_dims` when you know the
columns are informative.

## Benchmark against established methods

`experiments/benchmark_suite.py` compares AutoGACA with HDBSCAN, Isolation
Forest, LOF and K-Means. Every method gets the same preprocessed input, and
nothing is tuned per dataset with the labels.

**Clustering** (adjusted Rand index; HDBSCAN with minimum cluster size 1% of
rows, K-Means given the true k as a reference):

| Data | GACA | GACA, linked view | HDBSCAN | K-Means (true k) |
|---|---|---|---|---|
| wine | **0.85** | 0.00 | 0.27 | 0.85 |
| breast cancer | **0.56** | 0.01 | 0.08 | 0.69 |
| digits | **0.18** | 0.17 | **0.18** | 0.52 |
| uneven blobs 5-D | **0.99** | **0.99** | 0.71 | 0.98 |
| 6 clusters in 30-D noise | **0.94** | 0.87 | 0.18 | 0.94 |
| varied density | 0.93 | **0.95** | 0.78 | 0.90 |
| anisotropic | 0.32 | **0.99** | 0.95 | 0.78 |
| two moons | 0.60 | **1.00** | **1.00** | 0.56 |
| circles | 0.53 | 0.96 | **1.00** | 0.00 |
| SDSS objects (class) | 0.48 | 0.53 | **0.61** | 0.35 |

GACA's main view is the stronger method on overlapping, Gaussian-like groups,
varied densities and noisy dimensions. The linked view, which AutoGACA
computes as well and shows when it disagrees, handles curved and elongated
shapes, where it matches or nearly matches HDBSCAN. HDBSCAN stays ahead on
circles and the SDSS objects. When the two GACA views disagree, one of them
is the good one on every set here. GACA also assigns 90 to 100% of rows,
while HDBSCAN leaves up to 68% as noise. All methods run on AutoGACA's
preprocessed space, so its preprocessing helps them too: the within-group
scaling raised K-Means on iris from 0.58 to 0.87 and on digits from 0.27 to
0.52. K-Means, given the true number of clusters, remains the reference to
beat on the classic labelled sets. The ARI is for the single level in
`gaca_cluster`; the hierarchy often holds a better-matching level (iris 0.75,
digits 0.56).

**Anomaly detection** (average precision; ROC AUC in brackets):

| Data | GACA | Isolation Forest | LOF | HDBSCAN |
|---|---|---|---|---|
| blobs + planted anomalies | **1.00** (1.00) | 0.97 (1.00) | **1.00** (1.00) | 0.01 (0.67) |
| company-like + planted | **0.84** (1.00) | 0.62 (1.00) | 0.76 (1.00) | 0.45 (0.76) |
| breast cancer, 5% malignant | **0.47** (0.95) | 0.32 (0.91) | **0.47** (0.94) | 0.05 (0.53) |
| SDSS, quasars thinned to 1% | 0.15 (0.96) | **0.32** (0.96) | 0.03 (0.55) | 0.02 (0.60) |
| SDSS, white dwarfs (~1%) | 0.04 (0.90) | **0.06** (0.94) | 0.01 (0.47) | 0.01 (0.52) |
| KDD Cup 99 attacks | 0.03 (0.41) | **0.36** (0.91) | 0.03 (0.36) | 0.03 (0.25) |

GACA ranks isolated anomalies best and flags precisely: on the planted sets it
flags 0.2 to 0.7% of rows and catches them all, where Isolation Forest flags 9
to 16%. It is weaker when the "anomalies" are a dense group of their own (a
thousand similar quasars, or the floods of identical connections in KDD Cup,
where all density methods fail).

**Scale** (5-D data; seconds, and peak memory of the process):

| Rows | GACA, fixed γ, streamed | AutoGACA | HDBSCAN | Isolation Forest | LOF | K-Means |
|---|---|---|---|---|---|---|
| 10,000 | 1.0 s, 519 MB | 9.9 s, 733 MB | 0.3 s, 207 MB | 0.1 s | 0.1 s | 0.0 s |
| 100,000 | 1.1 s, 542 MB | 10.7 s, 962 MB | 39.8 s, 3.1 GB | 0.3 s | 1.1 s | 0.0 s |
| 1,000,000 | 2.2 s, 603 MB | 13.8 s, 2.7 GB | did not finish in 10 min | 3.0 s, 469 MB | 29 s, 1.2 GB | 0.2 s |

Streamed GACA keeps memory flat. AutoGACA adds a fixed ~10 s for choosing γ,
and in memory it holds the whole table (the CLI's `--chunksize` mode does not).
Isolation Forest and K-Means are faster at every size; the speed advantage is
over density-based clustering.

## Using the algorithm directly

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from gaca import GACA

X = StandardScaler().fit_transform(X_raw)   # standardise first

model = GACA(gamma_clustering=1.0, random_state=0).fit(X)
labels = model.assign(X)                    # Sun index for every row
anomalies = model.is_lone(labels)           # rows in Lone Suns

model.suns_          # (k, d) positions of the Suns found by genesis
model.sun_masses_    # (k,) mass each Sun accumulated from the coreset
model.n_suns_        # k plus the Lone Suns registered by assign
```

A runnable demo on synthetic data (five groups plus planted anomalies, compared
with K-Means):

```bash
python examples/quickstart.py
```

### Streaming data larger than memory

`fit` only needs the coreset, and `assign` works on any batch, so a large file
can be processed chunk by chunk:

```python
model = GACA(random_state=0).fit(coreset)          # e.g. 5,000 sampled rows
for chunk in pd.read_csv("big.csv", chunksize=100_000):
    labels = model.assign(scaler.transform(chunk[features].values))
```

Lone Suns registered in one chunk persist, so later copies of the same anomaly
receive the same label.

### Mixture of Experts

Passing a target to `fit` trains one Ridge regression per Sun (logistic
regression with `task_type='classification'`). `predict` routes each row to its
Sun's expert. Suns with fewer than `min_expert_size` training rows predict their
mean. Empty Suns, and rows that no Sun pulls, fall back to a global model.

```python
model = GACA(random_state=0).fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `gamma_clustering` | 1.0 | Gaussian bandwidth γ. The main dial: low γ merges everything into one Sun, high γ fragments into many |
| `epsilon` | `'auto'` | Condensation radius ε. `'auto'` = 0.05/√γ, capped at half the coreset's median nearest-neighbour distance (= 0.05 on the thesis data); a radius larger than the point spacing chains dense structures together |
| `eta` | 0.5 | Damping η of the transport step |
| `sample_size` | 5000 | Coreset size for Solar Genesis |
| `n_iterations` | 20 | Iteration cap for genesis |
| `theta` | 0.5 | Barnes-Hut opening angle (only with `method='barnes_hut'`) |
| `method` | `'exact'` | Genesis kernel sum: `'exact'` (vectorised) or `'barnes_hut'` (thesis) |
| `assignment` | `'pull'` | `'pull'` (member pull, Lone-Sun registration) or `'newton'` (thesis `M/d²`) |
| `kappa` | 1e-3 | Lone-Sun threshold, relative to the median pull inside the coreset |
| `link_tau` | None | Saddle-linking ratio; e.g. 0.6 together with γ ≈ 10 for non-convex clusters |
| `n_neighbors` | 32 | Coreset members used for the pull of each row |
| `lone_share` | 1e-3 | A genesis Sun with at most this share of the coreset counts as a Lone Sun |
| `min_expert_size` | 10 | Minimum rows for a Sun to get its own regression expert |
| `random_state` | None | Seed for the coreset draw |
| `verbose` | False | Print progress |

The lower-level functions are also exported: `solar_genesis` (the simulation on
its own; `return_members=True` gives the Sun of every input row, `plateau=w` adds
the plateau stopping rule), `assign` (Newtonian assignment given Suns and
masses), `kernel_pull`, `saddle_link`, `GACANode` and
`merge_connected_components`.

## Practical guidance

These apply to `GACA` used directly; `AutoGACA` handles the first three.

- **Scale your features.** γ is defined relative to the scale of the space.
- **Mind the dimension.** See [above](#why-max_dims-can-be-10-or-more): with γ
  scaled to the data, 10 or more informative columns work. Many noise columns
  need PCA first.
- **Choose γ from the plateau.** Sweep γ and pick a value where the Sun count is
  stable (`select_gamma` does this). γ = 1.0 sat in the stable band on the
  thesis data.
- **Genesis is cheap now.** With the exact backend, genesis takes about 0.6 s on
  a 5,000-point coreset and 28 s on 40,000 (the thesis reports 22.5 s and 370 s).
- **Non-convex clusters: link.** For crescents, rings or elongated groups, use
  `link_tau=0.6`. Linking joins clusters connected by a dense bridge, so it also
  joins clusters that overlap; leave it off when groups touch.
- **Anomaly sensitivity.** `kappa` sets how little pull makes a row a Lone Sun.
  Raising γ also makes more rows lone, since the kernel narrows; at very high γ
  a small coreset cannot cover sparse tails and genuine rows start to be
  flagged.

## Results from the thesis

| | Result |
|---|---|
| Scalability | 10⁷ records: 23.5 s total, 0.81 µs per record for assignment, peak memory flat at 707 MB |
| vs HDBSCAN | 1,766× faster at 10⁷ records; HDBSCAN is faster below ~3×10⁵ |
| Structure | Largest-Sun share 0.9203 to 0.9209 while streaming from 10⁵ to 10⁷ records |
| Fragmentation | 5 effective clusters vs 91 (HDBSCAN) and 94 (Mean-Shift) on 20k records |
| Anomalies | 92% of sampled planted anomalies isolated as Lone Suns; 100% of unsampled ones flagged by the OOD rule |
| Robustness | Under heavy feature noise (σ = 2), a K-Means mixture fell to R² = −0.004 while the GACA mixture kept 0.012 |

![Bandwidth sweep: phase transition and downstream utility](https://raw.githubusercontent.com/henrivuorinen/GACA/main/figures/phase_transition.png)

![Runtime versus dataset size](https://raw.githubusercontent.com/henrivuorinen/GACA/main/figures/scalability_comparison.png)

## Improvements since the thesis

Measured with `experiments/improvements_benchmark.py` on synthetic data (20,000
rows, coreset 2,000, γ = 1, 4 seeds, the nine planted anomaly groups of
Sec. 6.4). *Detected* is the share of anomaly groups whose every copy ends up
in a Lone Sun; *FP* is the share of genuine rows flagged; *R²* is the Mixture of
Experts on a fresh test set with new anomalies.

| Data | Method | Detected | FP | ARI | R² |
|---|---|---|---|---|---|
| blobs | thesis (`M/d²`) | 0.19 | 0.00% | 0.990 | 0.963 |
| blobs | thesis + OOD95 rule | 1.00 | 4.87% | 0.990 | 0.936 |
| blobs | **pull (new default)** | 0.97 | **0.00%** | **0.994** | **0.974** |
| company-like | thesis (`M/d²`) | 0.22 | 0.01% | 0.080 | 0.397 |
| company-like | thesis + OOD95 rule | 1.00 | 4.89% | 0.080 | 0.395 |
| company-like | **pull (new default)** | **1.00** | 0.35% | 0.173 | **0.579** |
| company-like | pull, γ = 5, `link_tau=0.6` | 1.00 | 5.76% | 0.604 | 0.738 |

What changed and why:

1. **Exact vectorised genesis.** Same Suns as the Barnes-Hut version (to within a
   unit of mass), no approximation error, and 90 to 140 times faster in this
   implementation. This also settles the "approximation bounds" limitation: the
   default operator is exact.
2. **Anomalies no longer depend on the coreset.** The thesis found that an
   unsampled anomaly is never isolated, and that the OOD rule that catches them
   also flags 5% of genuine rows by construction (about 1,000 false alarms per
   20,000 rows). The pull test catches them at 0 to 0.35% false positives,
   independently of the coreset size, and groups the copies of one anomaly
   into a single Lone Sun.
3. **Assignment follows the dynamics.** Routing by the pull of each Sun's
   members instead of `M_k / d²` improved the downstream Mixture of Experts
   (R² 0.40 to 0.58 on the company-like data), and rows that no Sun pulls are
   sent to the global model instead of a wrong local expert.
4. **Saddle linking** recovers non-convex clusters that the flat output could
   not express (two moons and concentric circles at γ = 10: ARI 0.2 to 0.5
   without, 0.99 to 1.0 with `link_tau=0.6`, over 4 seeds), and widens the
   usable γ range (five blobs at ARI ≥ 0.94 for γ = 3 to 10, where unlinked
   γ = 10 shatters them into 300 Suns). It is not uniformly safe: at γ = 1 it
   can join adjacent groups, and at γ = 30 it under-merges.
5. **Relative Lone-Sun threshold** (`kappa` as a fraction of the median pull) so
   that one setting works across bandwidths, dimensions and coreset sizes.
6. **Lone Sun by share, not by mass 1.** Once two copies of a rare group are
   sampled, the thesis definition (mass 1) no longer counts it as an anomaly.

Remaining caveats: the 5σ misses in the table are planted points within about
one unit of genuine data, where the kernel does not consider them separate.
The real company data was not available for these tests, so the thesis
numbers should be re-run before the new defaults are trusted on that data.

## Repository layout

```
gaca/            the package: Solar Genesis and assignment (genesis.py), the
                 GACA estimator (model.py), AutoGACA (auto.py), the HTML report
                 (report.py) and the gaca command (cli.py)
tests/           pytest suite (mass conservation, Lone Suns, estimator, AutoGACA, CLI)
examples/        runnable demo on synthetic data; sdss/ fetches public SDSS
                 data and shows AutoGACA on it
experiments/     the scripts behind every thesis table and figure, plus
                 improvements_benchmark.py (thesis vs current defaults)
figures/         images used in this README
data/            empty; put your own data here (git-ignored)
```

## Reproducing the thesis experiments

The thesis used proprietary company data, which is not included. The scripts in
`experiments/` run on any CSV with the columns below. Run them from the
repository root; outputs (CSV, PDF, PNG) are written to the current directory.

The company-data loader (`experiments/process_data.py`) builds five features:

| Feature | Built from |
|---|---|
| `log_emp` | `EMPLOYEES_COUNT`, falling back to the midpoint of the `SIZE` band (`1-10`, `11-50`, ...) |
| `log_reviews` | `TOTAL_GMAPS_REVIEW_COUNT` |
| `age` | `FOUNDED` (year) |
| `LATITUDE`, `LONGITUDE` | as is |

The regression target is `GMAPS_REVIEWS_AVERAGE`. The characterisation scripts
also use `NAME`, `GMAPS_PRIMARY_CATEGORY`, `INDUSTRY` and `COUNTRY` when present.
The dimensionality experiment accepts any wide table of numeric columns.

| Script | Thesis result |
|---|---|
| `sweep_gamma.py` → `plot_phase_transition.py` | γ phase transition (Sec. 5.1) |
| `sweep_epsilon.py` | ε sensitivity |
| `benchmark_clustering.py` | GACA vs K-Means, Mean-Shift, HDBSCAN (Table 5.4) |
| `final_benchmark_test.py` | Mixture of Experts under feature corruption (RQ3) |
| `characterize_suns.py`, `plot_suns.py` | What the Suns contain (Sec. 5.2.1) |
| `anomaly_recovery.py` | Planted-anomaly recovery through the coreset |
| `convergence_check.py` | Long-run convergence and plateaus |
| `dimensionality_experiment.py` → `plot_dimensionality.py` | Curse of dimensionality and PCA rescue (Sec. 4.3.2) |
| `run_scalability_test.py` | 10⁷-record streaming run (Ch. 6) |
| `scalability_comparison.py` → `plot_scalability_comparison.py` | Runtime vs baselines (Ch. 6) |
| `coreset_sweep.py` → `plot_coreset.py` | Coreset-size sensitivity (Ch. 6) |
| `plot_accretion_evolution.py` | The accretion figures |
| `make_tables.py` | Writes every thesis table as LaTeX from the result CSVs |

Example:

```bash
python experiments/sweep_gamma.py
python experiments/plot_phase_transition.py sweep_results_eta0.5.csv
```

## Limitations

- The Barnes-Hut acceptance rule (`method='barnes_hut'`) has no uniform error
  bound for a Gaussian kernel. The default exact backend avoids it, at O(n²) per
  iteration, which is fine up to coresets of a few tens of thousands.
- There is no global optimality guarantee, and the result is the state at the
  stopping time (the stopping rule is part of the algorithm).
- A uniform coreset can miss rare structure. Isolated anomalies are recovered at
  assignment time, but a small genuine cluster missed by the coreset becomes a
  set of Lone Suns rather than a regular Sun.
- Only low-dimensional inputs are practical; use a projection first.

## Citation

```bibtex
@mastersthesis{vuorinen2026gaca,
  author = {Vuorinen, Henri J.},
  title  = {The Gravitational Accretion Clustering Algorithm (GACA): Scalable,
            Physics-Inspired Clustering for Large Datasets},
  school = {Aalto University},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](https://github.com/henrivuorinen/GACA/blob/main/LICENSE).
