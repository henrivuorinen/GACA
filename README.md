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

![Solar Genesis: 3,000 particles condensing into 7 Suns](figures/accretion_evolution.png)

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
git clone https://github.com/<your-user>/GACA.git
cd GACA
pip install -e .                  # core: numpy, scipy, scikit-learn
pip install -e ".[experiments]"   # + pandas, matplotlib, hdbscan, lightgbm, ...
pip install -e ".[dev]"           # + pytest
```

Python 3.9 or newer.

## Quickstart

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
| `epsilon` | 0.05 | Condensation radius ε. Has little effect on structure; larger is faster |
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

- **Standardise your features.** γ and ε are defined on a standardised scale.
- **Keep dimensionality low.** The Barnes-Hut tree stops helping above roughly
  15 dimensions, and clustering quality collapses with it. Project to about 5
  dimensions with PCA first. In the thesis, 45 raw dimensions took over an hour
  and shattered into 1,377 singletons; PCA to 5 dimensions took 33 seconds.
- **Choose γ from the plateau.** Sweep γ and pick a value where the Sun count is
  stable. γ = 1.0 sat in the stable band on the thesis data.
- **Genesis is cheap now.** With the exact backend, genesis takes about 0.6 s on
  a 5,000-point coreset and 28 s on 40,000 (the thesis reports 22.5 s and 370 s).
- **Non-convex clusters: raise γ and link.** For crescents, rings or elongated
  groups, use a finer bandwidth (γ ≈ 10 worked best) with `link_tau=0.6`. At
  γ = 1 linking is unnecessary and can join adjacent groups; at γ = 30 the
  coreset is too sparse for the bridge estimates and linking under-merges.
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

![Bandwidth sweep: phase transition and downstream utility](figures/phase_transition.png)

![Runtime versus dataset size](figures/scalability_comparison.png)

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
| company-like | pull, γ = 5, `link_tau=0.6` | 1.00 | 2.82% | 0.579 | 0.712 |

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
gaca/            the package: Solar Genesis, assignment, GACA estimator
tests/           pytest suite (mass conservation, Lone Suns, estimator)
examples/        runnable demo on synthetic data
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

MIT, see [LICENSE](LICENSE).
