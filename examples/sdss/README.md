# Trying GACA on public astronomy data (SDSS)

Two test tables from the Sloan Digital Sky Survey, downloaded through its
public SQL service (no account needed):

```bash
pip install -e .                          # in a clone of the repository
python examples/sdss/fetch_sdss.py        # writes data/sdss_objects.csv and
                                          # data/sdss_galaxy_positions.csv
```

The objects query takes the first 100,000 matching rows in whatever order the
server returns them, so the exact rows (and the numbers below) vary slightly
between downloads.

## 1. Object properties: populations and odd objects

100,000 spectroscopically confirmed objects with redshift, magnitudes, colours
and size. SDSS's classification (`class`: galaxy, star or quasar, and
`subClass`) is in the file to check the result against, so leave it out:

```bash
gaca run data/sdss_objects.csv --exclude ra,dec,class,subClass
```

Everything else is automatic. In our run (about 15 s):

- **Columns:** `specObjID` was recognised as an identifier, and the `-9999`
  placeholders SDSS uses for missing measurements were treated as missing.
- **Dimensions:** 11 columns were projected to 5-D with PCA.
- **Clusters:** 28 clusters, which line up with physical populations GACA was
  never told about:
  - three quasar clusters (100% quasars), separated by redshift (median
    z ≈ 0.9, 1.25 and 1.6)
  - cool M-dwarf stars (100% stars, M5 to M7) and hot A/F stars (82 to 95%
    stars)
  - the main nearby galaxy population (97% galaxies) and distant luminous
    galaxies (z ≈ 0.4)
- **Agreement:** adjusted Rand index against galaxy/star/quasar is 0.48. GACA
  finds finer populations than three classes.
- **Anomalies (3.8%):** the highest-scoring ones are mostly broken photometry,
  such as an object with g = 18.3 but r = 30.6. Those are rows a survey
  scientist would want flagged.

## 2. Galaxy positions: finding groups and clusters

58,761 galaxies at redshift 0.01 to 0.05 around the Coma cluster. This is a
different task from the first. Most galaxies are in no group, groups hold well
under 1% of galaxies each, and the right scale is physical. So the scale is set
by hand rather than chosen automatically:

```bash
gaca run data/sdss_galaxy_positions.csv --columns tx,ty,los \
    --scale none --resolution 3.3 --sample-size 20000
```

- **`--scale none`:** positions are already in Mpc, so they are not rescaled.
- **`--resolution 3.3`:** keep groups whose centres are more than 3.3 Mpc
  apart separate. GACA turns this into a kernel width (here 1 Mpc, about the
  size of a cluster core) using the resolution law in `docs/theory.md`.
  `--bandwidth 1` sets the same kernel directly.
- **`tx, ty, los`:** sky-plane coordinates in Mpc, plus the line-of-sight
  distance divided by 10. Galaxies orbiting inside a cluster at ~1,000 km/s
  smear its redshift distance by over ±10 Mpc along the line of sight (the
  "finger of God"). In plain `x, y, z` GACA cuts Coma into three or four slices
  at different redshifts. With the line of sight compressed, it is one group.

Result (about 20 s): the largest group is **Coma**, with about 1,060 galaxies
(the literature counts roughly a thousand spectroscopic members). It contains
all 698 galaxies within 1.5° of Coma's centre. The Leo cluster (Abell 1367)
region follows. "Anomalies" here are galaxies in no group, the field
population: about 70% of galaxies at this scale.

Comparing against friends-of-friends group catalogues (e.g. Tempel et al. 2017
for SDSS) would be the natural next test.

## 3. Try the newer features, with known answers

These commands exercise the features added in 1.4 and put the results in the
report (`<out>/report.html`). Each compares against something known.

**Compare against SDSS's own classes and against other methods**:

```bash
gaca run data/sdss_objects.csv --exclude ra,dec,subClass \
    --truth class --baselines --out data/sdss_objects_gaca
```

- `--truth class` keeps the class column out of the clustering and adds a
  *Comparison with known labels* section: ARI and NMI for GACA's main view,
  linked view and best hierarchy level.
- `--baselines` adds K-Means (given the true number of classes) and HDBSCAN
  on the same preprocessed data.
- The cluster table now has a **rule** for each cluster in plain thresholds.

In our run, HDBSCAN matched the three classes best (ARI 0.61), GACA's levels
scored 0.47 to 0.50, and K-Means 0.35. GACA's clusters are purer but finer:
it separates quasars by redshift, which the three-class labels do not reward.

**Detect a real change in the data** (drift):

```bash
gaca run data/sdss_timeline.csv --exclude ra,dec,mjd,plate,subClass --truth class \
    --chunksize 5000 --fit-rows 15000 --fit-first --out data/sdss_timeline_gaca
```

The file is in observation order. Its first half comes from the original SDSS
survey and its second half from BOSS (from December 2009), which targeted more
distant galaxies and quasars (median redshift 0.1, then 0.5). `--fit-first`
fits on the first 15,000 rows, so the *Drift across the file* section charts
how each chunk differs from the start. In our run, chunks 0 to 5 were not
flagged, and all six BOSS chunks were (cluster mix moved by 0.17 to 0.20;
anomaly rate 1% → 17 to 21%, the new kinds of objects).

**Set the resolution in physical units**:

```bash
gaca run data/sdss_galaxy_positions.csv --columns tx,ty,los \
    --scale none --resolution 3.3 --sample-size 20000 --out data/sdss_groups_gaca
```

This keeps galaxy groups more than 3.3 Mpc apart separate. The largest group
is Coma (about 1,000 galaxies).

## Your own data

- **Properties** (colours, magnitudes, shapes, spectra features): run
  `gaca run file.csv --exclude <id and label columns>` and read `report.html`.
- **Positions:** convert to Cartesian coordinates in physical units, compress
  the line of sight if they are redshift-based, and use `--scale none
  --bandwidth <length>`.
