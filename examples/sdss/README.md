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
    --scale none --bandwidth 1 --sample-size 20000
```

- **`--scale none`:** positions are already in Mpc, so they are not rescaled.
- **`--bandwidth 1`:** a 1 Mpc kernel, about the size of a cluster core.
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

## Your own data

- **Properties** (colours, magnitudes, shapes, spectra features): run
  `gaca run file.csv --exclude <id and label columns>` and read `report.html`.
- **Positions:** convert to Cartesian coordinates in physical units, compress
  the line of sight if they are redshift-based, and use `--scale none
  --bandwidth <length>`.
