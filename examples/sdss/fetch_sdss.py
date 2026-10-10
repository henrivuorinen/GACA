"""
Download two public SDSS test tables for trying GACA on astronomy data.

Both come from the Sloan Digital Sky Survey (DR18) through its public SQL web
service; no account is needed. The files are written to data/ (git-ignored).

    data/sdss_objects.csv
        100,000 spectroscopically confirmed objects: position, redshift, the
        five dereddened magnitudes u g r i z, the four colours u-g, g-r, r-i,
        i-z, and the half-light radius. SDSS's own classification (class:
        GALAXY / STAR / QSO, and subClass) is included so the clusters can be
        checked against it; leave those columns out of the clustering.

    data/sdss_galaxy_positions.csv
        Nearby galaxies (redshift 0.01 to 0.05) in the region holding the Coma
        and Leo clusters, converted to 3-D positions in Mpc (x, y, z), using
        distance = c * redshift / H0 with H0 = 70 km/s/Mpc (fine at these
        redshifts). Clusters here should be real galaxy clusters and groups.
        Also tx, ty, los: flat-sky transverse coordinates in Mpc around
        RA 195, Dec 30, and the line-of-sight distance divided by 10. Galaxies
        orbiting inside a cluster smear its redshift distance along the line of
        sight by many Mpc (the "finger of God"); compressing that axis keeps a
        cluster in one piece, as friends-of-friends group finders do with a
        longer linking length along the line of sight.

    data/sdss_timeline.csv
        60,000 objects in the order they were observed (column mjd, the
        observation date): 30,000 from the original SDSS survey (2000 to
        2008), then 30,000 from BOSS (from December 2009), which targeted
        more distant galaxies and quasars. A real, documented change in the
        data, for trying drift monitoring.

    python examples/sdss/fetch_sdss.py
"""
import io
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data")

OBJECTS = """
SELECT TOP 100000 s.specObjID, s.ra, s.dec, s.z AS redshift, s.class, s.subClass,
       s.dered_u AS u, s.dered_g AS g, s.dered_r AS r, s.dered_i AS i, s.dered_z AS z,
       p.petroR50_r
FROM SpecPhoto s JOIN PhotoObj p ON p.objID = s.objID
WHERE s.zWarning = 0 AND s.ra BETWEEN 120 AND 240 AND s.dec BETWEEN 0 AND 60
"""

POSITIONS = """
SELECT s.specObjID, s.ra, s.dec, s.z AS redshift, s.dered_r AS r,
       s.dered_g - s.dered_r AS g_r
FROM SpecPhoto s
WHERE s.class = 'GALAXY' AND s.zWarning = 0 AND s.z BETWEEN 0.01 AND 0.05
  AND s.ra BETWEEN 150 AND 240 AND s.dec BETWEEN 0 AND 60
"""


TIMELINE = """
SELECT TOP {n} s.specObjID, s.plate, s.mjd, s.ra, s.dec, s.z AS redshift, s.class, s.subClass,
       s.dered_u AS u, s.dered_g AS g, s.dered_r AS r, s.dered_i AS i, s.dered_z AS z
FROM SpecPhoto s
WHERE s.zWarning = 0 AND s.ra BETWEEN 120 AND 240 AND s.dec BETWEEN 0 AND 60 AND {where}
"""


def add_colours(df):
    mags = df[["u", "g", "r", "i", "z"]].where(df[["u", "g", "r", "i", "z"]] > -1000)
    df["u_g"] = mags["u"] - mags["g"]
    df["g_r"] = mags["g"] - mags["r"]
    df["r_i"] = mags["r"] - mags["i"]
    df["i_z"] = mags["i"] - mags["z"]
    return df


def query(sql, attempts=4):
    """Run a query, retrying when the public server is temporarily slow."""
    params = urllib.parse.urlencode({"cmd": " ".join(sql.split()), "format": "csv"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(f"{URL}?{params}", timeout=120) as resp:
                text = resp.read().decode("utf-8")
            break
        except (TimeoutError, OSError) as err:
            if attempt == attempts:
                raise RuntimeError(f"SDSS server did not respond after {attempts} attempts; "
                                   "try again later") from err
            wait = 10 * attempt
            print(f"  server slow ({err.__class__.__name__}), retrying in {wait} s...")
            time.sleep(wait)
    if text.startswith("{"):
        raise RuntimeError(f"SDSS query failed: {text[:300]}")
    return pd.read_csv(io.StringIO(text), skiprows=1)        # first line is '#Table1'


def main():
    os.makedirs(OUT, exist_ok=True)

    print("Downloading SDSS objects...")
    obj = add_colours(query(OBJECTS))
    path = os.path.join(OUT, "sdss_objects.csv")
    obj.to_csv(path, index=False)
    print(f"  {len(obj):,} rows -> {path}")
    print("  " + ", ".join(f"{k}: {v:,}" for k, v in obj["class"].value_counts().items()))

    print("Downloading nearby galaxy positions...")
    pos = query(POSITIONS)
    d = 299792.458 * pos["redshift"] / 70.0                  # Mpc
    ra, dec = np.radians(pos["ra"]), np.radians(pos["dec"])
    pos["x"] = d * np.cos(dec) * np.cos(ra)
    pos["y"] = d * np.cos(dec) * np.sin(ra)
    pos["z"] = d * np.sin(dec)
    pos["tx"] = d * np.radians(pos["ra"] - 195.0) * np.cos(dec)
    pos["ty"] = d * np.radians(pos["dec"] - 30.0)
    pos["los"] = d / 10.0
    path = os.path.join(OUT, "sdss_galaxy_positions.csv")
    pos.to_csv(path, index=False)
    print(f"  {len(pos):,} galaxies -> {path}")

    print("Downloading the survey timeline (original SDSS, then BOSS)...")
    legacy = query(TIMELINE.format(n=30000, where="s.plate < 3000"))
    boss = query(TIMELINE.format(n=30000, where="s.plate > 3500"))
    tl = add_colours(pd.concat([legacy, boss], ignore_index=True))
    tl = tl.sort_values(["mjd", "specObjID"]).reset_index(drop=True)
    path = os.path.join(OUT, "sdss_timeline.csv")
    tl.to_csv(path, index=False)
    print(f"  {len(tl):,} objects, observed {tl.mjd.min()} to {tl.mjd.max()} (MJD) -> {path}")


if __name__ == "__main__":
    main()
