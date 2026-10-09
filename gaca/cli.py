"""Command line interface.

    gaca run data.csv                         # cluster every row, write report
    gaca run data.csv --exclude id,name       # leave columns out
    gaca run big.csv --chunksize 200000       # stream a file larger than memory
    gaca run data.csv --scale none --bandwidth 1  # positions in physical units, 1-unit kernel

Writes to ``<input>_gaca/`` (or ``--out``):

    labels.csv     the input rows with gaca_cluster, gaca_anomaly,
                   gaca_anomaly_group and gaca_anomaly_score appended
    report.html    clusters, anomalies, how the resolution was chosen
    summary.json   the settings and counts, machine readable
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import __version__
from .auto import AutoGACA


def _split(s):
    return [c.strip() for c in s.split(',') if c.strip()] if s else None


def _reservoir(path, n, chunksize, seed):
    """Uniform random sample of n rows from a CSV, in one streaming pass.

    Every row gets a random key; the n rows with the smallest keys form a
    uniform sample without replacement."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    sample, keys, seen = None, np.empty(0), 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        k = rng.random(len(chunk))
        seen += len(chunk)
        sample = chunk if sample is None else pd.concat([sample, chunk], ignore_index=True)
        keys = np.r_[keys, k]
        if len(sample) > n:
            keep = np.argpartition(keys, n)[:n]
            sample, keys = sample.iloc[keep].reset_index(drop=True), keys[keep]
    return sample, seen


def run(args):
    try:
        import pandas as pd
    except ImportError:
        sys.exit("gaca run needs pandas: pip install pandas")

    out = args.out or os.path.splitext(os.path.basename(args.input))[0] + "_gaca"
    os.makedirs(out, exist_ok=True)
    gamma = args.gamma if args.gamma == 'auto' else float(args.gamma)
    auto = AutoGACA(columns=_split(args.columns), exclude=_split(args.exclude),
                    scale=args.scale, max_dims=args.max_dims, gamma=gamma,
                    link_tau=args.link, sample_size=args.sample_size,
                    bandwidth=args.bandwidth,
                    random_state=args.seed, verbose=not args.quiet)
    t0 = time.time()
    labels_path = os.path.join(out, "labels.csv")

    if args.chunksize:
        sample, n_total = _reservoir(args.input, args.fit_rows, args.chunksize, args.seed)
        if not args.quiet:
            print(f"Fitting on a random sample of {len(sample):,} of {n_total:,} rows")
        auto.fit(sample)
        counts, n_anom = np.zeros(auto.n_clusters_, dtype=np.int64), 0
        first = True
        for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
            res = auto.assign(chunk)
            pd.concat([chunk.reset_index(drop=True), res], axis=1).to_csv(
                labels_path, mode='w' if first else 'a', header=first, index=False)
            first = False
            lab = res['gaca_cluster'].to_numpy()
            counts += np.bincount(lab[lab >= 0], minlength=auto.n_clusters_)[:auto.n_clusters_]
            n_anom += int((lab < 0).sum())
        report_data = sample
        note = f"fitted on a random sample of {len(sample):,} rows; report describes that sample"
    else:
        df = pd.read_csv(args.input)
        n_total = len(df)
        auto.fit(df)
        pd.concat([df.reset_index(drop=True), auto.result_], axis=1).to_csv(labels_path, index=False)
        counts = np.bincount(auto.labels_[auto.labels_ >= 0], minlength=auto.n_clusters_)
        n_anom = int(auto.anomaly_.sum())
        report_data = df
        note = None

    title = f"GACA: {os.path.basename(args.input)}"
    auto.report(os.path.join(out, "report.html"), data=report_data, title=title)
    p = auto.preprocessor_
    summary = dict(
        input=os.path.abspath(args.input), rows=int(n_total), version=__version__,
        columns_used=p.used_, column_decisions=p.decisions_,
        clustering_dims=int(p.n_dims_), pca=p.pca_ is not None,
        gamma=auto.gamma_, gamma_auto=auto.gamma_selection_ is not None,
        stable_plateau=None if auto.gamma_selection_ is None else bool(auto.gamma_selection_['found']),
        epsilon=auto.model_.epsilon_, link_tau=auto.link_tau,
        clusters=[dict(cluster=int(k), rows=int(c)) for k, c in enumerate(counts)],
        anomalous_rows=int(n_anom), seconds=round(time.time() - t0, 1), note=note,
        hierarchy=[dict(level=i, gamma=lv['gamma'], clusters=int(lv['k']),
                        in_gaca_cluster=bool(lv['chosen']))
                   for i, lv in enumerate(auto.levels_, start=1)])
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    if not args.quiet:
        print(auto.summary())
        print(f"\nWrote {labels_path}, report.html and summary.json to {out}/ "
              f"in {time.time() - t0:.1f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gaca", description="Gravitational Accretion Clustering")
    ap.add_argument("--version", action="version", version=f"gaca {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="cluster a CSV file and write labels and a report")
    r.add_argument("input", help="CSV file")
    r.add_argument("--columns", help="comma-separated columns to use (default: all usable)")
    r.add_argument("--exclude", help="comma-separated columns to leave out")
    r.add_argument("--max-dims", type=int, default=10,
                   help="project with PCA when more columns than this remain (default 10)")
    r.add_argument("--scale", choices=["robust", "none"], default="robust",
                   help="'none' keeps values as they are, e.g. positions in physical units")
    r.add_argument("--gamma", default="auto", help="kernel parameter gamma, or 'auto' (default)")
    r.add_argument("--bandwidth", type=float, default=None, metavar="H",
                   help="kernel width in data units instead of gamma, e.g. 1.0 for 1 Mpc "
                        "with --scale none (gamma = 1 / (2 H^2))")
    r.add_argument("--link", type=float, default=None, metavar="TAU",
                   help="join Suns connected by dense bridges (e.g. 0.6) for curved clusters")
    r.add_argument("--sample-size", type=int, default=5000, help="coreset size (default 5000)")
    r.add_argument("--chunksize", type=int, default=None,
                   help="stream the file in chunks of this many rows (for files larger than memory)")
    r.add_argument("--fit-rows", type=int, default=100_000,
                   help="with --chunksize: rows sampled to fit on (default 100000)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", help="output directory (default <input>_gaca)")
    r.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
