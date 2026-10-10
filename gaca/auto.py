"""AutoGACA: clustering and anomaly detection for an arbitrary table.

GACA itself needs numeric, scaled, low-dimensional input and a bandwidth. This
module makes those decisions from the data, records each one, and returns
readable output:

    Preprocessor   picks usable columns, imputes, log-transforms skewed ones,
                   scales robustly, and projects with PCA when the table has
                   more than ``max_dims`` columns
    select_gamma   sweeps the bandwidth and picks the most stable plateau of the
                   cluster count (the rule of thesis Sec. 5.2, automated)
    AutoGACA       the whole pipeline: fit, labels, anomaly flags and scores,
                   per-cluster summaries, assignment of new rows, HTML report

    from gaca import AutoGACA
    auto = AutoGACA().fit(df)
    auto.result_         # DataFrame: cluster, anomaly, anomaly score per row
    auto.report("report.html")
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from .genesis import resolution_to_gamma
from .model import GACA

ID_NAME = re.compile(r'(id|idx|index|key|uuid)$', re.IGNORECASE)   # objid, specObjID, row_index
SENTINELS = (-99999.0, -9999.0, -999.0, -99.0, 99.0, 999.0, 9999.0, 99999.0)


def _as_frame(data, columns=None):
    """Return (a DataFrame, or a minimal stand-in without pandas, column names)."""
    if isinstance(data, _ArrayFrame):
        return data, data.columns
    try:
        import pandas as pd
    except ImportError:          # arrays still work without pandas
        pd = None
    if pd is not None and isinstance(data, pd.DataFrame):
        return data, list(map(str, data.columns))
    if isinstance(data, str):
        if pd is None:
            raise ImportError("reading a CSV needs pandas: pip install pandas")
        df = pd.read_csv(data)
        return df, list(map(str, df.columns))
    X = np.asarray(data)
    if X.ndim != 2:
        raise ValueError("data must be 2-D (rows x columns)")
    names = list(columns) if columns is not None else [f"x{i}" for i in range(X.shape[1])]
    if pd is not None:
        df = pd.DataFrame(X, columns=names)
        return df, names
    return _ArrayFrame(X, names), names


class _ArrayFrame:
    """Minimal column access for plain arrays when pandas is not installed."""
    def __init__(self, X, names):
        self._X, self.columns, self.shape = X, names, X.shape

    def __getitem__(self, name):
        return self._X[:, self.columns.index(name)]

    def __len__(self):
        return self._X.shape[0]


def _to_numeric(col):
    """Column as float with NaN for unparseable entries, and the parsed share."""
    try:
        import pandas as pd
        v = pd.to_numeric(pd.Series(col), errors='coerce').to_numpy(dtype=float)
    except ImportError:
        v = np.array([_float_or_nan(x) for x in col], dtype=float)
    present = np.array([x is not None and not (isinstance(x, float) and np.isnan(x))
                        and str(x).strip() != '' for x in col]) if col.dtype == object \
        else ~np.isnan(np.asarray(col, dtype=float))
    parsed = np.isfinite(v).sum() / max(present.sum(), 1)
    return v, parsed


def _float_or_nan(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _skew(v):
    s = v.std()
    return 0.0 if s == 0 else float(np.mean(((v - v.mean()) / s) ** 3))


def _path_key(name):
    """Sort key for path-style cluster names: '0.2' before '0.10'."""
    return tuple(int(p) for p in name.split('.'))


def _within_mode_spread(v, spread, random_state=0, max_n=5000, min_weight=0.05,
                        min_sd_frac=0.01, min_d=2.0):
    """The typical spread inside the groups of a column whose values form
    clearly separated groups, or None.

    A 1-D Gaussian mixture with 1 to 3 components is chosen by BIC. Adjacent
    components that overlap (Ashman's D < ``min_d``) are merged, so a group that
    the mixture happened to split counts once. Components holding less than
    ``min_weight`` of the rows (a few outliers must not set the scale) or with
    almost no spread (a spike, such as an imputed default value) are ignored.
    If two or more well-separated groups remain, their pooled standard
    deviation is returned."""
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(random_state)
    x = v if len(v) <= max_n else rng.choice(v, max_n, replace=False)
    x = x.reshape(-1, 1)
    if len(np.unique(x)) < 10:
        return None
    best, best_bic = None, np.inf
    for k in (1, 2, 3):
        g = GaussianMixture(k, n_init=2, random_state=random_state).fit(x)
        b = g.bic(x)
        if b < best_bic:
            best, best_bic = g, b
    if best.n_components == 1:
        return None
    w = best.weights_.copy()
    mu = best.means_[:, 0].copy()
    var = best.covariances_.reshape(-1).copy()
    comps = sorted(zip(mu, var, w))
    merged = [list(comps[0])]
    for m2, v2, w2 in comps[1:]:
        m1, v1, w1 = merged[-1]
        d = np.sqrt(2) * abs(m2 - m1) / np.sqrt(v1 + v2)
        if d < min_d:                               # same group: moment-matched merge
            wt = w1 + w2
            m = (w1 * m1 + w2 * m2) / wt
            v = (w1 * (v1 + (m1 - m) ** 2) + w2 * (v2 + (m2 - m) ** 2)) / wt
            merged[-1] = [m, v, wt]
        else:
            merged.append([m2, v2, w2])
    groups = [(m, v, wt) for m, v, wt in merged
              if wt >= min_weight and np.sqrt(v) >= min_sd_frac * spread]
    if len(groups) < 2:
        return None
    for (m1, v1, _), (m2, v2, _) in zip(groups, groups[1:]):
        if np.sqrt(2) * abs(m2 - m1) / np.sqrt(v1 + v2) < min_d:
            return None
    wsum = sum(wt for _, _, wt in groups)
    return float(np.sqrt(sum(wt * v for _, v, wt in groups) / wsum))


def _gd_rank(X):
    """Number of components above the noise floor (Gavish & Donoho, 2014)."""
    n, d = X.shape
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    b = min(n, d) / max(n, d)
    omega = 0.56 * b ** 3 - 0.95 * b ** 2 + 1.82 * b + 1.43
    return int((s > omega * np.median(s)).sum())


class Preprocessor:
    """Turns a raw table into the scaled, low-dimensional space GACA runs in.

    Every decision is recorded in ``decisions_`` (column -> what was done) so the
    report can show exactly what the clustering saw.

    Parameters
    ----------
    columns : list of str, optional
        Columns to use. Default: every column that is usable.
    exclude : list of str, optional
        Columns never to use (identifiers, targets, free text).
    scale : {'robust', 'none'}
        'robust' log-transforms count-like columns (non-negative, skewed, wide
        range) and scales each column by its median and interquartile range. 'none' uses the values as they are, for
        data whose distances are already meaningful (e.g. coordinates in km);
        gamma is then in the data's own units.
    max_dims : int
        If more columns remain than this, project with PCA to between 5 (or
        ``max_dims`` if smaller) and ``max_dims`` components, choosing the number
        by the noise floor of the singular values.
    max_missing : float
        Columns with a larger share of missing values are dropped.
    separate_modes : bool
        With ``scale='robust'``, scale a column whose values form clearly
        separated groups by the spread inside the groups rather than by its
        interquartile range, which would span the gaps and squeeze the column.
    """

    def __init__(self, columns=None, exclude=None, scale='robust', max_dims=10,
                 max_missing=0.5, separate_modes=True, random_state=0):
        self.columns = columns
        self.separate_modes = separate_modes
        self.exclude = exclude
        self.scale = scale
        self.max_dims = max_dims
        self.max_missing = max_missing
        self.random_state = random_state

    def fit(self, data):
        if self.scale not in ('robust', 'none'):
            raise ValueError("scale must be 'robust' or 'none'")
        df, names = _as_frame(data)
        wanted = list(self.columns) if self.columns is not None else names
        missing = [c for c in wanted if c not in names]
        if missing:
            raise KeyError(f"columns not in the data: {missing}")
        exclude = set(self.exclude or [])
        self.decisions_ = {}
        self.used_, self.log_, self.center_, self.spread_, self.fill_ = [], {}, {}, {}, {}
        self.sentinels_ = {}

        for c in wanted:
            if c in exclude:
                self.decisions_[c] = 'excluded by user'
                continue
            raw = np.asarray(df[c])
            if raw.dtype == bool:
                raw = raw.astype(float)
            v, parsed = _to_numeric(raw)
            finite = np.isfinite(v)
            if parsed < 0.95:
                self.decisions_[c] = 'not used: non-numeric'
                continue
            if 1 - finite.mean() > self.max_missing:
                self.decisions_[c] = f'not used: {100 * (1 - finite.mean()):.0f}% missing'
                continue
            vf = v[finite]
            if len(np.unique(vf)) <= 1:
                self.decisions_[c] = 'not used: constant'
                continue
            if self.columns is None and self._looks_like_id(c, vf, len(v)):
                self.decisions_[c] = 'not used: looks like an identifier'
                continue

            notes = []
            # Placeholder codes for "no measurement" (-9999 and the like) would
            # otherwise be the most extreme values in the column and become
            # anomalies. A code counts as a placeholder when it repeats and lies
            # far outside the range of the other values.
            for code in SENTINELS:
                hit = vf == code
                if hit.sum() >= 2:
                    rest = vf[~hit]
                    lo, hi = np.percentile(rest, [0.5, 99.5]) if len(rest) else (code, code)
                    span = max(hi - lo, 1e-12)
                    if code < lo - 3 * span or code > hi + 3 * span:
                        v = np.where(v == code, np.nan, v)
                        notes.append(f'{int(hit.sum())} x {code:g} treated as missing')
                        self.sentinels_.setdefault(c, []).append(code)
            finite = np.isfinite(v)
            vf = v[finite]
            self.fill_[c] = float(np.median(vf))
            if not finite.all():
                notes.append(f'{100 * (1 - finite.mean()):.1f}% missing -> median')
            v = np.where(finite, v, self.fill_[c])

            mode = None
            if self.scale == 'robust':
                # Log only count- or flux-like columns: non-negative, skewed, and
                # spanning orders of magnitude. A log would also pull genuine
                # outliers back towards the data, so it is not a generic fix
                # for skew.
                sk = _skew(v)
                p50, p99 = np.percentile(v, [50, 99])
                if v.min() >= 0 and sk > 1.0 and p99 > 10 * max(p50, 1e-12):
                    mode = 'log1p'
                    notes.append(f'skew {sk:.1f}, wide range -> log1p')
                v = self._apply_log(v, mode)
                med = float(np.median(v))
                q75, q25 = np.percentile(v, [75, 25])
                spread = (q75 - q25) / 1.349           # = sd for normal data
                if spread <= 0:
                    spread = float(v.std())
                    notes.append('IQR 0 -> scaled by sd')
                # A column whose values form separate groups has an interquartile
                # range that spans the gap between them; scaling by it squeezes
                # the column and can merge the subgroups along it. Such columns
                # are scaled by the spread inside the groups instead (never by
                # more than the IQR, so no column is down-weighted by this).
                if self.separate_modes:
                    inner = _within_mode_spread(v, spread, self.random_state)
                    if inner is not None and inner < spread:
                        notes.append(f'separate groups -> scaled by within-group '
                                     f'spread ({inner:.3g} instead of {spread:.3g})')
                        spread = inner
                self.center_[c], self.spread_[c] = med, spread
            else:
                self.center_[c], self.spread_[c] = 0.0, 1.0
            self.log_[c] = mode
            self.used_.append(c)
            self.decisions_[c] = 'used' + (f" ({'; '.join(notes)})" if notes else '')

        if not self.used_:
            raise ValueError("no usable numeric columns; pass columns= explicitly")

        Z = self._scaled(df)
        self.pca_ = None
        if Z.shape[1] > self.max_dims:
            lo = min(5, self.max_dims)
            k = int(np.clip(_gd_rank(Z), lo, self.max_dims))
            self.pca_ = PCA(n_components=k, random_state=self.random_state).fit(Z)
        self.n_dims_ = self.pca_.n_components_ if self.pca_ is not None else Z.shape[1]
        return self

    @staticmethod
    def _looks_like_id(name, v, n):
        integral = np.all(v == np.round(v))
        unique = len(np.unique(v)) == n
        if not (integral and unique):
            return False
        monotone = np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)
        return bool(ID_NAME.search(name)) or monotone

    @staticmethod
    def _apply_log(v, mode):
        if mode == 'log1p':
            return np.log1p(np.maximum(v, 0))
        return v

    def _scaled(self, df):
        cols = []
        for c in self.used_:
            v, _ = _to_numeric(np.asarray(df[c]).astype(float) if np.asarray(df[c]).dtype == bool
                               else np.asarray(df[c]))
            for code in self.sentinels_.get(c, ()):
                v = np.where(v == code, np.nan, v)
            v = np.where(np.isfinite(v), v, self.fill_[c])
            v = self._apply_log(v, self.log_[c])
            cols.append((v - self.center_[c]) / self.spread_[c])
        return np.column_stack(cols)

    def transform(self, data):
        df, _ = _as_frame(data)
        Z = self._scaled(df)
        return self.pca_.transform(Z) if self.pca_ is not None else Z

    def scaled(self, data):
        """Scaled columns before any PCA (used to describe clusters)."""
        df, _ = _as_frame(data)
        return self._scaled(df)

    def values(self, data):
        """The used columns in their original units, with placeholders and
        missing values replaced by the column median (used for cluster rules)."""
        df, _ = _as_frame(data)
        cols = []
        for c in self.used_:
            raw = np.asarray(df[c])
            v, _ = _to_numeric(raw.astype(float) if raw.dtype == bool else raw)
            for code in self.sentinels_.get(c, ()):
                v = np.where(v == code, np.nan, v)
            cols.append(np.where(np.isfinite(v), v, self.fill_[c]))
        return np.column_stack(cols)


def _fmt_threshold(x):
    """A readable threshold: 3 significant digits, no exponent for ordinary sizes."""
    if x == 0 or 1e-3 <= abs(x) < 1e6:
        return f"{x:.3g}" if abs(x) < 1000 else f"{x:,.0f}"
    return f"{x:.2e}"


def cluster_rule(V, names, member, max_depth=3, max_rows=20000, random_state=0):
    """A short rule in the original column units that picks out a cluster.

    A shallow decision tree separates the cluster's rows (``member``) from the
    rest; the leaf predicting the cluster with the best F1 score gives the
    rule, its conditions on one column merged into an interval. Returns
    dict(text, conditions, precision, recall) measured on all rows, or None
    if the cluster is empty."""
    from sklearn.tree import DecisionTreeClassifier
    if not member.any():
        return None
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(V))
    if len(V) > max_rows:            # keep every member up to half the sample
        pos, neg = idx[member], idx[~member]
        pos = rng.choice(pos, min(len(pos), max_rows // 2), replace=False)
        neg = rng.choice(neg, min(len(neg), max_rows - len(pos)), replace=False)
        idx = np.r_[pos, neg]
    tree = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced',
                                  min_samples_leaf=max(5, int(0.02 * member[idx].sum())),
                                  random_state=random_state).fit(V[idx], member[idx])
    t = tree.tree_
    best = None

    def walk(node, conds):
        nonlocal best
        if t.children_left[node] == -1:
            if t.value[node][0].argmax() != 1:
                return
            mask = np.ones(len(V), bool)
            for j, op, thr in conds:
                mask &= (V[:, j] <= thr) if op == '<=' else (V[:, j] > thr)
            tp = (mask & member).sum()
            if tp == 0:
                return
            prec, rec = tp / mask.sum(), tp / member.sum()
            f1 = 2 * prec * rec / (prec + rec)
            if best is None or f1 > best[0]:
                best = (f1, conds, prec, rec)
            return
        j, thr = t.feature[node], t.threshold[node]
        walk(t.children_left[node], conds + [(j, '<=', thr)])
        walk(t.children_right[node], conds + [(j, '>', thr)])

    walk(0, [])
    if best is None:
        return dict(text=None, conditions=[], precision=0.0, recall=0.0)
    _, conds, prec, rec = best
    lo, hi = {}, {}
    for j, op, thr in conds:                       # one interval per column
        if op == '>':
            lo[j] = max(lo.get(j, -np.inf), thr)
        else:
            hi[j] = min(hi.get(j, np.inf), thr)
    parts = []
    for j in sorted(set(lo) | set(hi), key=lambda j: [c[0] for c in conds].index(j)):
        a, b = lo.get(j), hi.get(j)
        if a is not None and b is not None:
            parts.append(f"{_fmt_threshold(a)} < {names[j]} ≤ {_fmt_threshold(b)}")
        elif a is not None:
            parts.append(f"{names[j]} > {_fmt_threshold(a)}")
        else:
            parts.append(f"{names[j]} ≤ {_fmt_threshold(b)}")
    return dict(text=" and ".join(parts), conditions=parts,
                precision=float(prec), recall=float(rec))


def _median_sq_dist(Z, rng, m=4000):
    i = rng.integers(len(Z), size=m)
    j = rng.integers(len(Z), size=m)
    d = np.sum((Z[i] - Z[j]) ** 2, axis=1)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def _n_workers(n_jobs):
    """Threads to use: n_jobs, or by default up to 8 (the sweep's fits are
    small, and numpy already uses several cores inside each one)."""
    if n_jobs is None:
        return max(1, min(8, os.cpu_count() or 1))
    return max(1, int(n_jobs))


def _parallel_map(fn, items, n_jobs):
    """map() over threads. Threads, not processes: the heavy work runs in numpy
    and scipy with the GIL released, and processes would re-import the
    caller's script (a problem in notebooks and unguarded scripts on macOS)."""
    workers = _n_workers(n_jobs)
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _agreement(a, b, min_rows=10):
    """Adjusted Rand index over the rows that both labelings put in a cluster.
    Rows labelled lone (-1) are left out, so declaring everything lone does not
    count as agreement."""
    both = (a >= 0) & (b >= 0)
    if both.sum() < min_rows:
        return 0.0
    return float(adjusted_rand_score(a[both], b[both]))


def select_gamma(Z, grid=None, n_seeds=4, sweep_size=2000, eval_size=3000,
                 min_share=0.01, min_size=10, max_lone=0.10, min_stability=0.8,
                 link_tau=None, random_state=0, n_jobs=None, finer_within=0.1):
    """Choose the bandwidth from the data (thesis Sec. 5.2, automated).

    gamma is swept as c / s2, where s2 is the median squared distance between
    rows, so the grid ``c`` is dimensionless. It runs from c = 2 to whichever
    is finer: c = 160, or a kernel three nearest-neighbour distances wide. At each value GACA is fitted on
    ``n_seeds`` random subsamples and the labels of a common evaluation sample
    are compared:

        k          the number of clusters holding at least ``min_share`` of the
                   evaluation rows (and at least ``min_size`` rows)
        stability  mean pairwise adjusted Rand index between the subsamples

    The chosen value is the middle of the most stable stretch of the longest run of consecutive
    grid values with the same k >= 2 (a plateau), ignoring values where more
    than ``max_lone`` of rows are Lone or the subsamples agree less than
    ``min_stability``. Without that bar, featureless data (one Gaussian) yields
    a "best" split that is just noise: its stability stays around 0.6 to 0.75,
    while real structure in the tests scored 0.89 or more. Among equally long
    plateaus, the finest one whose stability is within ``finer_within`` of the
    most stable is chosen (``finer_within=None``: the most stable one). Other plateaus are returned as
    alternative resolutions, and ``plateaus`` lists every plateau (at most six),
    coarse to fine, for the cluster hierarchy.

    Returns a dict with ``gamma``, ``c``, ``k``, ``stability``, the full
    ``sweep`` table and ``alternatives``. If no plateau with k >= 2 exists, the
    data has no stable multi-cluster structure at any tested resolution;
    ``found`` is then False and the most stable value overall is returned.
    """
    rng = np.random.default_rng(random_state)
    s2 = _median_sq_dist(Z, rng)
    E = Z if len(Z) <= eval_size else Z[rng.choice(len(Z), eval_size, replace=False)]
    sub_n = min(sweep_size, max(int(0.8 * len(Z)), 2))
    if grid is None:
        # The coarse end is set by the typical distance between rows. The fine
        # end must reach the local scale too: in multi-scale data (groups made
        # of subgroups) the typical distance is set by the largest gaps, and a
        # grid that stops at c = 160 never resolves the subgroups. So it runs
        # until the kernel is about three nearest-neighbour distances wide, in
        # steps of about 1.37x.
        nn = cKDTree(E).query(E, k=2)[0][:, 1]
        nn = nn[nn > 0]
        hi = 160.0
        if len(nn):
            hi = max(hi, s2 / (2.0 * (3.0 * float(np.median(nn))) ** 2))
        n_grid = int(np.clip(round(np.log(hi / 2.0) / np.log(1.37)) + 1, 14, 24))
        grid = np.geomspace(2.0, hi, n_grid)
    else:
        grid = np.asarray(grid, float)

    min_rows = max(min_share * len(E), min_size)
    def labels_at(task):
        c, s = task
        r = np.random.default_rng(random_state + 101 * (s + 1))
        sub = r.choice(len(Z), sub_n, replace=False)
        m = GACA(gamma_clustering=c / s2, sample_size=sub_n, random_state=s,
                 link_tau=link_tau).fit(Z[sub])
        lab = m._route(E)
        # Rows in no Sun, in a genesis Lone Sun, or in a speck smaller than
        # min_size all count as lone (-1).
        lone = lab < 0
        lone[~lone] = m.is_lone(lab[~lone])
        lab = np.where(lone, -1, lab)
        small = np.flatnonzero(np.bincount(lab[lab >= 0]) < min_size) if (lab >= 0).any() else []
        lab[np.isin(lab, small)] = -1
        return lab

    # Every (grid value, subsample) fit is independent and seeded, so they run
    # in parallel with results identical to a serial run.
    tasks = [(c, s) for c in grid for s in range(n_seeds)]
    all_labels = _parallel_map(labels_at, tasks, n_jobs)

    sweep = []
    for gi, c in enumerate(grid):
        labels = all_labels[gi * n_seeds:(gi + 1) * n_seeds]
        ks = []
        for lab in labels:
            ok = lab[lab >= 0]
            ks.append(int((np.bincount(ok) >= min_rows).sum()) if len(ok) else 0)
        stab = float(np.mean([_agreement(labels[a], labels[b])
                              for a in range(n_seeds) for b in range(a + 1, n_seeds)]))
        lone = float(np.mean([np.mean(lab < 0) for lab in labels]))
        sweep.append(dict(c=float(c), gamma=float(c / s2), k=int(np.median(ks)),
                          stability=stab, lone_share=lone))

    valid = [r['k'] >= 2 and r['lone_share'] <= max_lone and r['stability'] >= min_stability
             for r in sweep]
    runs, cur = [], []
    for i, r in enumerate(sweep):
        if valid[i] and cur and r['k'] == sweep[cur[-1]]['k']:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = [i] if valid[i] else []
    if cur:
        runs.append(cur)

    def best_of(run):
        # The middle of the near-most-stable stretch: the edge of a plateau is
        # where a different coreset size tips the count over.
        top = max(sweep[i]['stability'] for i in run)
        near = [i for i in run if sweep[i]['stability'] >= top - 0.02]
        return near[len(near) // 2]

    if runs:
        # Longest plateau first; then the clearly more stable one (in steps of
        # 0.02); then, among plateaus that are equally long and equally stable,
        # the finer one. Such ties are common on a discrete grid, and the
        # hierarchy keeps the coarser resolutions anyway.
        def run_key(run):
            stab = np.mean([sweep[i]['stability'] for i in run])
            return len(run), int(round(stab / 0.02)), run[0]
        runs.sort(key=run_key, reverse=True)
        if finer_within is not None:
            # Among the longest plateaus, take the finest whose stability is
            # within finer_within of the best: a stability estimated from four
            # subsamples does not tell 0.99 from 0.94, and preferring the more
            # stable plateau in such ties kept choosing views that were too
            # coarse.
            mean_stab = lambda run: np.mean([sweep[i]['stability'] for i in run])
            longest = [run for run in runs if len(run) == len(runs[0])]
            top = max(mean_stab(run) for run in longest)
            first = max((run for run in longest if mean_stab(run) >= top - finer_within),
                        key=lambda run: run[0])
            runs = [first] + [run for run in runs if run is not first]
        pick = best_of(runs[0])
        alts = [best_of(run) for run in runs[1:4]]
        plateaus = sorted(best_of(run) for run in runs[:6])   # by gamma, coarse to fine
        found = True
    else:
        # No stable multi-cluster structure: report the most stable resolution
        # that does not declare most rows lone (typically one cluster).
        ok = [i for i, r in enumerate(sweep) if r['lone_share'] <= max_lone] or [0]
        pick = max(ok, key=lambda i: (round(sweep[i]['stability'], 3), -i))
        alts, plateaus, found = [], [pick], False

    for i, r in enumerate(sweep):
        r['chosen'] = i == pick
    out = dict(sweep[pick])
    out.update(found=found, median_sq_dist=s2, sweep=sweep,
               alternatives=[dict(sweep[i]) for i in alts],
               plateaus=[dict(sweep[i]) for i in plateaus])
    return out


class AutoGACA:
    """GACA for an arbitrary table: preprocessing, bandwidth and output handled.

    Parameters
    ----------
    columns, exclude, scale, max_dims
        Passed to :class:`Preprocessor`.
    gamma : 'auto' or float
        'auto' runs :func:`select_gamma`. A float is used as is (in the units
        of the preprocessed space).
    bandwidth : float, optional
        The kernel width h in the units of the clustering space, instead of
        gamma (gamma = 1 / (2 h^2)). With ``scale='none'`` this is a physical
        length, e.g. 2.0 for 2 km when coordinates are in km. Overrides gamma.
    resolution : float, optional
        The smallest gap between groups that should stay separate, in the units
        of the clustering space (physical units with ``scale='none'``). gamma is
        derived from the resolution law of docs/theory.md (h is about
        resolution / 3.3 for the default 20 iterations). It assumes groups no
        wider than about a third of the resolution. Overrides gamma; cannot be
        combined with bandwidth.
    link_tau : float, optional
        Saddle linking for curved or elongated clusters (e.g. 0.6). Off by
        default.
    sample_size : int
        Coreset size for the final fit.
    kappa : float
        Lone-Sun threshold relative to the median pull (default 1e-3, i.e. an
        anomaly score of 3).
    min_cluster_size : int
        Clusters with fewer rows are reported as anomaly groups (rare, isolated
        groups) rather than clusters.
    hierarchy : bool
        With gamma='auto', also fit every other stable resolution the sweep
        found and nest them into a tree around the chosen one (default True).
    n_jobs : int, optional
        Threads for the bandwidth sweep and the hierarchy fits (default: up to
        8). Results do not depend on it.
    rare_share : float
        In the anomaly score, a cluster holding less than this share of the
        rows pulls with weight share / rare_share instead of 1. Rows in a
        small group far from the bulk of the data then score as anomalous,
        while the flags (which rows are anomalies) do not change. 0 turns it
        off (default 0.05).
    link_view : bool
        Also cluster with saddle linking (link_tau=0.6, its own automatic
        bandwidth) as an alternative view: better for curved or elongated
        groups, worse when groups overlap. Stored in ``linked_labels_`` and shown
        in the report when it disagrees with the main view (default True; only
        with gamma='auto' and no link_tau).
    random_state : int

    Attributes after ``fit``
    ------------------------
    labels_ : cluster per row, 0 = largest, -1 = anomaly
    anomaly_ : bool per row
    anomaly_group_ : Lone Sun per anomalous row (copies share a group), else -1
    anomaly_score_ : -log10 of the pull on the row relative to the median pull;
        rows above 3 (with the default kappa) are anomalies
    clusters_ : list of dicts describing each cluster
    gamma_selection_ : output of :func:`select_gamma` (if gamma='auto')
    levels_ : the cluster hierarchy, coarse to fine; one dict per level with
        ``gamma``, ``k``, ``labels`` (cluster per row, -1 = anomaly), ``names``
        (path-style name per cluster, e.g. '0.2'), ``parent`` (index of each
        cluster's parent in the previous level) and ``chosen`` (the level of
        ``labels_``). A single level when no hierarchy was built.
    linked_labels_ : cluster per row in the linked view (0 = largest, -1 =
        anomaly), or None
    link_agreement_ : adjusted Rand index between the main and linked views
    result_ : DataFrame with the per-row outputs, including one column per
        hierarchy level (needs pandas)
    """

    def __init__(self, columns=None, exclude=None, scale='robust', max_dims=10,
                 gamma='auto', link_tau=None, sample_size=5000, kappa=1e-3,
                 min_cluster_size=10, bandwidth=None, resolution=None, hierarchy=True,
                 random_state=0, n_jobs=None, link_view=True, separate_modes=True,
                 rare_share=0.05, verbose=False):
        self.columns = columns
        self.exclude = exclude
        self.scale = scale
        self.max_dims = max_dims
        self.gamma = gamma
        self.link_tau = link_tau
        self.sample_size = sample_size
        self.kappa = kappa
        self.min_cluster_size = min_cluster_size
        self.bandwidth = bandwidth
        self.resolution = resolution
        self.hierarchy = hierarchy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.link_view = link_view
        self.separate_modes = separate_modes
        self.rare_share = rare_share
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def fit(self, data):
        df, names = _as_frame(data)
        self.n_rows_ = len(df)
        self.preprocessor_ = Preprocessor(self.columns, self.exclude, self.scale,
                                          self.max_dims, separate_modes=self.separate_modes,
                                          random_state=self.random_state).fit(df)
        Z = self.preprocessor_.transform(df)
        self._log(f"{len(self.preprocessor_.used_)} columns used, GACA space "
                  f"{Z.shape[1]}-D")

        if self.bandwidth is not None and self.resolution is not None:
            raise ValueError("give either bandwidth or resolution, not both")
        if self.bandwidth is not None:
            self.gamma_selection_ = None
            gamma = 1.0 / (2.0 * float(self.bandwidth) ** 2)
        elif self.resolution is not None:
            self.gamma_selection_ = None
            gamma = resolution_to_gamma(self.resolution)
        elif self.gamma == 'auto':
            self._log("Selecting gamma...")
            self.gamma_selection_ = select_gamma(Z, link_tau=self.link_tau,
                                                 min_size=self.min_cluster_size,
                                                 random_state=self.random_state,
                                                 n_jobs=self.n_jobs)
            gamma = self.gamma_selection_['gamma']
            self._log(f"  gamma = {gamma:.4g} (k = {self.gamma_selection_['k']}, "
                      f"stability {self.gamma_selection_['stability']:.2f})")
        else:
            self.gamma_selection_ = None
            gamma = float(self.gamma)
        self.gamma_ = gamma

        self.model_ = GACA(gamma_clustering=gamma, sample_size=self.sample_size,
                           random_state=self.random_state, kappa=self.kappa,
                           link_tau=self.link_tau).fit(Z)
        raw = self.model_.assign(Z)

        # Regular clusters, largest first; Lone Suns become anomaly groups.
        n_gen = self.model_.n_genesis_suns_
        lone_gen = self.model_.is_lone(np.arange(n_gen))
        counts = np.bincount(raw, minlength=n_gen)[:n_gen]
        regular = [k for k in np.argsort(-counts, kind='stable')
                   if not lone_gen[k] and counts[k] >= self.min_cluster_size]
        self._cluster_of = {int(k): i for i, k in enumerate(regular)}
        self._group_of = {}
        self.n_clusters_ = len(regular)

        regular_members = np.isin(self.model_.core_labels_, regular)
        self._regular_tree = (cKDTree(self.model_.core_[regular_members])
                              if regular_members.any() else None)
        # Pull weight of each regular member in the anomaly score: 1, or
        # share / rare_share for a cluster rarer than rare_share.
        share = {int(k): counts[k] / max(len(raw), 1) for k in regular}
        member_sun = self.model_.core_labels_[regular_members]
        self._regular_w = np.array([min(1.0, share[int(k)] / self.rare_share)
                                    if self.rare_share else 1.0 for k in member_sun])
        self._set_outputs(df, Z, raw)
        # Reference for drift monitoring: cluster shares (anomalies last) and the
        # anomaly rate on the data the model was fitted on.
        counts = np.bincount(self.labels_[self.labels_ >= 0], minlength=self.n_clusters_)
        shares = np.r_[counts, (self.labels_ < 0).sum()] / len(self.labels_)
        self.reference_ = dict(shares=shares, anomaly_rate=float(self.anomaly_.mean()),
                               n_groups=int(self.anomaly_group_.max() + 1))
        self._build_hierarchy(Z)
        self.clusters_ = self._describe(df)
        self._fit_link_view(df)
        return self

    def _fit_link_view(self, df):
        """The alternative, linked clustering (see ``link_view``)."""
        self.linked_ = None
        self.linked_labels_ = None
        self.link_agreement_ = None
        if not (self.link_view and self.link_tau is None and self.bandwidth is None
                and self.resolution is None and self.gamma == 'auto'):
            return
        self._log("Linked view...")
        alt = AutoGACA(columns=self.columns, exclude=self.exclude, scale=self.scale,
                       max_dims=self.max_dims, link_tau=0.6, sample_size=self.sample_size,
                       kappa=self.kappa, min_cluster_size=self.min_cluster_size,
                       hierarchy=False, link_view=False, separate_modes=self.separate_modes,
                       random_state=self.random_state,
                       n_jobs=self.n_jobs).fit(df)
        self.linked_ = alt
        self.linked_labels_ = alt.labels_
        both = (self.labels_ >= 0) & (alt.labels_ >= 0)
        self.link_agreement_ = (float(adjusted_rand_score(self.labels_[both], alt.labels_[both]))
                                if both.sum() >= 10 else 0.0)

    # ------------------------------------------------------------- hierarchy

    def _level_model(self, gamma):
        """A GACA fit at another resolution, on the same coreset."""
        return GACA(gamma_clustering=gamma, sample_size=self.sample_size,
                    random_state=self.random_state, kappa=self.kappa,
                    link_tau=self.link_tau).fit(self.Z_)

    def _regular_suns(self, model):
        """Genesis Suns big enough to be clusters, judged by expected row count."""
        scale = self.n_rows_ / max(model.sun_masses_.sum(), 1)
        lone = model.is_lone(np.arange(model.n_genesis_suns_))
        return np.flatnonzero(~lone & (model.sun_masses_ * scale >= self.min_cluster_size))

    def _build_hierarchy(self, Z):
        """Nest every stable resolution of the sweep around the chosen one.

        Every level is fitted on the same coreset, so each coreset particle has
        a cluster at every level, and the tree is read from the particles:

        * coarser level: each cluster of the level below goes to the coarse
          cluster that holds most of its particles (merging only);
        * finer level: each fine Sun goes to the cluster above that holds most
          of its particles, and each row is assigned to the strongest-pulling
          Sun among the children of its own cluster (splitting only).

        So the levels are strictly nested. Anomalies stay anomalies at every
        level."""
        sel = self.gamma_selection_
        plateaus = []
        if sel is not None and self.hierarchy and sel['found']:
            plateaus = [p for p in sel['plateaus'] if not np.isclose(p['gamma'], self.gamma_)]
        coarse = sorted((p for p in plateaus if p['gamma'] < self.gamma_),
                        key=lambda p: -p['gamma'])                    # nearest first
        fine = sorted((p for p in plateaus if p['gamma'] > self.gamma_),
                      key=lambda p: p['gamma'])
        core_chosen = np.array([self._cluster_of.get(int(r), -1)
                                for r in self.model_.core_labels_])
        # The level fits are independent: run them in parallel up front.
        level_gammas = [p['gamma'] for p in coarse + fine]
        models = dict(zip(level_gammas, _parallel_map(self._level_model, level_gammas,
                                                      self.n_jobs)))

        # Coarser levels: merge the clusters of the level below.
        self._coarse_chain = []
        below_core, below_k = core_chosen, self.n_clusters_
        for p in coarse:
            self._log(f"  hierarchy: coarser level at gamma {p['gamma']:.3g}")
            m = models[p['gamma']]
            reg = {int(sun): i for i, sun in enumerate(self._regular_suns(m))}
            member = np.array([reg.get(int(r), -1) for r in m.core_labels_])
            raw_parent = np.empty(below_k, dtype=int)
            for c in range(below_k):
                votes = member[(below_core == c) & (member >= 0)]
                raw_parent[c] = np.bincount(votes).argmax() if len(votes) else -(c + 1)
            _, parent = np.unique(raw_parent, return_inverse=True)
            if parent.max() + 1 >= below_k:                         # merges nothing
                continue
            self._coarse_chain.append(dict(gamma=p['gamma'], parent=parent))
            below_core = np.where(below_core >= 0, parent[np.maximum(below_core, 0)], -1)
            below_k = parent.max() + 1

        # Finer levels: split the clusters of the level above.
        self._fine_levels = []
        above_core, above_labels, above_k = core_chosen, self.labels_, self.n_clusters_
        fine_labels = []
        for p in fine:
            self._log(f"  hierarchy: finer level at gamma {p['gamma']:.3g}")
            m = models[p['gamma']]
            parent_of_sun = np.full(m.n_genesis_suns_, -1)
            for sun in self._regular_suns(m):
                votes = above_core[(m.core_labels_ == sun) & (above_core >= 0)]
                if len(votes):
                    parent_of_sun[sun] = np.bincount(votes, minlength=above_k).argmax()
            sun_of_row = self._children_assign(m, Z, above_labels, parent_of_sun, above_k)
            keys, counts = np.unique(sun_of_row[sun_of_row != -1], return_counts=True)
            relabel = {int(k): i for i, k in enumerate(keys[np.argsort(-counts, kind='stable')])}
            if len(relabel) <= above_k:                              # splits nothing
                continue
            labels = self._relabel(sun_of_row, relabel)
            core_new = self._relabel(
                self._children_assign(m, m.core_, above_core, parent_of_sun, above_k), relabel)
            self._fine_levels.append(dict(gamma=p['gamma'], k=len(relabel), model=m,
                                          parent_of_sun=parent_of_sun, relabel=relabel))
            fine_labels.append(labels)
            above_core, above_labels, above_k = core_new, labels, len(relabel)

        levels = []
        for cl, lab in zip(reversed(self._coarse_chain), self._coarse_labels(self.labels_)):
            levels.append(dict(gamma=cl['gamma'], k=int(cl['parent'].max() + 1), labels=lab))
        levels.append(dict(gamma=self.gamma_, k=self.n_clusters_, labels=self.labels_,
                           chosen=True))
        for fl, lab in zip(self._fine_levels, fine_labels):
            levels.append(dict(gamma=fl['gamma'], k=fl['k'], labels=lab))
        self.levels_ = self._finish_levels(levels)

    def _coarse_labels(self, labels):
        """Labels at each coarser level, coarsest first."""
        out, lab = [], labels
        for cl in self._coarse_chain:
            lab = np.where(lab >= 0, cl['parent'][np.maximum(lab, 0)], -1)
            out.append(lab)
        return out[::-1]

    @staticmethod
    def _children_assign(m, Z, above_labels, parent_of_sun, above_k):
        """Sun of model m for each row, chosen only among the children of the
        row's cluster in the level above. A cluster with no child Sun keeps a
        single virtual child, coded -(cluster + 2); anomalies stay -1."""
        children = [np.flatnonzero(parent_of_sun == c) for c in range(above_k)]
        out = np.full(len(Z), -1)
        for sl, acc, _ in m._pull_chunks(Z):
            par = above_labels[sl]
            allowed = np.zeros(acc.shape, dtype=bool)
            for c in range(above_k):
                rows = np.flatnonzero(par == c)
                if len(children[c]) and len(rows):
                    allowed[np.ix_(rows, children[c])] = True
            masked = np.where(allowed, acc, -1.0)
            best = masked.argmax(axis=1)
            pulled = masked[np.arange(len(best)), best] > 0
            for i in np.flatnonzero((par >= 0) & ~pulled):
                ch = children[par[i]]
                # no child pulls this row: the heaviest child, or the virtual one
                best[i] = ch[np.argmax(m.sun_masses_[ch])] if len(ch) else -(par[i] + 2)
            out[sl] = np.where(par >= 0, best, -1)
        return out

    @staticmethod
    def _relabel(values, relabel):
        return np.array([relabel.get(int(v), -1) if v != -1 else -1 for v in values], dtype=int)

    def _finish_levels(self, levels):
        """Parents between consecutive levels, and path-style names ('0.2.1')
        ordered by size within each parent."""
        for i, lv in enumerate(levels):
            lab = lv['labels']
            sizes = np.bincount(lab[lab >= 0], minlength=lv['k'])
            lv.setdefault('chosen', False)
            lv['names'] = [''] * lv['k']
            if i == 0:
                lv['parent'] = None
                for rank, c in enumerate(np.argsort(-sizes, kind='stable')):
                    lv['names'][c] = str(rank)
                continue
            up = levels[i - 1]
            parent = np.zeros(lv['k'], dtype=int)
            for c in range(lv['k']):
                above = up['labels'][(lab == c) & (up['labels'] >= 0)]
                parent[c] = np.bincount(above).argmax() if len(above) else 0
            lv['parent'] = parent
            for pc in range(up['k']):
                kids = [c for c in np.argsort(-sizes, kind='stable') if parent[c] == pc]
                for rank, c in enumerate(kids):
                    lv['names'][c] = f"{up['names'][pc]}.{rank}"
        return levels

    def _hier_labels(self, Z, labels):
        """Labels at every level for new rows, given their chosen-level labels."""
        out = self._coarse_labels(labels) + [labels]
        above, above_k = labels, self.n_clusters_
        for fl in self._fine_levels:
            sun = self._children_assign(fl['model'], Z, above, fl['parent_of_sun'], above_k)
            above, above_k = self._relabel(sun, fl['relabel']), fl['k']
            out.append(above)
        return out


    def _map(self, raw):
        labels = np.full(len(raw), -1)
        groups = np.full(len(raw), -1)
        for i, r in enumerate(raw):
            r = int(r)
            if r in self._cluster_of:
                labels[i] = self._cluster_of[r]
            else:
                if r not in self._group_of:
                    self._group_of[r] = len(self._group_of)
                groups[i] = self._group_of[r]
        return labels, groups

    def _score(self, Z):
        """-log10 of the pull of the regular clusters on each row, relative to
        the median pull inside the coreset.

        Only coreset members of regular clusters pull: members of Lone Suns and
        anomaly groups do not, so the copies of a sampled anomaly cannot make
        each other look ordinary. Members of clusters rarer than ``rare_share``
        pull with a weight proportional to their cluster's share, so a small
        isolated group also scores as unusual. A row that is itself a coreset member does not
        count its own pull. The sum is taken in log space, so rows far from
        everything keep a finite score that grows with distance."""
        m = self.model_
        if self._regular_tree is None:
            return np.full(len(Z), np.inf)
        K = min(m.n_neighbors, self._regular_tree.n)
        dist, idx = self._regular_tree.query(Z, k=K, workers=-1)
        dist = dist.reshape(len(Z), -1)
        idx = idx.reshape(len(Z), -1)
        logw = -m.gamma_clustering * dist ** 2 + np.log(self._regular_w[idx])
        logw[:, 0] = np.where(dist[:, 0] == 0, -np.inf, logw[:, 0])
        top = logw.max(axis=1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        log_pull = (top[:, 0] + np.log(np.exp(logw - top).sum(axis=1))
                    if K else np.full(len(Z), -np.inf))
        log_median = np.log(m.kappa_ / m.kappa)
        return -(log_pull - log_median) / np.log(10)

    def _set_outputs(self, df, Z, raw):
        self.labels_, self.anomaly_group_ = self._map(raw)
        self.anomaly_ = self.labels_ < 0
        self.anomaly_score_ = self._score(Z)
        self.Z_ = Z

    def assign(self, data):
        """Cluster labels, anomaly flags, groups and scores for new rows.

        Works batch by batch; anomalies seen before keep their group. Returns a
        DataFrame (or a dict of arrays without pandas)."""
        df, _ = _as_frame(data)
        Z = self.preprocessor_.transform(df)
        labels, groups = self._map(self.model_.assign(Z))
        linked = None
        if getattr(self, 'linked_', None) is not None:
            linked = np.asarray(self.linked_.assign(df)['gaca_cluster']) \
                if self._has_pandas() else self.linked_.assign(df)['gaca_cluster']
        return self._frame(labels, groups, self._score(Z), self._hier_labels(Z, labels), linked)

    def compare(self, truth, baselines=False, max_baseline_rows=200_000):
        """Score the clustering against known labels for the fitted rows.

        ``truth`` holds one label per fitted row (any type). The labels are used
        only here, after clustering. Returns a dict with:

        * ``methods``: per method, the adjusted Rand index (ARI), normalised
          mutual information (NMI), the number of clusters, and the share of
          rows put in a cluster. Methods are GACA's ``gaca_cluster``, the linked
          view, and the best-matching hierarchy level, plus, with ``baselines``,
          K-Means given the true number of classes and HDBSCAN (minimum cluster
          size 1% of rows), both run on the same preprocessed data.
        * ``clusters``: per GACA cluster, its size and the known labels it
          contains, with shares, and its purity (share of the commonest label).

        Rows GACA leaves unclustered (anomalies) count as one extra label in
        the ARI, as in the benchmark. The result is also stored in
        ``comparison_`` for the report."""
        from sklearn.metrics import normalized_mutual_info_score
        truth = np.asarray(truth)
        if len(truth) != len(self.labels_):
            raise ValueError("truth needs one label per fitted row")
        names, y = np.unique(truth.astype(str), return_inverse=True)

        def score(name, labels, note=''):
            covered = labels >= 0
            return dict(method=name, note=note, ari=float(adjusted_rand_score(y, labels)),
                        nmi=float(normalized_mutual_info_score(y, labels)),
                        clusters=int(len(set(labels[covered].tolist()))),
                        covered=float(covered.mean()))

        methods = [score('GACA', self.labels_, 'gaca_cluster')]
        if getattr(self, 'linked_labels_', None) is not None:
            methods.append(score('GACA, linked view', self.linked_labels_, 'gaca_linked_cluster'))
        if len(self.levels_) > 1:
            best = max(range(len(self.levels_)),
                       key=lambda i: adjusted_rand_score(y, self.levels_[i]['labels']))
            methods.append(score('GACA, best hierarchy level', self.levels_[best]['labels'],
                                 f"gaca_level_{best + 1}"))
        if baselines:
            from sklearn.cluster import HDBSCAN, KMeans
            Z = self.Z_
            methods.append(score('K-Means (given the true k)',
                                 KMeans(len(names), n_init=10,
                                        random_state=self.random_state).fit_predict(Z),
                                 f"k = {len(names)}"))
            if len(Z) <= max_baseline_rows:
                methods.append(score('HDBSCAN',
                                     HDBSCAN(min_cluster_size=max(5, len(Z) // 100)).fit_predict(Z),
                                     'minimum cluster size 1% of rows'))
        clusters = []
        for k in range(self.n_clusters_):
            m = self.labels_ == k
            if not m.any():
                continue
            counts = np.bincount(y[m], minlength=len(names))
            order = np.argsort(-counts)
            clusters.append(dict(cluster=k, size=int(m.sum()),
                                 purity=float(counts[order[0]] / m.sum()),
                                 labels=[(str(names[j]), float(counts[j] / m.sum()))
                                         for j in order[:3] if counts[j]]))
        self.comparison_ = dict(n_labels=int(len(names)), methods=methods, clusters=clusters)
        return self.comparison_

    def drift(self, result, max_shift=0.1, max_anomaly_ratio=3.0, alpha=1e-3):
        """Compare a batch of assigned rows with the data the model was fitted on.

        ``result`` is the output of :meth:`assign` for the batch. Returns a dict:

        * ``cluster_shift``: total variation distance between the batch's
          cluster shares and the fitted shares (anomalies count as a category;
          0 = same mix, 1 = disjoint). ``cluster_p`` is the chi-square test of
          the batch counts against the fitted shares.
        * ``anomaly_rate`` and ``anomaly_ratio`` (batch rate / fitted rate), with
          ``anomaly_p``, the one-sided binomial test for an increase.
        * ``new_groups``: anomaly groups first seen in this batch.
        * ``drift``: True when the cluster mix moved by more than ``max_shift``,
          or the anomaly rate rose more than ``max_anomaly_ratio`` times (and by
          at least one percentage point), with the change significant at
          ``alpha``. ``reasons`` says which.

        Both a size and a significance condition are required: small batches
        fluctuate, and large batches make any tiny difference significant."""
        from scipy.stats import binomtest, chisquare
        get = (lambda k: np.asarray(result[k]))
        labels = get('gaca_cluster')
        groups = get('gaca_anomaly_group')
        n = len(labels)
        ref = self.reference_
        k = len(ref['shares']) - 1
        counts = np.r_[np.bincount(labels[labels >= 0], minlength=k)[:k], (labels < 0).sum()]
        shares = counts / max(n, 1)
        shift = 0.5 * float(np.abs(shares - ref['shares']).sum())
        expected = np.maximum(ref['shares'], 0.5 / max(n, 1))
        expected = expected / expected.sum() * n
        cluster_p = float(chisquare(counts, expected).pvalue) if n else 1.0
        rate = float((labels < 0).mean()) if n else 0.0
        base = max(ref['anomaly_rate'], 1.0 / max(len(self.labels_), 1))
        anomaly_p = float(binomtest(int((labels < 0).sum()), n, base,
                                    alternative='greater').pvalue) if n else 1.0
        known = getattr(self, '_groups_seen', ref['n_groups'])
        new_groups = int(len(set(groups[groups >= known].tolist())))
        self._groups_seen = max(known, int(groups.max()) + 1 if (groups >= 0).any() else known)
        reasons = []
        if shift > max_shift and cluster_p < alpha:
            reasons.append(f"cluster mix moved by {shift:.2f}")
        if (rate > max_anomaly_ratio * base and rate - base > 0.01 and anomaly_p < alpha):
            reasons.append(f"anomaly rate {100 * rate:.1f}% vs {100 * base:.1f}% when fitted")
        return dict(rows=n, cluster_shift=shift, cluster_p=cluster_p, anomaly_rate=rate,
                    anomaly_ratio=rate / base, anomaly_p=anomaly_p, new_groups=new_groups,
                    drift=bool(reasons), reasons="; ".join(reasons))

    @staticmethod
    def _has_pandas():
        try:
            import pandas  # noqa: F401
            return True
        except ImportError:
            return False

    def _frame(self, labels, groups, score, level_labels=None, linked=None):
        cols = dict(gaca_cluster=labels, gaca_anomaly=labels < 0,
                    gaca_anomaly_group=groups, gaca_anomaly_score=np.round(score, 3))
        if linked is not None:
            cols['gaca_linked_cluster'] = linked
        levels = getattr(self, 'levels_', [])
        if level_labels is not None and len(levels) > 1:
            for i, (lv, lab) in enumerate(zip(levels, level_labels), start=1):
                names = np.array(lv['names'] + ['anomaly'], dtype=object)
                cols[f'gaca_level_{i}'] = names[np.where(lab >= 0, lab, len(lv['names']))]
        try:
            import pandas as pd
            return pd.DataFrame(cols)
        except ImportError:
            return cols

    @property
    def result_(self):
        return self._frame(self.labels_, self.anomaly_group_, self.anomaly_score_,
                           [lv['labels'] for lv in self.levels_], self.linked_labels_)

    def _describe(self, df):
        """Per cluster: size, medians of the original columns, and the columns
        that set it apart (difference of medians in robust-scaled units)."""
        S = self.preprocessor_.scaled(df)
        used = self.preprocessor_.used_
        overall = np.median(S, axis=0)
        out = []
        for k in range(self.n_clusters_):
            mask = self.labels_ == k
            if not mask.any():
                continue
            diff = np.median(S[mask], axis=0) - overall
            order = np.argsort(-np.abs(diff))[:4]
            medians = {}
            for c in used:
                v, _ = _to_numeric(np.asarray(df[c]))
                medians[c] = float(np.nanmedian(v[mask])) if np.isfinite(v[mask]).any() else np.nan
            out.append(dict(cluster=k, size=int(mask.sum()), share=float(mask.mean()),
                            medians=medians,
                            distinctive=[(used[j], float(diff[j])) for j in order
                                         if abs(diff[j]) >= 0.25]))
        self.tree_ = self._describe_tree(S, used, overall)
        # A short rule per cluster, in the original units of the columns.
        V = self.preprocessor_.values(df)
        for c in out:
            c['rule'] = cluster_rule(V, used, self.labels_ == c['cluster'],
                                     random_state=self.random_state)
        return out

    def _describe_tree(self, S, used, overall):
        """One entry per node of the hierarchy, top-down (depth-first): name,
        level, size, and the columns that set it apart from its parent (or, at
        the top level, from all rows)."""
        if len(self.levels_) < 2:
            return []
        nodes = []

        def visit(level, c, parent_median):
            lv = self.levels_[level]
            mask = lv['labels'] == c
            med = np.median(S[mask], axis=0)
            diff = med - parent_median
            order = np.argsort(-np.abs(diff))[:3]
            nodes.append(dict(name=lv['names'][c], level=level + 1, size=int(mask.sum()),
                              share=float(mask.mean()), chosen=lv['chosen'],
                              cluster=int(c) if lv['chosen'] else None,
                              distinctive=[(used[j], float(diff[j])) for j in order
                                           if abs(diff[j]) >= 0.25]))
            if level + 1 < len(self.levels_):
                nxt = self.levels_[level + 1]
                kids = [k for k in range(nxt['k']) if nxt['parent'][k] == c]
                kids.sort(key=lambda k: _path_key(nxt['names'][k]))
                if len(kids) > 1:                 # an only child repeats its parent
                    for k in kids:
                        visit(level + 1, k, med)
                elif kids:
                    deeper(level + 1, kids[0], med)

        def deeper(level, c, parent_median):
            """Skip levels where a node does not split; continue below them."""
            nxt_level = level + 1
            if nxt_level >= len(self.levels_):
                return
            nxt = self.levels_[nxt_level]
            kids = [k for k in range(nxt['k']) if nxt['parent'][k] == c]
            kids.sort(key=lambda k: _path_key(nxt['names'][k]))
            if len(kids) > 1:
                for k in kids:
                    visit(nxt_level, k, parent_median)
            elif kids:
                deeper(nxt_level, kids[0], parent_median)

        top = self.levels_[0]
        for c in sorted(range(top['k']), key=lambda k: _path_key(top['names'][k])):
            visit(0, c, overall)
        return nodes

    def summary(self):
        """A short plain-text summary of the fit."""
        p = self.preprocessor_
        lines = [f"AutoGACA on {self.n_rows_:,} rows",
                 f"  columns used: {len(p.used_)} ({', '.join(p.used_[:8])}"
                 f"{', ...' if len(p.used_) > 8 else ''})",
                 f"  GACA space: {p.n_dims_}-D" + (" after PCA" if p.pca_ is not None else "")]
        sel = self.gamma_selection_
        if sel is not None:
            lines.append(f"  gamma: {self.gamma_:.4g} (auto, stability {sel['stability']:.2f}"
                         f"{'' if sel['found'] else '; no stable multi-cluster plateau found'})")
        elif self.resolution is not None:
            lines.append(f"  gamma: {self.gamma_:.4g} (from resolution {self.resolution:g}: "
                         f"kernel width {1 / np.sqrt(2 * self.gamma_):.3g})")
        else:
            lines.append(f"  gamma: {self.gamma_:.4g} (given)")
        lines.append(f"  clusters: {self.n_clusters_}, anomalies: {int(self.anomaly_.sum())} rows "
                     f"in {len(set(self.anomaly_group_[self.anomaly_]))} groups")
        for c in self.clusters_[:10]:
            d = ', '.join(f"{n} {'+' if v > 0 else '-'}" for n, v in c['distinctive'][:3])
            lines.append(f"    cluster {c['cluster']}: {c['size']:,} rows ({c['share']:.1%})"
                         + (f"  [{d}]" if d else ''))
            rule = c.get('rule')
            if rule and rule['text'] and rule['precision'] >= 0.5 and rule['recall'] >= 0.5:
                lines.append(f"        ≈ {rule['text']}  (precision {rule['precision']:.0%}, "
                             f"recall {rule['recall']:.0%})")
        if self.n_clusters_ > 10:
            lines.append(f"    ... {self.n_clusters_ - 10} more")
        if self.linked_labels_ is not None:
            k_link = self.linked_.n_clusters_
            note = ("agrees with the main view" if self.link_agreement_ >= 0.8 else
                    "differs from the main view: see the report")
            lines.append(f"  linked view: {k_link} clusters, {note} "
                         f"(agreement {self.link_agreement_:.2f})")
        if len(self.levels_) > 1:
            lines.append("  hierarchy: " + " -> ".join(
                f"{lv['k']}{'*' if lv['chosen'] else ''}" for lv in self.levels_)
                + " clusters per level (* = the level in gaca_cluster)")
        return "\n".join(lines)

    def report(self, path, data=None, title=None, truth_name=None, drift=None):
        """Write a self-contained HTML report. Pass the original ``data`` to
        include the original column values of the top anomalies. After
        :meth:`compare`, the report includes the comparison (``truth_name``
        labels it); ``drift`` is a list of :meth:`drift` results, one per
        chunk, to chart."""
        from .report import write_report
        write_report(self, path, data=data, title=title, truth_name=truth_name, drift=drift)
        return path
