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
import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

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
        data whose distances are already meaningful (e.g. positions in Mpc);
        gamma is then in the data's own units.
    max_dims : int
        If more columns remain than this, project with PCA to between 5 (or
        ``max_dims`` if smaller) and ``max_dims`` components, choosing the number
        by the noise floor of the singular values.
    max_missing : float
        Columns with a larger share of missing values are dropped.
    """

    def __init__(self, columns=None, exclude=None, scale='robust', max_dims=10,
                 max_missing=0.5, random_state=0):
        self.columns = columns
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


def _median_sq_dist(Z, rng, m=4000):
    i = rng.integers(len(Z), size=m)
    j = rng.integers(len(Z), size=m)
    d = np.sum((Z[i] - Z[j]) ** 2, axis=1)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


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
                 link_tau=None,
                 random_state=0):
    """Choose the bandwidth from the data (thesis Sec. 5.2, automated).

    gamma is swept as c / s2, where s2 is the median squared distance between
    rows, so the grid ``c`` is dimensionless. At each value GACA is fitted on
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
    while real structure in the tests scored 0.89 or more. Other plateaus are returned as
    alternative resolutions.

    Returns a dict with ``gamma``, ``c``, ``k``, ``stability``, the full
    ``sweep`` table and ``alternatives``. If no plateau with k >= 2 exists, the
    data has no stable multi-cluster structure at any tested resolution;
    ``found`` is then False and the most stable value overall is returned.
    """
    grid = np.geomspace(2, 160, 14) if grid is None else np.asarray(grid, float)
    rng = np.random.default_rng(random_state)
    s2 = _median_sq_dist(Z, rng)
    E = Z if len(Z) <= eval_size else Z[rng.choice(len(Z), eval_size, replace=False)]
    sub_n = min(sweep_size, max(int(0.8 * len(Z)), 2))

    min_rows = max(min_share * len(E), min_size)
    sweep = []
    for c in grid:
        labels = []
        for s in range(n_seeds):
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
            labels.append(lab)
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
        runs.sort(key=lambda run: (len(run), np.mean([sweep[i]['stability'] for i in run])),
                  reverse=True)
        pick = best_of(runs[0])
        alts = [best_of(run) for run in runs[1:4]]
        found = True
    else:
        # No stable multi-cluster structure: report the most stable resolution
        # that does not declare most rows lone (typically one cluster).
        ok = [i for i, r in enumerate(sweep) if r['lone_share'] <= max_lone] or [0]
        pick = max(ok, key=lambda i: (round(sweep[i]['stability'], 3), -i))
        alts, found = [], False

    for i, r in enumerate(sweep):
        r['chosen'] = i == pick
    out = dict(sweep[pick])
    out.update(found=found, median_sq_dist=s2, sweep=sweep,
               alternatives=[dict(sweep[i]) for i in alts])
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
        length, e.g. 1.0 for 1 Mpc when positions are in Mpc. Overrides gamma.
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
    result_ : DataFrame with the four per-row outputs (needs pandas)
    """

    def __init__(self, columns=None, exclude=None, scale='robust', max_dims=10,
                 gamma='auto', link_tau=None, sample_size=5000, kappa=1e-3,
                 min_cluster_size=10, bandwidth=None,
                 random_state=0, verbose=False):
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
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def fit(self, data):
        df, names = _as_frame(data)
        self.n_rows_ = len(df)
        self.preprocessor_ = Preprocessor(self.columns, self.exclude, self.scale,
                                          self.max_dims, random_state=self.random_state).fit(df)
        Z = self.preprocessor_.transform(df)
        self._log(f"{len(self.preprocessor_.used_)} columns used, GACA space "
                  f"{Z.shape[1]}-D")

        if self.bandwidth is not None:
            self.gamma_selection_ = None
            gamma = 1.0 / (2.0 * float(self.bandwidth) ** 2)
        elif self.gamma == 'auto':
            self._log("Selecting gamma...")
            self.gamma_selection_ = select_gamma(Z, link_tau=self.link_tau,
                                                 min_size=self.min_cluster_size,
                                                 random_state=self.random_state)
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

        self._set_outputs(df, Z, raw)
        self.clusters_ = self._describe(df)
        return self

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
        """-log10 of the pull on each row relative to the median coreset pull.

        A row that is itself a coreset member does not count its own unit of
        pull, so coreset outliers score like any other outlier."""
        m = self.model_
        K = min(m.n_neighbors, len(m.core_))
        dist, _ = m.core_tree_.query(Z, k=K, workers=-1)
        dist = dist.reshape(len(Z), -1)
        w = np.exp(-m.gamma_clustering * dist ** 2)
        w[:, 0] = np.where(dist[:, 0] == 0, 0.0, w[:, 0])
        median_pull = m.kappa_ / m.kappa
        return -np.log10(np.maximum(w.sum(1), 1e-300) / median_pull)

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
        return self._frame(labels, groups, self._score(Z))

    def _frame(self, labels, groups, score):
        cols = dict(gaca_cluster=labels, gaca_anomaly=labels < 0,
                    gaca_anomaly_group=groups, gaca_anomaly_score=np.round(score, 3))
        try:
            import pandas as pd
            return pd.DataFrame(cols)
        except ImportError:
            return cols

    @property
    def result_(self):
        return self._frame(self.labels_, self.anomaly_group_, self.anomaly_score_)

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
        return out

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
        else:
            lines.append(f"  gamma: {self.gamma_:.4g} (given)")
        lines.append(f"  clusters: {self.n_clusters_}, anomalies: {int(self.anomaly_.sum())} rows "
                     f"in {len(set(self.anomaly_group_[self.anomaly_]))} groups")
        for c in self.clusters_[:10]:
            d = ', '.join(f"{n} {'+' if v > 0 else '-'}" for n, v in c['distinctive'][:3])
            lines.append(f"    cluster {c['cluster']}: {c['size']:,} rows ({c['share']:.1%})"
                         + (f"  [{d}]" if d else ''))
        if self.n_clusters_ > 10:
            lines.append(f"    ... {self.n_clusters_ - 10} more")
        return "\n".join(lines)

    def report(self, path, data=None, title=None):
        """Write a self-contained HTML report. Pass the original ``data`` to
        include the original column values of the top anomalies."""
        from .report import write_report
        write_report(self, path, data=data, title=title)
        return path
