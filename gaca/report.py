"""Self-contained HTML report for a fitted AutoGACA (no external files or scripts)."""
import datetime
import html

import numpy as np

PALETTE = ['#4269d0', '#efb118', '#ff725c', '#6cc5b0', '#3ca951', '#ff8ab7',
           '#a463f2', '#97bbf5', '#9c6b4e', '#9498a0']
OTHER = '#c4c7cc'

CSS = """
:root { --bg:#ffffff; --fg:#1d2127; --muted:#5f6670; --line:#e3e6ea; --card:#f6f7f9;
        --accent:#4269d0; --anom:#d0342c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171b; --fg:#e7e9ec; --muted:#9aa2ad; --line:#2c3238; --card:#1c2025;
          --accent:#7b9cf0; --anom:#ff6b61; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }
p.sub, .muted { color: var(--muted); }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin:20px 0; }
.card { background:var(--card); border-radius:10px; padding:12px 14px; }
.card .v { font-size:24px; font-weight:600; font-variant-numeric: tabular-nums; }
.card .l { color:var(--muted); font-size:13px; }
table { border-collapse: collapse; width:100%; font-size:14px; font-variant-numeric: tabular-nums; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:500; }
.scroll { overflow-x:auto; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
.up, .down { color: var(--fg); }
svg text { fill: var(--muted); font-size: 11px; }
svg .axis { stroke: var(--line); }
.note { background:var(--card); border-left:3px solid var(--accent); padding:10px 14px; border-radius:6px; }
code { font-size: 13px; }
"""


def _e(x):
    return html.escape(str(x))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return '–'
        a = abs(v)
        if a != 0 and (a >= 1e5 or a < 1e-3):
            return f"{v:.3g}"
        return f"{v:,.3f}".rstrip('0').rstrip('.') if a < 100 else f"{v:,.1f}"
    return _e(v)


def _sweep_svg(sel, width=960, height=260):
    sw = sel['sweep']
    c = np.log10([r['c'] for r in sw])
    k = np.array([max(r['k'], 0) for r in sw], float)
    st = np.array([r['stability'] for r in sw])
    L, R, T, gap = 50, 20, 14, 30
    ph = (height - T - gap - 30) / 2
    x = lambda v: L + (v - c.min()) / max(c.max() - c.min(), 1e-9) * (width - L - R)
    kmax = max(k.max(), 2)
    yk = lambda v: T + ph - np.log1p(v) / np.log1p(kmax) * ph
    ys = lambda v: T + ph + gap + (1 - v) * ph
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="Bandwidth sweep: clusters and stability">']
    chosen = [i for i, r in enumerate(sw) if r['chosen']]
    alts = {round(a['c'], 6) for a in sel.get('alternatives', [])}
    for i in chosen:
        out.append(f'<line x1="{x(c[i]):.1f}" x2="{x(c[i]):.1f}" y1="{T}" y2="{T + 2 * ph + gap}" '
                   f'stroke="var(--accent)" stroke-dasharray="4 3"/>')
    for y0 in (T + ph, T + 2 * ph + gap):
        out.append(f'<line class="axis" x1="{L}" x2="{width - R}" y1="{y0}" y2="{y0}"/>')
    for v in sorted({1, 2, 5, 10, 20, 50, 100, 200, 500} & set(range(1, int(kmax) + 1))):
        out.append(f'<text x="{L - 6}" y="{yk(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for v in (0, 0.5, 1):
        out.append(f'<text x="{L - 6}" y="{ys(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    out.append(f'<text x="{L}" y="{T - 2}">clusters</text>')
    out.append(f'<text x="{L}" y="{T + ph + gap - 4}">stability (agreement between subsamples)</text>')
    pk = " ".join(f"{x(a):.1f},{yk(b):.1f}" for a, b in zip(c, k))
    ps = " ".join(f"{x(a):.1f},{ys(b):.1f}" for a, b in zip(c, st))
    out.append(f'<polyline points="{pk}" fill="none" stroke="var(--fg)" stroke-width="1.5"/>')
    out.append(f'<polyline points="{ps}" fill="none" stroke="var(--fg)" stroke-width="1.5"/>')
    for i, (a, b, s) in enumerate(zip(c, k, st)):
        col = 'var(--accent)' if sw[i]['chosen'] else ('#efb118' if round(sw[i]['c'], 6) in alts else 'var(--fg)')
        r = 5 if col != 'var(--fg)' else 3
        tip = (f"c = {sw[i]['c']:.3g}, gamma = {sw[i]['gamma']:.3g}: {sw[i]['k']} clusters, "
               f"stability {s:.2f}, lone {100 * sw[i]['lone_share']:.1f}%")
        out.append(f'<circle cx="{x(a):.1f}" cy="{yk(b):.1f}" r="{r}" fill="{col}"><title>{tip}</title></circle>')
        out.append(f'<circle cx="{x(a):.1f}" cy="{ys(s):.1f}" r="{r}" fill="{col}"><title>{tip}</title></circle>')
        out.append(f'<text x="{x(a):.1f}" y="{height - 8}" text-anchor="middle">{sw[i]["c"]:.3g}</text>')
    out.append('</svg>')
    return "".join(out)


def _scatter_svg(auto, size=520, max_points=4000):
    Z = auto.Z_
    if Z.shape[1] >= 2:
        Zc = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Zc[np.random.default_rng(0).choice(len(Z), min(len(Z), 20000), replace=False)],
                                 full_matrices=False)
        P = Zc @ Vt[:2].T
    else:
        P = np.c_[Z[:, 0], np.zeros(len(Z))]
    rng = np.random.default_rng(0)
    normal = np.flatnonzero(~auto.anomaly_)
    anom = np.flatnonzero(auto.anomaly_)
    if len(normal) > max_points:
        normal = rng.choice(normal, max_points, replace=False)
    if len(anom) > 400:
        anom = anom[np.argsort(-auto.anomaly_score_[anom])[:400]]
    shown = np.r_[normal, anom]
    lo, hi = np.percentile(P[shown], [0.5, 99.5], axis=0)
    lo = np.minimum(lo, P[anom].min(0)) if len(anom) else lo
    hi = np.maximum(hi, P[anom].max(0)) if len(anom) else hi
    span = np.maximum(hi - lo, 1e-9)
    pad = 16
    sc = lambda p: pad + (p - lo) / span * (size - 2 * pad)
    out = [f'<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:{size}px" role="img" '
           f'aria-label="Rows projected on the two main axes of the clustering space">']
    for i in normal:
        lab = auto.labels_[i]
        col = PALETTE[lab] if lab < len(PALETTE) else OTHER
        px, py = sc(P[i])
        out.append(f'<circle cx="{px:.1f}" cy="{size - py:.1f}" r="1.8" fill="{col}" fill-opacity="0.7"/>')
    for i in anom:
        px, py = sc(P[i])
        py = size - py
        out.append(f'<path d="M{px - 3:.1f},{py - 3:.1f}L{px + 3:.1f},{py + 3:.1f}M{px - 3:.1f},{py + 3:.1f}'
                   f'L{px + 3:.1f},{py - 3:.1f}" stroke="var(--anom)" stroke-width="1.6">'
                   f'<title>row {i}, anomaly score {auto.anomaly_score_[i]:.2f}</title></path>')
    out.append('</svg>')
    return "".join(out)


def _anomaly_rows(auto, data, top=25):
    idx = np.flatnonzero(auto.anomaly_)
    idx = idx[np.argsort(-auto.anomaly_score_[idx])][:top]
    if not len(idx):
        return '<p class="muted">No anomalies at this setting.</p>'
    p = auto.preprocessor_
    S = auto._scaled_cache if hasattr(auto, '_scaled_cache') else None
    rows = []
    df = None
    if data is not None:
        from .auto import _as_frame, _to_numeric
        df, _ = _as_frame(data)
        S = p.scaled(df)
    head = '<tr><th>Row</th><th>Score</th><th>Group</th><th>Most unusual columns</th></tr>'
    for i in idx:
        cells = ''
        if S is not None:
            order = np.argsort(-np.abs(S[i]))[:3]
            parts = []
            for j in order:
                name = p.used_[j]
                v, _ = _to_numeric(np.asarray(df[name])[i:i + 1])
                parts.append(f"{_e(name)} = {_fmt(float(v[0]))} "
                             f"<span class='muted'>({S[i, j]:+.1f} spreads)</span>")
            cells = '; '.join(parts)
        rows.append(f"<tr><td>{i}</td><td>{auto.anomaly_score_[i]:.2f}</td>"
                    f"<td>{auto.anomaly_group_[i]}</td><td>{cells}</td></tr>")
    return f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>'


def _tree_html(auto):
    """The cluster hierarchy as an indented table, top-down."""
    levels = auto.levels_
    head = ("<h2>Cluster tree</h2><p class='muted'>The data has stable structure at "
            f"{len(levels)} resolutions: " + " → ".join(
                f"{lv['k']} clusters" for lv in levels) +
            ". Each cluster splits into the sub-clusters listed under it. A node is "
            "described by what sets it apart from its parent (top level: from all rows). "
            "Every level is in <code>labels.csv</code> as <code>gaca_level_1</code>, "
            "<code>gaca_level_2</code>, …; <b>bold</b> rows are the level used for "
            "<code>gaca_cluster</code>.</p>")
    rows = []
    for n in auto.tree_:
        pad = 8 + 22 * (n['level'] - 1)
        top = n['name'].split('.')[0]
        col = PALETTE[int(top)] if top.isdigit() and int(top) < len(PALETTE) else OTHER
        feats = ", ".join(f"{_e(c)} {'▲' if d > 0 else '▼'} {abs(d):.1f}"
                          for c, d in n['distinctive']) or '–'
        tag = f" <span class='muted'>(cluster {n['cluster']})</span>" if n['chosen'] else ''
        weight = "font-weight:600;" if n['chosen'] else ''
        rows.append(f"<tr><td style='padding-left:{pad}px;{weight}'>"
                    f"<span class='swatch' style='background:{col}'></span>{_e(n['name'])}{tag}</td>"
                    f"<td>{n['size']:,}</td><td>{100 * n['share']:.1f}%</td><td>{feats}</td></tr>")
    return (head + "<div class='scroll'><table><tr><th>Cluster</th><th>Rows</th><th>Share</th>"
            "<th>Sets it apart</th></tr>" + "".join(rows) + "</table></div>")


def write_report(auto, path, data=None, title=None):
    p = auto.preprocessor_
    sel = auto.gamma_selection_
    n_anom = int(auto.anomaly_.sum())
    n_groups = len(set(auto.anomaly_group_[auto.anomaly_]))
    title = title or "GACA clustering report"

    cards = [(f"{auto.n_rows_:,}", "rows"), (f"{len(p.used_)}", "columns used"),
             (f"{p.n_dims_}-D", "clustering space" + (" (PCA)" if p.pca_ is not None else "")),
             (f"{auto.n_clusters_}", "clusters"),
             (f"{n_anom:,}", f"anomalous rows ({100 * n_anom / max(auto.n_rows_, 1):.2f}%)"),
             (f"{n_groups}", "anomaly groups")]

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>",
             f"<h1>{_e(title)}</h1>",
             f"<p class='sub'>Gravitational Accretion Clustering (GACA), "
             f"{datetime.date.today().isoformat()}</p>",
             "<div class='cards'>" + "".join(
                 f"<div class='card'><div class='v'>{v}</div><div class='l'>{_e(l)}</div></div>"
                 for v, l in cards) + "</div>"]

    if sel is not None and not sel['found']:
        parts.append("<p class='note'>No resolution produced a stable split into two or more "
                     "clusters. The data looks like one continuous group in these columns; "
                     "anomalies are still reported.</p>")

    # Clusters
    parts.append("<h2>Clusters</h2><p class='muted'>Largest first. <em>Sets it apart</em> lists the "
                 "columns whose median in the cluster differs most from the overall median, in "
                 "units of the column's spread (interquartile range / 1.35).</p>")
    show_cols = p.used_[:6]
    head = ("<tr><th>Cluster</th><th>Rows</th><th>Share</th><th>Sets it apart</th>"
            + "".join(f"<th>median {_e(c)}</th>" for c in show_cols) + "</tr>")
    body = []
    for cl in auto.clusters_:
        k = cl['cluster']
        col = PALETTE[k] if k < len(PALETTE) else OTHER
        dist = ", ".join(f"<span class='{'up' if d > 0 else 'down'}'>{_e(n)} "
                         f"{'▲' if d > 0 else '▼'} {abs(d):.1f}</span>" for n, d in cl['distinctive'][:3])
        body.append(f"<tr><td><span class='swatch' style='background:{col}'></span>{k}</td>"
                    f"<td>{cl['size']:,}</td><td>{100 * cl['share']:.1f}%</td><td>{dist or '–'}</td>"
                    + "".join(f"<td>{_fmt(cl['medians'][c])}</td>" for c in show_cols) + "</tr>")
    parts.append(f"<div class='scroll'><table>{head}{''.join(body)}</table></div>")

    if getattr(auto, 'tree_', None):
        parts.append(_tree_html(auto))

    parts.append("<h2>Map</h2><p class='muted'>Rows projected on the two main axes of the "
                 "clustering space (a flat view of a higher-dimensional space, so clusters can "
                 "appear to overlap). Colours are clusters; red crosses are anomalies.</p>")
    parts.append(_scatter_svg(auto))

    parts.append("<h2>Anomalies</h2><p class='muted'>Rows that no cluster pulls: they would not "
                 "have moved in the simulation. Rows in the same group are near-copies of each "
                 "other. The score is −log<sub>10</sub> of the pull on the row relative to a "
                 f"typical row; rows above {-np.log10(auto.kappa):.0f} are anomalies.</p>")
    parts.append(_anomaly_rows(auto, data))

    if sel is not None:
        parts.append("<h2>How the resolution was chosen</h2>")
        parts.append("<p class='muted'>The bandwidth γ was swept over a range. At each value GACA "
                     "was fitted on four random subsamples. GACA picks the longest run of values "
                     "giving the same number of clusters (a plateau), and within it the value "
                     "where the subsamples agree most. "
                     "<span style='color:var(--accent)'>Blue</span> is the chosen value, "
                     "<span style='color:#efb118'>amber</span> marks other stable resolutions. "
                     "c is γ times the median squared distance between rows.</p>")
        parts.append(_sweep_svg(sel))
        alts = sel.get('alternatives', [])
        if alts:
            parts.append("<p>Other stable resolutions: " + "; ".join(
                f"{a['k']} clusters at γ = {a['gamma']:.3g} (stability {a['stability']:.2f})"
                for a in alts) + ". Pass <code>gamma=</code> to use one.</p>")

    parts.append("<h2>What the clustering saw</h2>")
    rows = "".join(f"<tr><td>{_e(c)}</td><td>{_e(d)}</td></tr>" for c, d in p.decisions_.items())
    parts.append(f"<div class='scroll'><table><tr><th>Column</th><th>Decision</th></tr>{rows}</table></div>")
    settings = [("scaling", p.scale), ("PCA", f"{p.n_dims_} components, "
                 f"{100 * p.pca_.explained_variance_ratio_.sum():.0f}% of variance"
                 if p.pca_ is not None else f"not needed ({len(p.used_)} ≤ max_dims {p.max_dims})"),
                ("γ", f"{auto.gamma_:.4g}" + (" (automatic)" if sel is not None else
                       f" (from bandwidth {auto.bandwidth:g})" if auto.bandwidth else " (given)")),
                ("ε", f"{auto.model_.epsilon_:.3g}"), ("coreset", f"{len(auto.model_.core_):,} rows"),
                ("saddle linking", f"τ = {auto.link_tau}" if auto.link_tau else "off"),
                ("random seed", auto.random_state)]
    parts.append("<table style='margin-top:16px'>" + "".join(
        f"<tr><th>{_e(a)}</th><td>{_e(b)}</td></tr>" for a, b in settings) + "</table>")
    parts.append("</main></body></html>")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("".join(parts))
