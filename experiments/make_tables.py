"""
Emit every thesis table as LaTeX, directly from the result CSVs.

Nothing here recomputes anything. It reads whatever result files exist and writes
tables.tex, so the numbers in the thesis are the numbers on disk by construction.
Hand-transcribing them is what made Table 6.3 disagree with coreset_results.csv.

Usage:
    python make_tables.py            # writes tables.tex, reports what is missing
    python make_tables.py --stdout   # also print everything

Inputs (each optional; a missing file just skips its table):
    dimensionality_results.csv          -> Table: dimensionality sweep
    sweep_results_eta0.5.csv            -> Table: bandwidth sweep
    robustness_summary.csv              -> Table: robustness under corruption
    clustering_benchmark.csv            -> Table: unsupervised benchmark
    scalability_comparison.csv          -> Table: comparative scaling
    coreset_results.csv                 -> Table: coreset-size sensitivity
    scalability_results.csv             -> prose numbers for section 6.3
    sun_profiles.csv                    -> prose numbers for section 5.2.1
"""
import os
import sys
import numpy as np
import pandas as pd

OUT = 'tables.tex'
NA = 'n/a'   # never leave an empty cell: an empty CSV field became a stray comma
             # in the LaTeX source of the old Table 6.2


# ---------------------------------------------------------------- formatting ---
def grp(x, dec=0):
    """LaTeX-safe number with a thin thousands separator: 16000 -> 16{,}000."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return NA
    s = f"{x:,.{dec}f}"
    return s.replace(',', '{,}')


def sig(x, dec=3):
    """Fixed-decimal number, negatives wrapped so the minus renders as math."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return NA
    return f"${x:.{dec}f}$" if x < 0 else f"{x:.{dec}f}"


def pm(mean, sd, dec=3):
    """mean +/- sd, or just the mean if the sd is missing."""
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return NA
    if sd is None or (isinstance(sd, float) and not np.isfinite(sd)):
        return sig(mean, dec)
    return f"${mean:.{dec}f} \\pm {sd:.{dec}f}$"


def sci_n(n):
    """10000 -> $10^{4}$, 30000 -> $3\\times10^{4}$."""
    n = int(n)
    e = int(np.floor(np.log10(n)))
    m = n / 10 ** e
    if abs(m - 1) < 1e-9:
        return f"$10^{{{e}}}$"
    mm = f"{m:g}"
    return f"${mm}\\times10^{{{e}}}$"


def read(name):
    if not os.path.exists(name):
        return None
    return pd.read_csv(name)


def get(row, col, default=np.nan):
    return row[col] if col in row.index and pd.notna(row[col]) else default


# -------------------------------------------------------------------- tables ---
def table_dimensionality(df):
    lines = [
        r'\begin{table}[H]', r'\centering',
        r'\caption{Effect of raw dimensionality $D$ on GACA, at fixed data '
        r'($20{,}000$ rows sampled from the $205{,}213$-company, $45$-feature '
        r'set; $\gamma=1.0$, $\epsilon=0.05$, $\eta=0.5$, $\theta=0.5$, coreset '
        r'$n_c=5{,}000$, seed $42$). As $D$ grows past $\approx 15$ the spatial '
        r'tree degrades, the runtime explodes, and the clustering shatters into '
        r'singletons. A preliminary PCA projection to $d=5$ restores all three.}',
        r'\label{tab:dimensionality}',
        r'\begin{tabular}{rrrrr}', r'\toprule',
        r'$D$ & Genesis (s) & Tree visited & Effective Suns & Silhouette \\',
        r'\midrule',
    ]
    sweep = df[df['label'].str.startswith('D=')]
    pca = df[~df['label'].str.startswith('D=')]
    for _, r in sweep.iterrows():
        lines.append(f"{int(r['D'])} & {grp(r['genesis_s'], 0)} & "
                     f"{r['tree_visit_frac']:.2f} & {grp(r['effective'], 0)} & "
                     f"{sig(r['silhouette'], 2)} \\\\")
    for _, r in pca.iterrows():
        lines += [r'\midrule',
                  f"\\textbf{{PCA}} $d{{=}}{int(r['D'])}$ & "
                  f"\\textbf{{{grp(r['genesis_s'], 0)}}} & "
                  f"\\textbf{{{r['tree_visit_frac']:.2f}}} & "
                  f"\\textbf{{{grp(r['effective'], 0)}}} & "
                  f"\\textbf{{{sig(r['silhouette'], 2)}}} \\\\"]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def table_gamma_sweep(df):
    ns = int(get(df.iloc[0], 'n_seeds', 1))
    eps = get(df.iloc[0], 'epsilon', 0.05)
    nc = int(get(df.iloc[0], 'coreset', 5000))
    g0 = df.iloc[0]
    glob = f"{get(g0, 'global_r2'):.3f}"
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        rf'\caption{{Bandwidth sweep on the $20{{,}}000$-company sample '
        rf'($\epsilon={eps}$, $\eta=0.5$, coreset $n_c={grp(nc)}$), reported as '
        rf'mean $\pm$ standard deviation over {ns} seeds. \emph{{Effective}} counts '
        rf'Suns that receive at least one training point; \emph{{Lone}} counts Suns '
        rf'receiving exactly one. GACA and K-Means $R^2$ are Mixture-of-Experts '
        rf'test scores with K-Means matched at $k=$ Effective; the global Ridge '
        rf'baseline is $R^2 = {glob}$. Suns with fewer than ten training points '
        rf'use a constant expert, and any point routed to an unpopulated Sun falls '
        rf'back to the global model.}}',
        r'\label{tab:gamma_sweep}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lrrrrrrr}', r'\toprule',
        r'$\gamma$ & Suns & Effective & Lone & Median size & Silhouette & '
        r'GACA $R^2$ & K-Means $R^2$ \\',
        r'\midrule',
    ]
    for _, r in df.sort_values('gamma').iterrows():
        lines.append(
            f"{r['gamma']:.1f} & "
            f"{pm(get(r, 'suns_found'), get(r, 'suns_found_sd'), 1)} & "
            f"{pm(get(r, 'effective'), get(r, 'effective_sd'), 1)} & "
            f"{pm(get(r, 'lone'), get(r, 'lone_sd'), 1)} & "
            f"{grp(get(r, 'median'), 1)} & "
            f"{pm(get(r, 'silhouette'), get(r, 'silhouette_sd'), 3)} & "
            f"{pm(get(r, 'gaca_r2'), get(r, 'gaca_r2_sd'), 3)} & "
            f"{pm(get(r, 'km_r2'), get(r, 'km_r2_sd'), 3)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}']
    return '\n'.join(lines)


def table_robustness(df):
    order = ['Global', 'KMeans', 'GACA', 'GACA+OOD']
    head = {'Global': 'Global Ridge', 'KMeans': 'K-Means',
            'GACA': 'GACA', 'GACA+OOD': 'GACA + OOD'}
    ns = int(df['n_seeds'].max())
    keff = df['effective'].mean()
    mean = df.pivot(index='sigma', columns='model', values='r2_mean')
    sd = df.pivot(index='sigma', columns='model', values='r2_sd')
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        rf'\caption{{Mixture-of-Experts test $R^2$ under increasing feature '
        rf'corruption ($\gamma=1.0$, $\eta=0.5$, $\epsilon=0.05$, coreset '
        rf'$n_c=5{{,}}000$), as mean $\pm$ standard deviation over {ns} seeds. '
        rf'K-Means is matched at $k=$ the number of effective GACA Suns '
        rf'({keff:.1f} on average), the same convention as '
        rf'Table~\ref{{tab:gamma_sweep}}. At each level, $10\%$ of test rows '
        rf'receive additive noise $\sigma Z$ with a single $Z \sim '
        rf'\mathcal{{N}}(0,I)$ drawn once, so severity is the only quantity that '
        rf'varies. Bold marks the best model at each noise level.}}',
        r'\label{tab:predictive_benchmark}',
        r'\begin{tabular}{l' + 'c' * len(order) + r'}', r'\toprule',
        r'Noise $\sigma$ & ' + ' & '.join(head[m] for m in order) + r' \\',
        r'\midrule',
    ]
    for sigma in mean.index:
        best = mean.loc[sigma, order].idxmax()
        cells = []
        for m in order:
            cell = pm(mean.loc[sigma, m], sd.loc[sigma, m], 4)
            cells.append(rf'\textbf{{{cell}}}' if m == best else cell)
        label = 'clean' if float(sigma) == 0.0 else f"{sigma:g}"
        lines.append(f"{label} & " + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def table_clustering(df):
    ren = {'GACA': 'GACA (coreset $n_c=5{,}000$)', 'K-Means': 'K-Means',
           'HDBSCAN': 'HDBSCAN', 'Mean-Shift': 'Mean-Shift'}
    sil = pd.to_numeric(df['Silhouette'], errors='coerce')
    ch = pd.to_numeric(df['Calinski-Harabasz'], errors='coerce')
    best_s, best_c = sil.idxmax(), ch.idxmax()
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Unsupervised clustering benchmark on $N=20{,}000$ companies '
        r'(five standardised surface features; GACA at $\gamma=1.0$, $\eta=0.5$, '
        r'$\epsilon=0.05$, coreset $n_c=5{,}000$). Mean-Shift runs at the bandwidth '
        r"matched to GACA's kernel scale through $\gamma = 1/(2h^2)$. Every timing "
        r'covers fit and assignment. Internal indices are computed for all methods '
        r"on one shared, seeded $5{,}000$-point sample. HDBSCAN's cluster count "
        r'excludes its noise label, and its indices are also reported with noise '
        r'points removed.}',
        r'\label{tab:unsupervised_benchmark}',
        r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Algorithm & Time (s) & Clusters found & Silhouette & '
        r'Calinski--Harabasz \\', r'\midrule',
    ]
    for i, r in df.iterrows():
        name = ren.get(r['Algorithm'], r['Algorithm'])
        s = sig(sil[i], 3)
        c = grp(ch[i], 1)
        if i == best_s:
            s = rf'\textbf{{{s}}}'
        if i == best_c:
            c = rf'\textbf{{{c}}}'
        lines.append(f"{name} & {r['Time (s)']:.2f} & "
                     f"{int(r['Clusters Found'])} & {s} & {c} \\\\")
        dn = pd.to_numeric(pd.Series([r.get('Silhouette (noise excl.)', '')]),
                           errors='coerce').iloc[0]
        if r['Algorithm'] == 'HDBSCAN' and np.isfinite(dn):
            dch = pd.to_numeric(pd.Series([r.get('CH (noise excl.)', '')]),
                                errors='coerce').iloc[0]
            lines.append(r'\quad \emph{noise excluded} & & & '
                         f"{sig(dn, 3)} & {grp(dch, 1)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def table_scaling(df):
    """Wall-clock ladder.

    A blank cell means the harness did not attempt that size, which is a design
    choice, not an observed failure. Only a recorded MemoryError is rendered as
    did-not-finish. Conflating the two is what let an earlier draft assert that
    HDBSCAN could not reach sizes it had never been asked to reach.
    """
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Clustering wall-clock time (seconds) versus dataset size, and '
        r'the HDBSCAN-to-GACA ratio. A dash means the size was \emph{not '
        r'attempted}: exact Mean-Shift is capped by the harness, and the cap is '
        r'a design choice rather than an observed failure. \textsc{dnf} is '
        r'reserved for a recorded memory error. HDBSCAN completed every size on '
        r'the ladder. Rows are shuffled before the ladder is built, so every size '
        r'is a random sample of the same population. All methods cluster the same '
        r'standardised features in memory.}',
        r'\label{tab:scaling_comparison}',
        r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'$N$ & GACA & MiniBatch K-Means & Mean-Shift & HDBSCAN & '
        r'HDBSCAN/GACA \\', r'\midrule',
    ]

    def cell(v, status=''):
        if pd.notna(v):
            return f"{v:,.2f}"
        return r'\textsc{dnf}' if 'Error' in str(status) else r'--'

    for _, r in df.sort_values('N').iterrows():
        if pd.notna(r.get('hdbscan_s')) and pd.notna(r.get('gaca_s')):
            v = r['hdbscan_s'] / r['gaca_s']
            # below unity HDBSCAN is the faster of the two; rounding that to
            # "0x" would hide the crossover, which is the point of the column
            ratio = (f"{v:,.0f}$\\times$" if v >= 10 else f"{v:.2f}$\\times$")
        else:
            ratio = r'--'
        lines.append(f"{sci_n(r['N'])} & {cell(r['gaca_s'])} & "
                     f"{cell(r['minikmeans_s'])} & "
                     f"{cell(r['meanshift_s'], r.get('meanshift_status'))} & "
                     f"{cell(r['hdbscan_s'], r.get('hdbscan_status'))} & "
                     f"{ratio} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']

    # the measured reason for each DNF, so the prose can cite it instead of
    # asserting a memory failure the code never tested
    notes = []
    for _, r in df.sort_values('N').iterrows():
        for name, col in (('Mean-Shift', 'meanshift_status'),
                          ('HDBSCAN', 'hdbscan_status')):
            st = str(r.get(col, ''))
            if 'probe' in st or 'Error' in st:
                notes.append(f"{name} at $N={grp(r['N'])}$: {st}")
    if notes:
        lines += [r'\vspace{0.5em}', r'{\footnotesize Measured truncation: ' +
                  '; '.join(notes).replace('_', r'\_') + r'.}']
    lines += [r'\end{table}']
    return '\n'.join(lines)


def table_coreset(df):
    ns = int(get(df.iloc[0], 'n_seeds', 1))
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        rf'\caption{{Coreset-size sensitivity ($\gamma=1.0$, $\eta=0.5$, '
        rf'$\epsilon=0.05$), as mean $\pm$ standard deviation over {ns} seeds. '
        rf'Each coreset is an independent draw, and the standardisation is fitted '
        rf'on the coreset alone; the Suns of each coreset are then used to assign a '
        rf'fixed held-out evaluation set. Genesis cost grows with $n_c$, but the '
        rf'recovered structure (largest-Sun share, effective Suns, silhouette) is '
        rf'invariant from the smallest coreset upward.}}',
        r'\label{tab:coreset}',
        r'\begin{tabular}{rrrrr}', r'\toprule',
        r'$n_c$ & Genesis (s) & Effective Suns & Largest share & Silhouette \\',
        r'\midrule',
    ]
    for _, r in df.sort_values('coreset').iterrows():
        lines.append(
            f"{grp(r['coreset'])} & "
            f"{pm(get(r, 'genesis_s'), get(r, 'genesis_s_sd'), 1)} & "
            f"{pm(get(r, 'effective'), get(r, 'effective_sd'), 1)} & "
            f"{pm(get(r, 'largest_share'), get(r, 'largest_share_sd'), 3)} & "
            f"{pm(get(r, 'silhouette'), get(r, 'silhouette_sd'), 3)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def table_hyperparameters():
    """Every parameter the pipeline takes, with the value used. The thesis
    currently never states theta, the leaf size, I_max or the distance floors."""
    rows = [
        (r'$\gamma$', 'Gaussian bandwidth', '1.0', r'Sec.~\ref{sec:latent_gam_operator}'),
        (r'$\epsilon$', 'Condensation radius', '0.05', r'Sec.~3.3'),
        (r'$\eta$', 'Transport damping', '0.5', r'Eqs.~3.9--3.11'),
        (r'$\theta$', 'Multipole acceptance threshold', '0.5', r'Sec.~\ref{sec:tree_approx}'),
        (r'$I_{\max}$', 'Iteration cap', '20', r'Alg.~\ref{alg:gam_pipeline}'),
        (r'$\ell$', 'GACANode leaf size', '1', r'Alg.~\ref{alg:gamnode}'),
        (r'$n_c$', 'Coreset size', r'$5{,}000$', r'Sec.~\ref{sec:coreset}'),
        (r'$d$', 'Latent dimension', '5', r'Sec.~\ref{sec:dim_experiment}'),
        (r'$B$', 'Streaming batch size', r'$10^5$', r'Sec.~\ref{sec:empirical_scaling}'),
        (r'$\delta_{\mathrm{MAC}}$', 'MAC distance floor', r'$10^{-9}$', r'Alg.~\ref{alg:kernel_sum}'),
        (r'$\delta$', 'Accretion distance floor', r'$10^{-9}$', r'Sec.~\ref{sec:accretion}'),
        (r'tol', 'Mean-movement convergence tolerance', r'$10^{-7}$', r'Alg.~\ref{alg:gam_pipeline}'),
        (r'$\alpha$', 'Ridge regularisation (experts)', '1.0', r'Sec.~4.3.4'),
        (r'$n_{\min}$', 'Minimum points for a local expert', '10', r'Sec.~4.3.4'),
    ]
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Every parameter of the GACA pipeline and the value used '
        r'throughout this thesis unless a sweep states otherwise. All experiments '
        r'use a fixed set of random seeds, reported per table.}',
        r'\label{tab:hyperparameters}',
        r'\begin{tabular}{llll}', r'\toprule',
        r'Symbol & Meaning & Value & Defined in \\', r'\midrule',
    ]
    lines += [' & '.join(r) + r' \\' for r in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ------------------------------------------------------------- prose numbers ---
def prose_streaming(df):
    last = df.iloc[-1]
    out = ['% ---- numbers for Section 6.3 (empirical validation) ----']
    for target in (1e6, 5e6, 1e7):
        row = df.iloc[(df['N'] - target).abs().argmin()]
        out.append(f"%   N={int(row['N']):>12,}  accretion {row['accretion_s']:.2f}s  "
                   f"peak RSS {row['peak_mb']:.1f} MB  "
                   f"effective Suns {int(row['effective_suns'])}  "
                   f"largest share {row['largest_share']:.4f}")
    us = last['accretion_s'] / last['N'] * 1e6
    out += [
        f"%   per-record cost: {us:.3f} microseconds (INCLUDES CSV parse; "
        f"t0 is outside the streaming loop)",
        f"%   peak RSS is flat at {df['peak_mb'].max():.1f} MB "
        f"(min {df['peak_mb'].min():.1f} MB) across the whole run",
        f"%   genesis (fixed in N): {last.get('genesis_s', float('nan')):.2f}s, "
        f"{int(last.get('genesis_iters', 0))} iterations, "
        f"{int(last.get('suns', 0))} Suns",
        f"%   extrapolation to 83M records: "
        f"{83e6 * us / 1e6:.0f}s of streaming plus the fixed genesis",
    ]
    return '\n'.join(out)


def prose_suns(df):
    tot = df['count'].sum()
    lone = df[df['tag'].str.contains('LONE', na=False)]
    out = ['% ---- numbers for Section 5.2.1 (what the Suns represent) ----',
           f"%   {len(df)} populated Suns over {tot:,} companies",
           f"%   largest Sun holds {df['count'].max() / tot * 100:.1f}% of companies",
           f"%   {len(lone)} Suns tagged as anomalies (<=3 members)"]
    for _, r in df.head(6).iterrows():
        out.append(f"%     Sun {int(r['sun']):>3} [{r['tag']:<14}] n={int(r['count']):>6,} "
                   f"({r['share_pct']:>5.2f}%)  med_emp={r['med_employees']:>7.0f}  "
                   f"med_rev={r['med_reviews']:>7.0f}  cat={r['top_category']}")
    return '\n'.join(out)


# ---------------------------------------------------------------------- main ---
def table_epsilon_sweep(df):
    """Condensation-radius sweep, from sweep_epsilon.py.

    Laid out with gamma as a blocked row group rather than a column, because the
    point of the table is that reading down an epsilon block changes almost
    nothing while stepping between blocks changes everything.
    """
    ns = int(get(df.iloc[0], 'n_seeds', 1))
    nc = int(get(df.iloc[0], 'coreset', 5000))
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        rf'\caption{{Condensation-radius sweep at fixed $\eta=0.5$ and coreset '
        rf'$n_c={grp(nc)}$, reported as mean $\pm$ standard deviation over {ns} '
        rf'seeds. Within each $\gamma$ block, $\epsilon$ varies over a twenty-fold '
        rf'range while the recovered structure does not: the spread across '
        rf'$\epsilon$ is smaller than the seed-to-seed spread at fixed $\epsilon$. '
        rf'Genesis time is the exception, falling by roughly a factor of two and a '
        rf'half, and so is cohesion at the coarsest setting.}}',
        r'\label{tab:epsilon_sweep}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'$\gamma$ & $\epsilon$ & Suns & Effective & Lone & Largest share & '
        r'Silhouette & Genesis (s) \\',
        r'\midrule',
    ]
    first = True
    for gamma, block in df.sort_values(['gamma', 'epsilon']).groupby('gamma'):
        if not first:
            lines.append(r'\addlinespace')
        first = False
        for i, (_, r) in enumerate(block.iterrows()):
            head = f"{gamma:.1f}" if i == 0 else ''
            lines.append(
                f"{head} & {r['epsilon']:g} & "
                f"{pm(get(r, 'suns_found'), get(r, 'suns_found_sd'), 1)} & "
                f"{pm(get(r, 'effective'), get(r, 'effective_sd'), 1)} & "
                f"{pm(get(r, 'lone'), get(r, 'lone_sd'), 1)} & "
                f"{pm(get(r, 'largest_share'), get(r, 'largest_share_sd'), 3)} & "
                f"{pm(get(r, 'silhouette'), get(r, 'silhouette_sd'), 3)} & "
                f"{sig(get(r, 'genesis_s'), 1)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}']
    return '\n'.join(lines)


def table_anomaly_recovery(df):
    """Coreset vs anomaly recovery, from anomaly_recovery.py.

    Reads the per-run file rather than the summary, because the headline of this
    experiment is a pair of conditional rates that cannot be recovered from
    per-cell means. Distances are pooled: the gate is coreset membership, and
    that does not depend on how far out the anomaly sits.
    """
    ns = df['seed'].nunique()
    samp = df[df.sampled]
    nsamp = df[~df.sampled]
    iso_given_samp = samp.isolated.mean() if len(samp) else float('nan')
    iso_given_not = nsamp.isolated.mean() if len(nsamp) else float('nan')
    ood_given_not = nsamp.ood_flagged.mean() if len(nsamp) else float('nan')

    pooled = (df.groupby(['coreset', 'multiplicity'], as_index=False)
              [['p_sampled_theory', 'sampled', 'isolated']].mean())

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        rf'\caption{{Recovery of planted anomalies as a function of coreset size '
        rf'$n_c$ and population multiplicity $m$, over {ns} seeds and pooled '
        rf'across the three injection distances. \emph{{Sampled}} is the share of '
        rf'groups with at least one copy drawn into the coreset; '
        rf'\emph{{Isolated}} the share recovered in a Sun containing no genuine '
        rf'data. The theoretical column is $1 - (1 - n_c/N)^m$ under uniform '
        rf'sampling. Conditionally, isolation given sampling is '
        rf'${iso_given_samp:.3f}$ and isolation given non-sampling is '
        rf'${iso_given_not:.3f}$, while the out-of-distribution rule of '
        rf'Section~\ref{{sec:imputation}} flags ${ood_given_not:.3f}$ of the '
        rf'groups the coreset missed.}}',
        r'\label{tab:anomaly_recovery}',
        r'\begin{tabular}{rrrrr}', r'\toprule',
        r'$n_c$ & $m$ & $1-(1-n_c/N)^m$ & Sampled & Isolated \\',
        r'\midrule',
    ]
    prev = None
    for _, r in pooled.sort_values(['coreset', 'multiplicity']).iterrows():
        nc = int(r['coreset'])
        if prev is not None and nc != prev:
            lines.append(r'\addlinespace')
        head = grp(nc) if nc != prev else ''
        prev = nc
        lines.append(
            f"{head} & {int(r['multiplicity'])} & "
            f"{sig(get(r, 'p_sampled_theory'), 3)} & "
            f"{sig(get(r, 'sampled'), 3)} & "
            f"{sig(get(r, 'isolated'), 3)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


SPECS = [
    ('dimensionality_results.csv', table_dimensionality, 'Table 4.1  dimensionality sweep'),
    ('anomaly_recovery_allseeds.csv', table_anomaly_recovery, 'Table 6.x  anomaly recovery'),
    ('sweep_results_eta0.5.csv', table_gamma_sweep, 'Table 5.1  bandwidth sweep'),
    ('eps_sweep.csv', table_epsilon_sweep, 'Table 5.x  condensation-radius sweep'),
    ('robustness_summary.csv', table_robustness, 'Table 5.2  robustness under corruption'),
    ('clustering_benchmark.csv', table_clustering, 'Table 5.4  unsupervised benchmark'),
    ('scalability_comparison.csv', table_scaling, 'Table 6.2  comparative scaling'),
    ('coreset_results.csv', table_coreset, 'Table 6.3  coreset sensitivity'),
]
PROSE = [
    ('scalability_results.csv', prose_streaming, 'Section 6.3 streaming numbers'),
    ('sun_profiles.csv', prose_suns, 'Section 5.2.1 Sun profiles'),
]


def main(to_stdout=False):
    blocks = ['% Generated by make_tables.py from the result CSVs. Do not hand-edit '
              'the numbers here;', '% re-run the experiment and re-run this script '
              'instead.', '']
    missing = []

    blocks += ['% ==================== hyperparameters ====================',
               table_hyperparameters(), '']

    for fname, fn, label in SPECS:
        df = read(fname)
        if df is None:
            missing.append((fname, label))
            continue
        try:
            blocks += [f'% ==================== {label}  (from {fname}) '
                       f'====================', fn(df), '']
            print(f"ok      {label:<44} <- {fname}")
        except Exception as e:                                  # noqa: BLE001
            print(f"FAILED  {label:<44} <- {fname}: {type(e).__name__}: {e}")

    for fname, fn, label in PROSE:
        df = read(fname)
        if df is None:
            missing.append((fname, label))
            continue
        try:
            blocks += [fn(df), '']
            print(f"ok      {label:<44} <- {fname}")
        except Exception as e:                                  # noqa: BLE001
            print(f"FAILED  {label:<44} <- {fname}: {type(e).__name__}: {e}")

    text = '\n'.join(blocks)
    with open(OUT, 'w') as f:
        f.write(text)
    print(f"\nWrote {OUT}")

    if missing:
        print("\nStill missing (run the matching experiment first):")
        for fname, label in missing:
            print(f"  {fname:<34} {label}")

    if to_stdout:
        print('\n' + '=' * 78 + '\n')
        print(text)


if __name__ == '__main__':
    main(to_stdout='--stdout' in sys.argv)
