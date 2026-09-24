"""
Comparative-scaling figure for Chapter 6 (section 6.2.4).

Reads the CSV written by scalability_comparison.py and redraws the figure. It is
a separate script on purpose: the plotting in scalability_comparison.py sits
inside main(), so regenerating the figure from there would repeat the benchmark,
and the 2026-09-15 run took over eleven hours for the HDBSCAN column alone.

It also corrects a labelling problem in the original figure. That version tagged
both baselines "$O(N^2)$" in the legend and repeated the claim in a corner
annotation. The measured exponents do not support it: HDBSCAN grows as roughly
N^1.3 to N^1.4 while its spatial index holds and only becomes superquadratic over
the last two rungs. Asserting a complexity class the data does not show is the
same error as calling a size that was never attempted a did-not-finish, so this
version reports the measured exponents instead and marks the crossover.

Usage:
    python plot_scalability_comparison.py [scalability_comparison.csv]
"""
import sys

import matplotlib
matplotlib.use('Agg')  # file output only; no interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.grid': True,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Okabe-Ito (colourblind-safe)
BLUE, VERMILLION, GREEN, ORANGE, GREY = (
    '#0072B2', '#D55E00', '#009E73', '#E69F00', '#555555')


def exponents(n, t):
    """Empirical local exponent e in t ~ N^e between consecutive rungs."""
    return np.log(t[1:] / t[:-1]) / np.log(n[1:] / n[:-1])


def crossover(df, a='gaca_s', b='hdbscan_s'):
    """N at which series b overtakes series a, by log-linear interpolation."""
    d = df.dropna(subset=[a, b]).sort_values('N')
    r = np.log(d[b].values / d[a].values)
    for i in range(1, len(r)):
        if r[i - 1] < 0 <= r[i]:
            n0, n1 = np.log(d.N.values[i - 1]), np.log(d.N.values[i])
            f = -r[i - 1] / (r[i] - r[i - 1])
            return float(np.exp(n0 + f * (n1 - n0)))
    return None


def main(csv_path='scalability_comparison.csv'):
    df = pd.read_csv(csv_path).sort_values('N')

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    def series(col, colour, marker, label):
        d = df.dropna(subset=[col])
        if d.empty:
            return
        ax.plot(d['N'], d[col], marker=marker, color=colour, lw=1.8, ms=6,
                label=label)

    series('gaca_s', BLUE, 'o', 'GACA (coreset + streaming, no $k$)')
    series('minikmeans_s', ORANGE, '^', 'MiniBatch K-Means')
    series('meanshift_s', VERMILLION, 's', 'Mean-Shift (exact)')
    series('hdbscan_s', GREEN, 'd', 'HDBSCAN')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of records $N$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('Clustering runtime vs. dataset size')

    # Crossover: the honest headline is that the advantage is asymptotic and
    # begins at a specific, measurable size, not that the baselines fail.
    x = crossover(df)
    if x:
        ax.axvline(x, color=GREY, ls=':', lw=1.2, zorder=0)
        # top of the line, clear of the measured-growth note in the bottom right
        ax.text(x * 1.2, 0.72, f'crossover\n$N \\approx {x/1000:.0f}$k',
                transform=ax.get_xaxis_transform(), ha='left', va='center',
                fontsize=8.5, color=GREY)

    # Measured growth of the HDBSCAN column, stated rather than assumed.
    h = df.dropna(subset=['hdbscan_s'])
    e = exponents(h['N'].values, h['hdbscan_s'].values)
    ax.text(0.98, 0.03,
            f'HDBSCAN measured growth: $N^{{{e.min():.1f}}}$ to $N^{{{e.max():.1f}}}$\n'
            f'Mean-Shift capped at $N={int(df.dropna(subset=["meanshift_s"]).N.max()):,}$'
            .replace(',', '{,}'),
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color=GREY)

    ax.legend(frameon=False, loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig('scalability_comparison.pdf', bbox_inches='tight')
    fig.savefig('scalability_comparison.png', dpi=200, bbox_inches='tight')
    print('Wrote scalability_comparison.pdf / .png')

    print(f'\nCrossover (HDBSCAN overtakes GACA): N ~ {x:,.0f}' if x else
          '\nNo crossover found')
    print('HDBSCAN local exponents:')
    for i in range(len(e)):
        print(f'  {int(h.N.values[i]):>10,} -> {int(h.N.values[i+1]):>10,}'
              f'   N^{e[i]:.2f}')
    last = df.iloc[-1]
    if pd.notna(last.hdbscan_s):
        print(f'\nAt N={int(last.N):,}: HDBSCAN {last.hdbscan_s:,.0f}s vs '
              f'GACA {last.gaca_s:.1f}s  ->  '
              f'{last.hdbscan_s / last.gaca_s:,.0f}x')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'scalability_comparison.csv')
