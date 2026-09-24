"""
Regenerate the dimensionality figure from dimensionality_results.csv, WITHOUT
re-running the (very slow) experiment.

Panels: (a) genesis runtime, (b) Barnes-Hut degradation, (c) cluster cohesion --
the three clean, monotonic signals of the curse of dimensionality, each with the
PCA(d) rescue level marked. The distance-concentration CV is deliberately NOT
plotted: on heavy-tailed real data it is non-monotonic (a handful of outlier
companies dominate the pairwise-distance variance), so the curse manifests through
the tree / runtime / clustering channels rather than through clean concentration.

Usage:
    python plot_dimensionality.py [dimensionality_results.csv]
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BLUE, GREEN, GREY = '#0072B2', '#009E73', '#555555'


def main(csv='dimensionality_results.csv'):
    df = pd.read_csv(csv)
    sweep = df[df['label'].str.startswith('D=')].sort_values('D')
    pca = df[df['label'].str.startswith('PCA')].iloc[0]
    Ds = sweep['D'].tolist()

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                         'grid.alpha': 0.4, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(6.2, 7.6), sharex=True,
                                     gridspec_kw={'hspace': 0.15})

    a1.plot(Ds, sweep['genesis_s'], color=BLUE, lw=2, marker='o', ms=6)
    a1.set_yscale('log')
    a1.set_ylabel('Genesis time (s)')
    a1.set_title('(a) Runtime explodes', loc='left', fontsize=10, pad=4)
    a1.axhline(pca['genesis_s'], color=GREEN, ls=':', lw=1.5)
    a1.text(Ds[-1], pca['genesis_s'], 'PCA rescue ', color=GREEN, va='bottom',
            ha='right', fontsize=8)

    a2.plot(Ds, sweep['tree_visit_frac'], color=BLUE, lw=2, marker='s', ms=6)
    a2.set_ylabel('Fraction of tree\nnodes visited')
    a2.set_title('(b) Barnes--Hut degradation', loc='left', fontsize=10, pad=4)
    a2.axhline(pca['tree_visit_frac'], color=GREEN, ls=':', lw=1.5)

    a3.plot(Ds, sweep['silhouette'], color=BLUE, lw=2, marker='^', ms=7)
    a3.set_ylabel('Silhouette')
    a3.set_title('(c) Cluster cohesion collapses', loc='left', fontsize=10, pad=4)
    a3.axhline(0, color=GREY, lw=0.8)
    a3.axhline(pca['silhouette'], color=GREEN, ls=':', lw=1.5)
    a3.set_xlabel('Number of raw dimensions $D$')

    for ax in (a1, a2, a3):          # theoretical D~15 degradation threshold
        ax.axvline(15, color=GREY, ls='--', lw=1)

    fig.savefig('dimensionality.pdf', bbox_inches='tight')
    fig.savefig('dimensionality.png', dpi=200, bbox_inches='tight')
    print("Wrote dimensionality.pdf / dimensionality.png (genesis / tree / silhouette)")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'dimensionality_results.csv')
