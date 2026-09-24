"""
Regenerate the coreset-sweep figure from coreset_results.csv, WITHOUT re-running
the genesis sweep. Panel (c) plots the effective (populated) Sun count, which is
the stable convergence signal (the raw mode count wobbles with sampling).

Usage:
    python plot_coreset.py [coreset_results.csv]
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BLUE, GREEN = '#0072B2', '#009E73'


def main(csv='coreset_results.csv'):
    df = pd.read_csv(csv).sort_values('coreset')
    ks = df['coreset'].tolist()

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                         'grid.alpha': 0.4, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(6.2, 7.4), sharex=True,
                                     gridspec_kw={'hspace': 0.15})
    a1.plot(ks, df['genesis_s'], color=BLUE, lw=2, marker='o', ms=6)
    a1.set_ylabel('Genesis time (s)')
    a1.set_title('(a) Cost grows with coreset size', loc='left', fontsize=10, pad=4)

    a2.plot(ks, df['largest_share'], color=GREEN, lw=2, marker='s', ms=6)
    a2.set_ylabel('Largest-Sun\nmass share')
    a2.set_title('(b) Structure converges', loc='left', fontsize=10, pad=4)

    a3.plot(ks, df['effective'], color=BLUE, lw=2, marker='^', ms=7)
    a3.set_ylabel('Effective Suns')
    a3.set_title('(c) Effective-Sun count converges', loc='left', fontsize=10, pad=4)
    a3.set_xlabel('Coreset size $n_c$')
    a3.set_xscale('log')
    a3.set_xticks(ks)
    a3.set_xticklabels([f'{k // 1000}k' if k >= 1000 else str(k) for k in ks])

    fig.savefig('coreset_sweep.pdf', bbox_inches='tight')
    fig.savefig('coreset_sweep.png', dpi=200, bbox_inches='tight')
    print("Wrote coreset_sweep.pdf / coreset_sweep.png (panel c = effective Suns)")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'coreset_results.csv')
