"""
Phase-transition figure for Chapter 5 (section 5.1 / 5.2).

Reads the CSV written by sweep_gamma.py and produces a two-panel figure sharing
the bandwidth (gamma) axis:

  (a) Cluster structure   : Effective Suns and Lone Suns vs gamma  -> the phase
                            transition and emergent anomaly isolation (RQ2)
  (b) Downstream utility   : Mixture-of-Experts test R2 for GACA vs matched
                            K-Means vs the global baseline (RQ3)

Two separate panels (never a dual y-axis). Colours are the Okabe-Ito
colourblind-safe set; every series also has a distinct marker so identity never
depends on colour alone (needed for print / greyscale). Saved as vector PDF for
LaTeX (\\includegraphics) plus a PNG preview.

Usage:
    python plot_phase_transition.py [sweep_results_eta0.5.csv]
"""
import sys
import matplotlib
matplotlib.use('Agg')  # file output only; no interactive window
import matplotlib.pyplot as plt
import pandas as pd

# --- print styling (serif to sit next to LaTeX body text) ---
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

# Okabe-Ito (colourblind-safe). Validated: worst adjacent CVD dE 18.3.
BLUE, VERMILLION, GREEN, ORANGE, GREY = '#0072B2', '#D55E00', '#009E73', '#E69F00', '#555555'


def main(csv_path='sweep_results_eta0.5.csv'):
    df = pd.read_csv(csv_path).sort_values('gamma')
    g = df['gamma']

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.2, 6.4), sharex=True,
        gridspec_kw={'hspace': 0.12})

    # (a) cluster structure -----------------------------------------------
    ax_top.plot(g, df['effective'], color=BLUE, lw=2, marker='o', ms=6,
                label='Effective Suns')
    ax_top.plot(g, df['lone'], color=VERMILLION, lw=2, marker='s', ms=6,
                label='Lone Suns ($n=1$)')
    ax_top.set_ylabel('Number of Suns')
    ax_top.legend(frameon=False, loc='upper left')
    ax_top.set_title('(a) Cluster structure', loc='left', fontsize=10, pad=4)

    # (b) downstream predictive utility -----------------------------------
    ax_bot.plot(g, df['gaca_r2'], color=GREEN, lw=2, marker='o', ms=6,
                label='GACA experts')
    ax_bot.plot(g, df['km_r2'], color=ORANGE, lw=2, marker='^', ms=7,
                label='K-Means experts (matched $k$)')
    ax_bot.plot(g, df['global_r2'], color=GREY, lw=1.6, ls='--',
                label='Global baseline')
    ax_bot.set_ylabel('Test $R^2$ (Mixture of Experts)')
    ax_bot.set_xlabel('Bandwidth $\\gamma$')
    ax_bot.set_xscale('log')
    ax_bot.set_xticks(g)
    ax_bot.set_xticklabels([f'{v:g}' for v in g])
    ax_bot.legend(frameon=False, loc='upper left')
    ax_bot.set_title('(b) Downstream predictive utility', loc='left',
                     fontsize=10, pad=4)

    fig.savefig('phase_transition.pdf', bbox_inches='tight')
    fig.savefig('phase_transition.png', dpi=200, bbox_inches='tight')
    print('Wrote phase_transition.pdf and phase_transition.png')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'sweep_results_eta0.5.csv')
