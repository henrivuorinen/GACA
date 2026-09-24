"""
What Solar Genesis actually looks like: two explanatory figures.

Chapter 4 describes the accretion in words and Chapter 3 gives the operator, but
nothing in the thesis shows the particle system actually moving. These two
figures do, and they are meant for the seminar rather than as evidence.

    accretion_evolution.png      the active system at five iterations, particle
                                 area proportional to accumulated mass
    accretion_trajectories.png   the path every starting point takes to the Sun
                                 that ends up holding it, with Lone Suns marked

HONEST FRAMING, and it belongs in the caption: this is a genuine GACA run, not a
cartoon, but it is run in TWO dimensions on two real features so that every merge
you see is a merge that happened rather than an artefact of projecting five
dimensions onto a page. The thesis experiments run in five. The bandwidth is
chosen so the final state shows both a dominant Mainstream Sun and at least one
singleton, which is the behaviour the figures are there to explain.

Trajectories are reconstructed by assigning every starting point to its nearest
active particle at each snapshot. Because condensation only ever merges particles
that are already within epsilon, and transport is a damped step, nearest-active
tracks true lineage closely; it is a drawing aid, not a measurement.

Usage:
    python plot_accretion_evolution.py [gamma]
"""
import contextlib
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from characterize_suns import load
from gaca import solar_genesis

DATA = 'data/20k_sample_data.csv'
FEATURES = ['log_emp', 'log_reviews']      # two real features, so 2-D is honest
CORESET = 3000
EPSILON, ETA, N_ITER, THETA = 0.05, 0.5, 20, 0.5
SEED = 42
PANELS = [0, 2, 8, 12, 20]                 # iterations to draw

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.grid': True,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.35,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
BLUE, VERMILLION, GREEN, ORANGE, GREY = (
    '#0072B2', '#D55E00', '#009E73', '#E69F00', '#555555')


def run(gamma, eps=None, eta=None):
    eps = EPSILON if eps is None else eps
    eta = ETA if eta is None else eta
    df, _ = load(DATA, 50_000)
    X = StandardScaler().fit_transform(df[FEATURES].values.astype(float))
    rng = np.random.default_rng(SEED)
    core = X[rng.choice(len(X), min(CORESET, len(X)), replace=False)]

    snaps = []
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        solar_genesis(core, gamma=gamma, n_iterations=N_ITER, theta=THETA,
                           epsilon=eps, eta=eta, snapshots=snaps)
    return core, snaps


def sizes(masses, lo=6.0, hi=900.0):
    """Marker area proportional to mass, square-rooted so a Sun holding a
    thousand particles does not swamp the panel."""
    m = np.sqrt(masses)
    if m.max() == m.min():
        return np.full_like(m, lo)
    return lo + (hi - lo) * (m - m.min()) / (m.max() - m.min())


def fig_evolution(snaps, gamma, out='accretion_evolution.png'):
    """Panels of the active system. A singleton is only labelled a Lone Sun in
    the final panel: mid-run, a mass-1 particle is simply one that has not merged
    yet, and marking those would misrepresent what a Lone Sun is."""
    idx = [i for i in PANELS if i < len(snaps)]
    fig, axes = plt.subplots(1, len(idx), figsize=(2.75 * len(idx), 3.5),
                             sharex=True, sharey=True)
    pos0 = snaps[0][0]
    pad = 0.5
    xlim = (pos0[:, 0].min() - pad, pos0[:, 0].max() + pad)
    ylim = (pos0[:, 1].min() - pad, pos0[:, 1].max() + pad)

    for ax, k in zip(axes, idx):
        pos, mass = snaps[k]
        final = (k == idx[-1])
        lone = (mass <= 1.0) & final          # Lone Suns exist only at the end
        ax.scatter(pos[~lone, 0], pos[~lone, 1], s=sizes(mass)[~lone],
                   c=BLUE, alpha=0.5, linewidths=0, zorder=2)
        if lone.any():
            ax.scatter(pos[lone, 0], pos[lone, 1], s=90, marker='*',
                       c=VERMILLION, linewidths=0, zorder=3, label='Lone Sun')
            ax.legend(frameon=False, fontsize=8, loc='lower right')
        held = mass.max() / mass.sum()
        ax.set_title(f'$t = {k}$\n{len(pos):,} particles'
                     + (f'\nlargest holds {held:.0%}' if k else '\nall mass = 1'),
                     fontsize=9.5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('log employees', fontsize=8.5)

    axes[0].set_ylabel('log reviews', fontsize=8.5)
    fig.suptitle(f'Solar Genesis: 3,000 particles condensing into '
                 f'{len(snaps[-1][0])} Suns '
                 f'($\\gamma={gamma}$, $\\epsilon={EPSILON}$, $\\eta={ETA}$, '
                 f'2-D illustration). Marker area $\\propto$ mass.',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Wrote {out}')


def fig_trajectories(core, snaps, gamma, out='accretion_trajectories.png'):
    """Exact lineage, not a nearest-neighbour guess.

    Condensation only ever replaces a component by its mass-weighted centre, and
    every member of that component lies within epsilon of it, so the parent of an
    active particle at step t-1 is unambiguously the nearest particle at step t.
    Chaining those one-step maps gives each starting particle its true path, which
    matters here: a Lone Sun has mass 1, so exactly one path may end at it.
    """
    chain = []
    for t in range(1, len(snaps)):
        prev, cur = snaps[t - 1][0], snaps[t][0]
        d = ((prev[:, None, :] - cur[None, :, :]) ** 2).sum(-1)
        chain.append(np.argmin(d, axis=1))

    paths = np.zeros((len(core), len(snaps), 2))
    idx = np.arange(len(core))
    paths[:, 0, :] = snaps[0][0][idx]
    for t in range(1, len(snaps)):
        idx = chain[t - 1][idx]
        paths[:, t, :] = snaps[t][0][idx]
    final_of = idx                                  # Sun index per starting point

    final_pos, final_mass = snaps[-1]
    lone = final_mass <= 1.0

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    step = max(1, len(core) // 700)
    for i in range(0, len(core), step):
        ax.plot(paths[i, :, 0], paths[i, :, 1], color=GREY, lw=0.35, alpha=0.22,
                zorder=1)
    ax.scatter(core[:, 0], core[:, 1], s=4, c=BLUE, alpha=0.3, linewidths=0,
               zorder=2)
    ax.scatter(final_pos[~lone, 0], final_pos[~lone, 1],
               s=sizes(final_mass)[~lone], c=ORANGE, alpha=0.9,
               edgecolors='white', linewidths=1.0, zorder=4)
    if lone.any():
        ax.scatter(final_pos[lone, 0], final_pos[lone, 1], s=150, marker='*',
                   c=VERMILLION, edgecolors='white', linewidths=0.8, zorder=5)
        # the single starting particle that ends alone: its whole path is a dot
        k = np.flatnonzero(lone)[int(np.argmax(
            np.linalg.norm(final_pos[lone] - final_pos[~lone].mean(0), axis=1)))]
        n_owned = int((final_of == k).sum())
        p = final_pos[k]
        ax.annotate(f'one particle, never moved\n($n={n_owned}$; pull decays '
                    f'as $e^{{-\\gamma d^2}}$)',
                    xy=p, xytext=(p[0] + 1.4, p[1] - 0.12), fontsize=8.5,
                    color=VERMILLION, ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', color=VERMILLION, lw=0.9,
                                    shrinkB=6))

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker='o', ls='', ms=4, color=BLUE, alpha=0.6,
               label='starting particles ($t = 0$)'),
        Line2D([], [], color=GREY, lw=0.8, alpha=0.6, label='path to its Sun'),
        Line2D([], [], marker='o', ls='', ms=10, color=ORANGE,
               markeredgecolor='white', label='Suns (area $\\propto$ mass)'),
        Line2D([], [], marker='*', ls='', ms=12, color=VERMILLION,
               markeredgecolor='white', label='Lone Suns (mass 1)'),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc='upper left',
              handletextpad=0.6, borderpad=0.2)

    ax.set_xlabel('log employees (standardised)')
    ax.set_ylabel('log reviews (standardised)')
    ax.set_title(f'Every particle\'s path to its Sun '
                 f'($\\gamma={gamma}$, {len(snaps)-1} iterations, 2-D)')
    fig.tight_layout()
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Wrote {out}')
    big = np.argmax(final_mass)
    print(f'  Mainstream Sun holds {int((final_of == big).sum()):,} of '
          f'{len(core):,} starting particles')


def main(gamma=1.0, eps=None, eta=None):
    global EPSILON, ETA
    EPSILON = EPSILON if eps is None else eps
    ETA = ETA if eta is None else eta
    core, snaps = run(gamma, eps, eta)
    print(f'gamma={gamma}: ' +
          ' -> '.join(str(len(p)) for p, _ in snaps))
    fp, fm = snaps[-1]
    print(f'final: {len(fp)} Suns, {int((fm <= 1.0).sum())} of them singletons, '
          f'largest holds {fm.max():.0f} of {fm.sum():.0f}')
    fig_evolution(snaps, gamma)
    fig_trajectories(core, snaps, gamma)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(float(a[0]) if len(a) > 0 else 1.0,
         float(a[1]) if len(a) > 1 else None,
         float(a[2]) if len(a) > 2 else None)
