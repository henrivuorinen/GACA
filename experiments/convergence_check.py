"""
Does the Sun count emerge from the dynamics, or is it set by the iteration cap?

This is the experiment that defends the thesis's central claim. GACA advertises
that k is not specified in advance. But every run so far stopped at I_max = 20
while the system was still merging (the 10M genesis went 14 -> 13 Suns on its
final iteration), so as things stand the iteration budget is a hyperparameter that
controls k just as surely as gamma and epsilon do.

Gaussian blurring mean-shift is known to have a degenerate fixed point: iterate it
long enough and everything collapses into one mode (Carreira-Perpinan). Clusters
therefore have to be read off before convergence. That is a legitimate design, but
it has to be stated and evidenced rather than left implicit.

This script runs a long simulation (default 200 iterations) at several bandwidths
and records the active-particle count at every step. Three outcomes are possible,
and each implies a different sentence in the thesis:

  (a) The count plateaus at some k > 1 and stays there. Then k really is emergent,
      I_max only has to be "large enough", and you say so with this figure.

  (b) The count decays slowly but monotonically to 1. Then k is set by the
      stopping rule, and the honest framing is a fixed iteration budget plus
      evidence that the structure is stable in a band around it (which panel (b)
      of the figure measures directly).

  (c) It plateaus for a long stretch and then collapses. This is the most likely
      outcome and the best one to report: name the plateau, show its width, and
      justify I_max as sitting inside it.

Usage:
    python convergence_check.py [file.csv]
Produces: convergence_results.csv, convergence.pdf/.png
"""
import sys
import os
import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from gaca import solar_genesis
from run_scalability_test import engineer, iter_chunks

RANDOM_SEED = 42
CORESET = 5000
EPS, ETA, THETA = 0.05, 0.5, 0.5
GAMMAS = [0.5, 1.0, 2.0, 5.0]
LONG_ITERS = 200        # far past the I_max = 20 used everywhere else
CAP_MARK = 20           # the cap the thesis currently uses
PLATEAU_TOL = 0         # a plateau is a run of iterations with no net merge


def load_coreset(path, n_rows=200_000, chunksize=100_000):
    rng = np.random.default_rng(RANDOM_SEED)
    rows, got = [], 0
    for ch in iter_chunks(path, chunksize):
        f = engineer(ch)
        rows.append(f)
        got += len(f)
        if got >= n_rows:
            break
    X = np.vstack(rows)[:n_rows]
    X = StandardScaler().fit_transform(X)
    return X[rng.choice(len(X), min(CORESET, len(X)), replace=False)]


def longest_plateau(counts):
    """Longest run of consecutive iterations over which the count does not drop,
    returned as (start_iteration, end_iteration, value)."""
    best = (0, 0, counts[0] if len(counts) else 0)
    i = 0
    while i < len(counts):
        j = i
        while j + 1 < len(counts) and counts[j + 1] >= counts[i] - PLATEAU_TOL:
            j += 1
        if (j - i) >= (best[1] - best[0]):
            best = (i + 1, j + 1, counts[i])
        i = j + 1
    return best


def main(path):
    print(f"Loading a {CORESET}-point coreset from {path}...")
    core = load_coreset(path)
    print(f"  coreset {core.shape}\n")

    records, summary = [], []
    for gamma in GAMMAS:
        hist = []
        t0 = time.time()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            suns, masses, iters = solar_genesis(
                core, gamma=gamma, n_iterations=LONG_ITERS, theta=THETA,
                epsilon=EPS, eta=ETA, return_iters=True, history=hist)
        elapsed = time.time() - t0

        counts = [h['n_after'] for h in hist]
        for h in hist:
            records.append(dict(gamma=gamma, **h))

        at_cap = counts[CAP_MARK - 1] if len(counts) >= CAP_MARK else counts[-1]
        final = counts[-1]
        s, e, v = longest_plateau(counts)
        collapsed = final <= 1
        summary.append(dict(gamma=gamma, n_at_cap=at_cap, n_final=final,
                            iters_run=iters, collapsed_to_one=collapsed,
                            plateau_start=s, plateau_end=e, plateau_value=v,
                            plateau_width=e - s + 1, seconds=elapsed))
        print(f"gamma={gamma:<4} | at I={CAP_MARK}: {at_cap:>4} Suns | "
              f"after {iters} iters: {final:>4} Suns | "
              f"longest plateau: {v} Suns over iterations {s}-{e} "
              f"(width {e - s + 1}) | {elapsed:.0f}s")

    pd.DataFrame(records).to_csv('convergence_results.csv', index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv('convergence_summary.csv', index=False)

    print("\n--- verdict ---")
    for _, r in sdf.iterrows():
        if r['collapsed_to_one']:
            verdict = ("collapses to a single Sun: k is set by the stopping rule, "
                       "report I_max as a stated budget")
        elif r['n_final'] == r['n_at_cap']:
            verdict = f"stable at {int(r['n_final'])} Suns: k is genuinely emergent"
        else:
            verdict = (f"still merging ({int(r['n_at_cap'])} -> {int(r['n_final'])}): "
                       f"plateau of width {int(r['plateau_width'])} is the honest claim")
        print(f"  gamma={r['gamma']:<4}: {verdict}")

    # ---------- figure ----------
    BLUE, GREEN, ORANGE, VERM, GREY = ('#0072B2', '#009E73', '#E69F00',
                                       '#D55E00', '#555555')
    colors = dict(zip(GAMMAS, [BLUE, GREEN, ORANGE, VERM]))
    markers = dict(zip(GAMMAS, ['o', 's', '^', 'd']))
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                         'grid.alpha': 0.4, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                 gridspec_kw={'hspace': 0.15})
    df = pd.DataFrame(records)
    for gamma in GAMMAS:
        d = df[df['gamma'] == gamma]
        a1.plot(d['iteration'], d['n_after'], color=colors[gamma], lw=1.8,
                marker=markers[gamma], ms=3, markevery=10, label=rf'$\gamma={gamma}$')
        a2.plot(d['iteration'], d['movement'], color=colors[gamma], lw=1.8,
                marker=markers[gamma], ms=3, markevery=10, label=rf'$\gamma={gamma}$')
    for ax in (a1, a2):
        ax.axvline(CAP_MARK, color=GREY, ls='--', lw=1)
        ax.set_yscale('log')
    a1.text(CAP_MARK, a1.get_ylim()[1], f' $I_{{\\max}}={CAP_MARK}$ ', color=GREY,
            va='top', ha='left', fontsize=8)
    a1.set_ylabel('Active particles\n(Suns after condensation)')
    a1.set_title('(a) Does the Sun count stabilise, or keep collapsing?',
                 loc='left', fontsize=10, pad=4)
    a1.legend(frameon=False, ncol=len(GAMMAS), fontsize=8)
    a2.set_ylabel('Mean transport\nmovement')
    a2.set_title('(b) Transport movement per iteration', loc='left',
                 fontsize=10, pad=4)
    a2.set_xlabel('Iteration $t$')
    fig.savefig('convergence.pdf', bbox_inches='tight')
    fig.savefig('convergence.png', dpi=200, bbox_inches='tight')
    print("\nWrote convergence.pdf / .png, convergence_results.csv, "
          "convergence_summary.csv")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/scalability_slim_1M.csv')
