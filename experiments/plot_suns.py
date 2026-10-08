"""
Visualize the Suns GACA discovers (for the Chapter 5 characterization subsection).
Produces three figures:

  suns_geographic.pdf  -- companies by longitude/latitude, coloured by Sun, with
                          Lone Suns marked as stars: the archetypes as regions.
  suns_anomalies.pdf   -- log(reviews) vs log(employees); mainstream in grey, Lone
                          Suns highlighted and annotated: WHY they were isolated.
  suns_sizes.pdf       -- Suns ranked by size on a log axis: the heavy tail.

Colours are the Okabe-Ito colourblind-safe set; the six largest Suns get distinct
hues, the rest fold into a neutral "Other", and anomalies use a separate star
marker so identity never depends on colour alone.

Usage:
    python plot_suns.py [file.csv] [gamma]
"""
import sys
import os
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from characterize_suns import load, CORESET, EPS, ETA, THETA, N_ITER, RANDOM_SEED
from gaca import solar_genesis
from run_scalability_test import assign

PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#CC79A7']  # Okabe-Ito
GREY, ANOM = '#BBBBBB', '#111111'

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                     'font.size': 10, 'axes.grid': True, 'grid.linewidth': 0.4,
                     'grid.alpha': 0.4, 'axes.spines.top': False, 'axes.spines.right': False})


def main(path, gamma):
    df, X = load(path, 50000)
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(RANDOM_SEED)
    core = Xs[rng.choice(len(Xs), min(CORESET, len(Xs)), replace=False)]
    print(f"Clustering {len(df):,} companies (gamma={gamma})...")
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        suns, masses = solar_genesis(core, method='barnes_hut', gamma=gamma, n_iterations=N_ITER,
                                          theta=THETA, epsilon=EPS, eta=ETA)
        labels = assign(Xs, suns, masses)
    df['sun'] = labels

    sizes = df['sun'].value_counts()
    top_ids = list(sizes.index[:6])
    lone_ids = list(sizes.index[sizes <= 3])
    is_lone = df['sun'].isin(lone_ids)
    cmap = {sid: PALETTE[i] for i, sid in enumerate(top_ids)}

    # ---------- 1. geographic map ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    other = ~df['sun'].isin(top_ids) & ~is_lone
    ax.scatter(df.loc[other, 'LONGITUDE'], df.loc[other, 'LATITUDE'], s=2,
               c=GREY, alpha=0.4, linewidths=0, label='Other Suns')
    for sid in top_ids:
        m = (df['sun'] == sid) & ~is_lone
        ax.scatter(df.loc[m, 'LONGITUDE'], df.loc[m, 'LATITUDE'], s=3,
                   c=cmap[sid], alpha=0.6, linewidths=0,
                   label=f'Sun {sid} (n={sizes[sid]:,})')
    ax.scatter(df.loc[is_lone, 'LONGITUDE'], df.loc[is_lone, 'LATITUDE'], s=110,
               marker='*', c=ANOM, edgecolors='white', linewidths=0.6,
               label='Lone Suns (anomalies)', zorder=5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic distribution of discovered Suns', loc='left', fontsize=11)
    # custom legend handles so marker sizes are consistent (scatter dots are tiny,
    # the anomaly star is large -- markerscale cannot serve both)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=cmap[sid],
                      markersize=6, label=f'Sun {sid} (n={sizes[sid]:,})') for sid in top_ids]
    handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=GREY,
                          markersize=6, label='Other Suns'))
    handles.append(Line2D([0], [0], marker='*', color='none', markerfacecolor=ANOM,
                          markeredgecolor='white', markersize=13, label='Lone Suns (anomalies)'))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc='lower left', ncol=2)
    fig.savefig('suns_geographic.pdf', bbox_inches='tight')
    fig.savefig('suns_geographic.png', dpi=200, bbox_inches='tight')

    # ---------- 2. feature-space anomalies ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(df.loc[~is_lone, 'log_emp'], df.loc[~is_lone, 'log_reviews'], s=4,
               c=GREY, alpha=0.35, linewidths=0, label='Mainstream / satellites')
    ax.scatter(df.loc[is_lone, 'log_emp'], df.loc[is_lone, 'log_reviews'], s=120,
               marker='*', c=ANOM, edgecolors='white', linewidths=0.6,
               label='Lone Suns', zorder=5)
    for _, r in df[is_lone].sort_values('rev', ascending=False).head(3).iterrows():
        ax.annotate(f"{str(r['NAME'])[:24]}\n({int(r['rev']):,} reviews)",
                    (r['log_emp'], r['log_reviews']), textcoords='offset points',
                    xytext=(8, -2), fontsize=7, color=ANOM)
    ax.set_xlabel('$\\log(1+\\mathrm{employees})$')
    ax.set_ylabel('$\\log(1+\\mathrm{reviews})$')
    ax.set_title('Lone Suns include extreme review outliers', loc='left', fontsize=11)
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    fig.savefig('suns_anomalies.pdf', bbox_inches='tight')
    fig.savefig('suns_anomalies.png', dpi=200, bbox_inches='tight')

    # ---------- 3. heavy-tailed size distribution ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    s_sorted = np.sort(sizes.values)[::-1]
    ax.bar(range(1, len(s_sorted) + 1), s_sorted, color=PALETTE[0], width=0.75)
    ax.set_yscale('log')
    ax.set_xlabel('Sun (ranked by size)')
    ax.set_ylabel('Companies (log scale)')
    ax.set_title('Heavy-tailed cluster-size distribution', loc='left', fontsize=11)
    fig.savefig('suns_sizes.pdf', bbox_inches='tight')
    fig.savefig('suns_sizes.png', dpi=200, bbox_inches='tight')

    print("Wrote suns_geographic.pdf, suns_anomalies.pdf, suns_sizes.pdf (+ .png)")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/20k_sample_data.csv'
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    main(path, gamma)
