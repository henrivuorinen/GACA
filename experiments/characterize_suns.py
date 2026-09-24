"""
Qualitative characterization of the Suns GACA discovers -- what real structure the
gravity simulation finds, and what the isolated anomalies (Lone Suns) actually are.

For a sample of companies it runs GACA clustering, then profiles each Sun: size,
share of the dataset, typical size / engagement / rating, dominant categories
(and industry / country if those columns exist), and a few example companies.
Smallest / singleton Suns are tagged as anomalies -- the entities GACA detaches
instead of absorbing into the mainstream, which is exactly the behaviour a
partitioning method like K-Means cannot produce.

Usage:
    python characterize_suns.py [file.csv] [gamma]
Default file is the descriptive 20k sample (it carries INDUSTRY/COUNTRY etc.);
gamma defaults to 2.0 to surface several archetypes (the quantitative benchmarks
use gamma=1.0). Try 1.0 vs 3.0 to see coarser vs finer structure.
"""
import sys
import os
import contextlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gaca import solar_genesis
from run_scalability_test import assign

RANDOM_SEED = 42
CORESET = 5000
EPS, ETA, THETA, N_ITER = 0.05, 0.5, 0.5, 20
N_SAMPLE = 50000
FEATURES = ['log_emp', 'log_reviews', 'age', 'LATITUDE', 'LONGITUDE']
DESC_COLS = ['GMAPS_PRIMARY_CATEGORY', 'INDUSTRY', 'COUNTRY']
SIZE_MAP = {'1-10': 5, '11-50': 30, '51-200': 125, '201-500': 350,
            '501-1000': 750, '1001-5000': 3000, '5001-10000': 7500, '10001+': 15000}


def load(path, n_sample):
    df = pd.read_csv(path, low_memory=False)
    founded = pd.to_numeric(df.get('FOUNDED'), errors='coerce').fillna(2015)
    df['age'] = 2026 - founded
    emp = pd.to_numeric(df.get('EMPLOYEES_COUNT'), errors='coerce')
    if 'SIZE' in df:
        emp = emp.fillna(df['SIZE'].map(SIZE_MAP))
    df['emp_derived'] = emp.fillna(1)
    df['log_emp'] = np.log1p(df['emp_derived'].clip(lower=0))
    rev = pd.to_numeric(df.get('TOTAL_GMAPS_REVIEW_COUNT'), errors='coerce').fillna(0)
    df['rev'] = rev
    df['log_reviews'] = np.log1p(rev.clip(lower=0))
    df['rating'] = pd.to_numeric(df.get('GMAPS_REVIEWS_AVERAGE'), errors='coerce')
    df['LATITUDE'] = pd.to_numeric(df.get('LATITUDE'), errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df.get('LONGITUDE'), errors='coerce')
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE']).reset_index(drop=True)
    if n_sample and len(df) > n_sample:
        df = df.sample(n_sample, random_state=RANDOM_SEED).reset_index(drop=True)
    return df, df[FEATURES].values.astype(float)


def main(path, gamma):
    df, X = load(path, N_SAMPLE)
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(RANDOM_SEED)
    core = Xs[rng.choice(len(Xs), min(CORESET, len(Xs)), replace=False)]

    print(f"Clustering {len(df):,} companies from {path} (gamma={gamma})...")
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        suns, masses = solar_genesis(core, gamma=gamma, n_iterations=N_ITER,
                                          theta=THETA, epsilon=EPS, eta=ETA)
        labels = assign(Xs, suns, masses)
    df['sun'] = labels
    desc = [c for c in DESC_COLS if c in df.columns]

    sizes = df['sun'].value_counts()
    lone = int((sizes <= 3).sum())
    print(f"\nGACA found {len(suns)} Suns; {sizes.size} populated, {lone} of them "
          f"anomalies (<=3 members). Largest holds {sizes.iloc[0] / len(df) * 100:.1f}% "
          f"of companies.")
    print("=" * 80)

    rows = []
    for rank, (sid, cnt) in enumerate(sizes.items()):
        sub = df[df['sun'] == sid]
        share = cnt / len(df) * 100
        tag = "MAINSTREAM" if rank == 0 else ("LONE / anomaly" if cnt <= 3 else "satellite")
        print(f"\nSun {sid}  [{tag}]   n = {cnt:,}  ({share:.2f}%)")
        print(f"   median employees = {sub['emp_derived'].median():.0f} | "
              f"median reviews = {sub['rev'].median():.0f} | "
              f"mean rating = {sub['rating'].mean():.2f}")
        for c in desc:
            top = sub[c].dropna().value_counts().head(3)
            if len(top):
                print(f"   top {c}: " + ", ".join(f"{k} ({v})" for k, v in top.items()))
        examples = sub['NAME'].dropna().astype(str).head(3).tolist()
        print(f"   examples: {', '.join(examples)}")

        top_cat = ''
        if 'GMAPS_PRIMARY_CATEGORY' in df and sub['GMAPS_PRIMARY_CATEGORY'].notna().any():
            top_cat = sub['GMAPS_PRIMARY_CATEGORY'].dropna().value_counts().index[0]
        rows.append(dict(sun=sid, count=cnt, share_pct=round(share, 3),
                         med_employees=sub['emp_derived'].median(),
                         med_reviews=sub['rev'].median(),
                         mean_rating=round(sub['rating'].mean(), 3),
                         top_category=top_cat, tag=tag))

    pd.DataFrame(rows).to_csv('sun_profiles.csv', index=False)
    print("\n" + "=" * 80)
    print("Saved sun_profiles.csv")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/20k_sample_data.csv'
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    main(path, gamma)
