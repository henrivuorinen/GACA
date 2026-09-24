import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from gaca import GACA
import matplotlib.pyplot as plt


# --- shared preprocessing constants -------------------------------------------
# These are also imported by run_scalability_test.engineer() so that the in-memory
# pipeline of Chapter 5 and the streaming pipeline of Chapter 6 derive identical
# features. A fixed FOUNDED fallback (rather than a per-file median) is used
# deliberately: a median is not computable chunk-locally, so a constant is the
# only choice that makes the two pipelines agree and keeps chunking reproducible.
REFERENCE_YEAR = 2026
FOUNDED_FALLBACK = 2015

SIZE_MAP = {
    '1-10': 5,
    '11-50': 30,
    '51-200': 125,
    '201-500': 350,
    '501-1000': 750,
    '1001-5000': 3000,
    '5001-10000': 7500,
    '10001+': 15000,
}

# The five "surface" features used for clustering throughout Chapters 5 and 6.
# GMAPS_REVIEWS_AVERAGE is deliberately NOT among them: it is the regression
# target of RQ3, so clustering on it would put the target in the feature space.
SURFACE_FEATURES = ['log_emp', 'log_reviews', 'age', 'LATITUDE', 'LONGITUDE']

# The target attribute predicted by the Mixture-of-Experts experiments.
TARGET = 'GMAPS_REVIEWS_AVERAGE'


def load_and_clean_data(file_path, features=None, drop_missing_target=False):
    """Load and engineer the company sample.

    Returns ``(X_scaled, df_clean)`` where X_scaled holds ``features``
    (default: SURFACE_FEATURES) standardised over the whole file, and df_clean
    carries the engineered columns plus the untouched descriptive columns.

    The target GMAPS_REVIEWS_AVERAGE is left as NaN where it is missing. It used
    to be imputed with 0, which is outside the plausible 1-5 rating range and put
    a spurious spike at zero into both the RQ3 target and (when it was still part
    of the feature vector) the clustering space. Supervised callers should pass
    ``drop_missing_target=True`` so that only rows with an observed rating are
    used.
    """
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    # --- 1. Feature Engineering ---

    founded = pd.to_numeric(df.get('FOUNDED'), errors='coerce').fillna(FOUNDED_FALLBACK)
    df['age'] = (REFERENCE_YEAR - founded).clip(lower=0)

    # Employees: map the 'SIZE' band to its midpoint when EMPLOYEES_COUNT is null
    emp = pd.to_numeric(df.get('EMPLOYEES_COUNT'), errors='coerce')
    if 'SIZE' in df.columns:
        emp = emp.fillna(df['SIZE'].map(SIZE_MAP))
    df['emp_derived'] = emp.fillna(1)  # final fallback for tiny companies

    reviews = pd.to_numeric(df.get('TOTAL_GMAPS_REVIEW_COUNT'), errors='coerce').fillna(0)
    df['TOTAL_GMAPS_REVIEW_COUNT'] = reviews

    # The target is coerced but NOT imputed: missing means "no rating observed".
    df[TARGET] = pd.to_numeric(df.get(TARGET), errors='coerce')

    # --- 2. Log Transforms (Crucial for Power-Law data like Revenue/Employees) ---
    df['log_emp'] = np.log1p(df['emp_derived'].clip(lower=0))
    df['log_reviews'] = np.log1p(reviews.clip(lower=0))

    # --- 3. Selection ---
    features = list(SURFACE_FEATURES if features is None else features)

    # Drop rows that have NaN in Lat/Long (vital for spatial gravity)
    df_clean = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    if drop_missing_target:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=[TARGET])
        print(f"  dropped {before - len(df_clean):,} rows with no observed "
              f"{TARGET} ({len(df_clean):,} remain)")
    df_clean = df_clean.reset_index(drop=True)
    X = df_clean[features].values.astype(float)

    # --- 4. Scaling (Standardization) ---
    # This is ESSENTIAL so that 1 degree of Latitude isn't "heavier" than 1 employee
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df_clean


def run_real_data_test(csv_path):
    X, original_df = load_and_clean_data(csv_path)

    # Initialize the GACA
    # gamma: try 1.0 first, adjust if you get too many/few clusters
    # epsilon: merge radius in scaled space (0.1 is usually a good start)
    gaca = GACA(
        gamma_clustering=0.8,
        n_iterations=30,
        theta=0.7,
        epsilon=0.2,
        sample_size=10000
    )

    print(f"\nStarting Gravity Clustering on {X.shape[0]} companies...")
    gaca.fit(X)

    # Get assignments for all 20k rows
    assignments = gaca._assign_to_suns(X)
    original_df['cluster_id'] = assignments

    print(f"\n--- SUCCESS ---")
    print(f"Total Companies processed: {len(original_df)}")
    print(f"Number of Suns identified: {len(gaca.suns_)}")

    # --- 5. Analysis: What do the Suns represent? ---
    # Let's look at the average stats for each cluster
    summary = original_df.groupby('cluster_id').agg({
        'NAME': 'count',
        'emp_derived': 'mean',
        'TOTAL_GMAPS_REVIEW_COUNT': 'mean',
        'age': 'mean',
        'GMAPS_PRIMARY_CATEGORY': lambda x: x.value_counts().index[0] if not x.empty else "N/A"
    }).rename(columns={'NAME': 'Company_Count', 'GMAPS_PRIMARY_CATEGORY': 'Top_Category'})

    print("\nCluster profiles:")
    print(summary)

    # Visualize Geospatial Clusters (Lat/Long)
    lat_col = SURFACE_FEATURES.index('LATITUDE')
    lon_col = SURFACE_FEATURES.index('LONGITUDE')
    plt.figure(figsize=(10, 7))
    plt.scatter(original_df['LONGITUDE'], original_df['LATITUDE'],
                c=assignments, cmap='tab20', s=1, alpha=0.5)
    plt.scatter(gaca.suns_[:, lon_col], gaca.suns_[:, lat_col],  # scaled sun positions
                color='red', marker='X', s=50, label='Suns (Lat/Long projection)')
    plt.title("Geospatial Distribution of Gravitational Clusters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


if __name__ == "__main__":
    # Expects a CSV in the company-data format described in the README.
    run_real_data_test('data/20k_sample_data.csv')