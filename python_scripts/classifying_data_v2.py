"""
classifying_data_clustering.py
Classifies properties as undervalued, fairly_valued, or overvalued
using KMeans clustering to define peer groups dynamically.
Every property gets a label — no unknowns.

Strategy:
1. Cluster properties by structural + location features
2. Within each cluster compute assessed_value_per_sqft distribution
3. Label each property based on where it falls in its cluster distribution
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gc

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/assessment_wide.parquet"
OUTPUT_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"

# ── Configuration ─────────────────────────────────────────────────────────────
N_CLUSTERS  = 200    # number of peer groups — more = finer, fewer = coarser
THRESHOLD   = 0.15   # ±15% from cluster median
RANDOM_SEED = 42

print(f"Config: {N_CLUSTERS} clusters, ±{THRESHOLD*100:.0f}% threshold")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded shape: {df.shape}")

# ── Convert numeric columns ───────────────────────────────────────────────────
numeric_cols = (
    ["GROSS_SQFT", "LAND_AREA", "NUM_BLDGS", "YRBUILT",
     "UNITS", "COOP_APTS", "BLD_STORY", "LOT_FRT", "LOT_DEP",
     "FINACTTOT", "FINACTLAND", "FINMKTTOT", "PYACTTOT"] +
    [f"FINACTTOT_FY{y}"  for y in [2020, 2021, 2022, 2023, 2024, 2025]] +
    [f"FINACTLAND_FY{y}" for y in [2020, 2021, 2022, 2023, 2024, 2025]] +
    [f"FINMKTTOT_FY{y}"  for y in [2020, 2021, 2022, 2023, 2024, 2025]]
)
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Initial cleaning ──────────────────────────────────────────────────────────
df = df[df["GROSS_SQFT"] > 0].copy()
print(f"Shape after GROSS_SQFT filter: {df.shape}")

# ── Build clustering features ─────────────────────────────────────────────────
print("\nBuilding clustering features...")

# Encode categoricals
le_boro  = LabelEncoder()
le_bldg  = LabelEncoder()
le_zip   = LabelEncoder()
le_zone  = LabelEncoder()

df["BORO_ENC"]  = le_boro.fit_transform(df["BORO"].fillna("0").astype(str))
df["BLDG_ENC"]  = le_bldg.fit_transform(df["BLDG_CLASS"].fillna("Unknown").astype(str))
df["ZIP_ENC"]   = le_zip.fit_transform(df["ZIP_CODE"].fillna("00000").astype(str))
df["ZONE_ENC"]  = le_zone.fit_transform(df["ZONING"].fillna("Unknown").astype(str))

# Features for clustering — structural + location + basic value
# We use log transforms for skewed variables
cluster_features = [
    "BORO_ENC",
    "BLDG_ENC",
    "ZIP_ENC",
    "ZONE_ENC",
]

# Add numeric features
df["LOG_GROSS_SQFT"] = np.log1p(df["GROSS_SQFT"].fillna(0))
df["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"].fillna(0))
df["BUILDING_AGE"]   = (2026 - df["YRBUILT"].fillna(1900)).clip(0, 200)
df["SQFT_PER_UNIT"]  = (df["GROSS_SQFT"] / df["UNITS"].clip(lower=1)).clip(upper=50000)
df["BLD_STORY"]      = pd.to_numeric(df["BLD_STORY"], errors="coerce").fillna(0)

cluster_features += [
    "LOG_GROSS_SQFT",
    "LOG_LAND_AREA",
    "BUILDING_AGE",
    "SQFT_PER_UNIT",
    "BLD_STORY",
    "UNITS",
]

# Fill any remaining nulls with median
cluster_features = [f for f in cluster_features if f in df.columns]
X_cluster = df[cluster_features].copy()
for col in cluster_features:
    X_cluster[col] = pd.to_numeric(X_cluster[col], errors="coerce")
    X_cluster[col] = X_cluster[col].fillna(X_cluster[col].median())

print(f"  Clustering on {len(cluster_features)} features: {cluster_features}")

# ── Scale and cluster ─────────────────────────────────────────────────────────
print(f"\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
del X_cluster; gc.collect()

print(f"Running MiniBatchKMeans with {N_CLUSTERS} clusters...")
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_SEED,
    batch_size=10_000,
    n_init=10,
    verbose=0
)
df["CLUSTER"] = kmeans.fit_predict(X_scaled)
del X_scaled; gc.collect()

# Check cluster sizes
cluster_sizes = df["CLUSTER"].value_counts()
print(f"  Cluster size stats:")
print(f"    Min:    {cluster_sizes.min():,}")
print(f"    Median: {cluster_sizes.median():,.0f}")
print(f"    Max:    {cluster_sizes.max():,}")
print(f"    Clusters with < 10 props: {(cluster_sizes < 10).sum()}")


# ── Classification function ───────────────────────────────────────────────────
def classify_by_cluster(dataframe, value_col, area_col,
                         cluster_col="CLUSTER", threshold=THRESHOLD):
    """
    Classifies every property relative to its cluster median.
    Uses KNN fallback for any property without a valid value.
    No unknowns — every property gets a label.
    """
    val_per_sqft = dataframe[value_col] / dataframe[area_col]

    # Compute median per cluster
    cluster_medians = val_per_sqft.groupby(
        dataframe[cluster_col]
    ).transform("median")

    # For any property where val_per_sqft is null/zero,
    # use the cluster median itself (= fairly_valued)
    val_per_sqft = val_per_sqft.fillna(cluster_medians)
    cluster_medians = cluster_medians.fillna(val_per_sqft)

    upper = cluster_medians * (1 + threshold)
    lower = cluster_medians * (1 - threshold)

    conditions = [
        val_per_sqft > upper,
        val_per_sqft < lower,
    ]
    choices = ["overvalued", "undervalued"]

    result = np.select(conditions, choices, default="fairly_valued")
    return result


# ── Generate labels for 2020-2025 ────────────────────────────────────────────
print("\nGenerating historical labels...")
historical_years = [2020, 2021, 2022, 2023, 2024, 2025]

for yr in historical_years:
    col = f"FINACTTOT_FY{yr}"
    if col not in df.columns:
        print(f"  WARNING: {col} not found — skipping year {yr}")
        continue

    # For missing historical values, impute with cluster median
    col_filled = df[col].copy()
    cluster_medians_yr = df.groupby("CLUSTER")[col].transform("median")
    col_filled = col_filled.fillna(cluster_medians_yr)

    # Create temp df with filled values for classification
    temp = df.copy()
    temp[col] = col_filled.clip(lower=1)

    status = classify_by_cluster(temp, col, "GROSS_SQFT")

    df[f"overvalued_{yr}"]    = (status == "overvalued").astype(int)
    df[f"undervalued_{yr}"]   = (status == "undervalued").astype(int)
    df[f"fairly_valued_{yr}"] = (status == "fairly_valued").astype(int)

    dist = pd.Series(status).value_counts()
    print(f"  FY{yr} ({len(df):,} rows — all classified): {dist.to_dict()}")
    del temp; gc.collect()

# ── Target variable: FY2026 ───────────────────────────────────────────────────
print("\nGenerating target variable (FY2026)...")

# Impute missing FINACTTOT with cluster median
finacttot_filled = df["FINACTTOT"].copy()
cluster_med_2026 = df.groupby("CLUSTER")["FINACTTOT"].transform("median")
finacttot_filled = finacttot_filled.fillna(cluster_med_2026).clip(lower=1)

temp = df.copy()
temp["FINACTTOT"] = finacttot_filled

df["target_2026"] = classify_by_cluster(temp, "FINACTTOT", "GROSS_SQFT")
del temp; gc.collect()

print(f"Target distribution:\n{df['target_2026'].value_counts()}")
print(f"Target proportions:\n{df['target_2026'].value_counts(normalize=True).round(3)}")
print(f"Unknowns: {(df['target_2026'] == 'unknown').sum()} ← should be 0")

# ── Drop helper columns ───────────────────────────────────────────────────────
df = df.drop(columns=[
    "BORO_ENC", "BLDG_ENC", "ZIP_ENC", "ZONE_ENC",
    "LOG_GROSS_SQFT", "LOG_LAND_AREA", "BUILDING_AGE",
    "SQFT_PER_UNIT", "CLUSTER"
], errors="ignore")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nFinal shape: {df.shape}")
print(f"Config: {N_CLUSTERS} clusters, ±{THRESHOLD*100:.0f}% threshold")
print(f"No unknowns: {(df['target_2026'] == 'unknown').sum() == 0}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")