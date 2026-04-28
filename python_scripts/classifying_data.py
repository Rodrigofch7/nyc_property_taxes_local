import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/assessment_wide.parquet"
OUTPUT_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded shape: {df.shape}")

# ── Convert numeric columns ───────────────────────────────────────────────────
numeric_cols = (
    ["GROSS_SQFT", "LAND_AREA", "NUM_BLDGS", "YRBUILT",
     "UNITS", "COOP_APTS", "BLD_STORY", "LOT_FRT", "LOT_DEP",
     "FINACTTOT", "FINACTLAND", "FINMKTTOT", "PYACTTOT"] +
    [f"FINACTTOT_FY{y}" for y in [2020, 2021, 2022, 2023, 2024, 2025]]
)
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Initial cleaning ──────────────────────────────────────────────────────────
# Remove properties with 0 or missing square footage
df = df[df["GROSS_SQFT"] > 0].copy()
print(f"Shape after GROSS_SQFT filter: {df.shape}")

# Drop columns not needed for modeling
cols_to_drop = [
    "FINACTLAND_FY2020", "FINMKTTOT_FY2020",
    "FINACTLAND_FY2021", "FINMKTTOT_FY2021",
    "FINACTLAND_FY2022", "FINMKTTOT_FY2022",
    "FINACTLAND_FY2023", "FINMKTTOT_FY2023",
    "FINACTLAND_FY2024", "FINMKTTOT_FY2024",
    "FINACTLAND_FY2025", "FINMKTTOT_FY2025",
]
df = df.drop(columns=cols_to_drop, errors="ignore")

# ── Classification function ───────────────────────────────────────────────────
def classify_valuation(dataframe, value_col, area_col,
                        group_cols=("BORO", "BLDG_CLASS"),
                        threshold=0.15):
    """
    Classifies properties based on assessed value per sqft
    relative to the median of their peer group (borough + building class).

    Args:
        dataframe:   the full DataFrame
        value_col:   assessed value column (e.g. FINACTTOT or FINACTTOT_FY2023)
        area_col:    square footage column
        group_cols:  columns that define the peer group
        threshold:   % deviation from group median to flag as over/under

    Returns:
        Series with values: 'overvalued', 'undervalued', 'fairly_valued'
    """
    val_per_sqft = dataframe[value_col] / dataframe[area_col]

    # Compute group median (peer group = same borough + building class)
    group_medians = val_per_sqft.groupby(
        [dataframe[c] for c in group_cols]
    ).transform("median")

    upper = group_medians * (1 + threshold)
    lower = group_medians * (1 - threshold)

    conditions = [
        val_per_sqft > upper,
        val_per_sqft < lower,
    ]
    choices = ["overvalued", "undervalued"]

    return np.select(conditions, choices, default="fairly_valued")

# ── Generate labels for 2020-2025 (historical features) ──────────────────────
# FY2026 is the base year (current assessment in FINACTTOT) → this is the TARGET
# FY2020-FY2025 are historical → these become features

historical_years = [2020, 2021, 2022, 2023, 2024, 2025]

for yr in historical_years:
    col = f"FINACTTOT_FY{yr}"

    if col not in df.columns:
        print(f"  WARNING: {col} not found — skipping year {yr}")
        continue

    # Only classify rows where historical value is available
    mask = df[col].notna() & (df[col] > 0)

    status = pd.Series("unknown", index=df.index)
    status[mask] = classify_valuation(
        df[mask], col, "GROSS_SQFT"
    )

    # Store as binary indicator columns (one-hot style)
    df[f"overvalued_{yr}"]    = (status == "overvalued").astype(int)
    df[f"undervalued_{yr}"]   = (status == "undervalued").astype(int)
    df[f"fairly_valued_{yr}"] = (status == "fairly_valued").astype(int)

    n_classified = mask.sum()
    dist = status[mask].value_counts()
    print(f"  FY{yr} ({n_classified:,} rows): {dist.to_dict()}")

# ── Target variable: FY2026 classification (FINACTTOT is the 2026 value) ──────
print("\nGenerating target variable (FY2026)...")
mask_2026 = df["FINACTTOT"].notna() & (df["FINACTTOT"] > 0)
df["target_2026"] = "unknown"
df.loc[mask_2026, "target_2026"] = classify_valuation(
    df[mask_2026], "FINACTTOT", "GROSS_SQFT"
)
print(f"Target distribution:\n{df['target_2026'].value_counts()}")
print(f"Target proportions:\n{df['target_2026'].value_counts(normalize=True).round(3)}")

# ── Feature summary ───────────────────────────────────────────────────────────
base_features = [
    "BORO", "BLOCK", "LOT", "BLDG_CLASS",
    "GROSS_SQFT", "LAND_AREA", "NUM_BLDGS",
    "YRBUILT", "UNITS", "COOP_APTS",
    "BLD_STORY", "LOT_FRT", "LOT_DEP",
    "ZIP_CODE", "ZONING", "PYACTTOT"
]

historical_status_features = [
    c for c in df.columns
    if any(str(y) in c for y in historical_years)
    and any(x in c for x in ["overvalued", "undervalued", "fairly_valued", "FINACTTOT"])
]

print(f"\nBase features ({len(base_features)}): {base_features}")
print(f"\nHistorical features ({len(historical_status_features)}): {historical_status_features}")
print(f"\nTarget: target_2026")
print(f"\nFinal shape: {df.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")