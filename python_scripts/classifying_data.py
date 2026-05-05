import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/assessment_wide.parquet"
OUTPUT_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"

# ── Config ────────────────────────────────────────────────────────────────────
THRESHOLD      = 0.15   # ±15% from peer group median
MIN_GROUP_SIZE = 10     # minimum peers to trust a group — falls back to coarser if below

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# ── Convert numeric columns ───────────────────────────────────────────────────
numeric_cols = (
    ["GROSS_SQFT", "LAND_AREA", "NUM_BLDGS", "YRBUILT",
     "UNITS", "COOP_APTS", "BLD_STORY", "LOT_FRT", "LOT_DEP",
     "FINACTTOT", "FINACTLAND", "FINMKTTOT", "PYACTTOT",
     "RESIDENTIAL_AREA_GROSS"] +
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

# ── Bin continuous columns into peer-friendly ranges ─────────────────────────
# Raw continuous values make groups of 1 — binning fixes that
print("\nBinning continuous columns...")

# Year built → decade buckets (1950s, 1960s, etc.)
if "YRBUILT" in df.columns:
    df["YRBUILT_BIN"] = (df["YRBUILT"].fillna(0) // 10 * 10).astype(int).astype(str)
    print(f"  YRBUILT_BIN unique values: {sorted(df['YRBUILT_BIN'].unique())}")

# Residential area → quintile buckets (5 size bands: xs/sm/md/lg/xl)
# fillna with median instead of 0 to avoid duplicate bin edges
if "RESIDENTIAL_AREA_GROSS" in df.columns:
    resarea_median = df["RESIDENTIAL_AREA_GROSS"].median()
    resarea_filled = df["RESIDENTIAL_AREA_GROSS"].fillna(resarea_median)
    resarea_bins   = pd.qcut(resarea_filled, q=5, duplicates="drop", retbins=True)[1]
    n_resarea_bins = len(resarea_bins) - 1
    resarea_labels = ["xs", "sm", "md", "lg", "xl"][:n_resarea_bins]
    df["RESAREA_BIN"] = pd.cut(
        resarea_filled, bins=resarea_bins, labels=resarea_labels, include_lowest=True
    ).astype(str)
    print(f"  RESAREA_BIN distribution:\n{df['RESAREA_BIN'].value_counts()}")

# Market value FY2025 → quintile buckets (5 value tiers)
if "FINMKTTOT_FY2025" in df.columns:
    mktval_median = df["FINMKTTOT_FY2025"].median()
    mktval_filled = df["FINMKTTOT_FY2025"].fillna(mktval_median)
    mktval_bins   = pd.qcut(mktval_filled, q=5, duplicates="drop", retbins=True)[1]
    n_mktval_bins = len(mktval_bins) - 1
    mktval_labels = ["v_low", "low", "mid", "high", "v_high"][:n_mktval_bins]
    df["MKTVAL_BIN"] = pd.cut(
        mktval_filled, bins=mktval_bins, labels=mktval_labels, include_lowest=True
    ).astype(str)
    print(f"  MKTVAL_BIN distribution:\n{df['MKTVAL_BIN'].value_counts()}")

# ── Group hierarchy — coarsens automatically if a group is too small ──────────
# Tried finest → coarsest until MIN_GROUP_SIZE peers are found
GROUP_HIERARCHY = [
    # Finest: all 6 dimensions — most similar peers
    ["BORO", "BLDG_CLASS", "FINTAXCLASS", "YRBUILT_BIN", "RESAREA_BIN", "MKTVAL_BIN"],
    # Drop size bin
    ["BORO", "BLDG_CLASS", "FINTAXCLASS", "YRBUILT_BIN", "MKTVAL_BIN"],
    # Drop year bin
    ["BORO", "BLDG_CLASS", "FINTAXCLASS", "MKTVAL_BIN"],
    # Drop market value bin
    ["BORO", "BLDG_CLASS", "FINTAXCLASS"],
    # Coarsest fallback
    ["BORO", "BLDG_CLASS"],
]

# Filter each level to only columns that actually exist in the dataframe
GROUP_HIERARCHY = [
    [c for c in level if c in df.columns]
    for level in GROUP_HIERARCHY
]


# ── Classification function with fallback hierarchy ───────────────────────────
def classify_with_fallback(dataframe, value_col, area_col,
                            group_hierarchy=GROUP_HIERARCHY,
                            threshold=THRESHOLD,
                            min_group_size=MIN_GROUP_SIZE):
    """
    Classifies every property relative to its peer group median.
    Tries the finest grouping first; falls back to coarser groups
    for properties whose fine-grained group has fewer than min_group_size peers.
    """
    val_per_sqft = (dataframe[value_col] / dataframe[area_col]).copy()
    result       = pd.Series("unknown", index=dataframe.index)
    unresolved   = dataframe.index.copy()

    for level_idx, group_cols in enumerate(group_hierarchy):
        if len(unresolved) == 0:
            break

        sub     = dataframe.loc[unresolved]
        sub_vps = val_per_sqft.loc[unresolved]

        group_keys    = [sub[c] for c in group_cols]
        group_size    = sub_vps.groupby(group_keys).transform("count")
        group_medians = sub_vps.groupby(group_keys).transform("median")

        # Only classify properties whose group meets the minimum size
        large_enough = group_size >= min_group_size
        can_classify = unresolved[large_enough]

        upper = group_medians[large_enough] * (1 + threshold)
        lower = group_medians[large_enough] * (1 - threshold)
        vps   = sub_vps[large_enough]

        labels = np.select(
            [vps > upper, vps < lower],
            ["overvalued", "undervalued"],
            default="fairly_valued",
        )
        result.loc[can_classify] = labels

        # Pass remaining unresolved to next coarser level
        unresolved   = unresolved[~large_enough]
        n_resolved   = large_enough.sum()
        print(f"    Level {level_idx} ({'+'.join(group_cols)}): "
              f"classified {n_resolved:,} | still unresolved {len(unresolved):,}")

    if len(unresolved) > 0:
        print(f"    WARNING: {len(unresolved):,} properties unresolved after all fallbacks")

    return result


# ── Generate labels for 2020-2025 (historical features) ──────────────────────
print("\nGenerating historical labels...")
historical_years = [2020, 2021, 2022, 2023, 2024, 2025]

for yr in historical_years:
    col = f"FINACTTOT_FY{yr}"
    if col not in df.columns:
        print(f"  WARNING: {col} not found — skipping year {yr}")
        continue

    print(f"\n  FY{yr}:")
    mask   = df[col].notna() & (df[col] > 0)
    status = pd.Series("unknown", index=df.index)
    status[mask] = classify_with_fallback(df[mask], col, "GROSS_SQFT")

    df[f"overvalued_{yr}"]    = (status == "overvalued").astype(int)
    df[f"undervalued_{yr}"]   = (status == "undervalued").astype(int)
    df[f"fairly_valued_{yr}"] = (status == "fairly_valued").astype(int)

    dist = status[mask].value_counts()
    print(f"  FY{yr} ({mask.sum():,} rows): {dist.to_dict()}")

# ── Target variable: FY2026 ───────────────────────────────────────────────────
print("\nGenerating target variable (FY2026)...")
mask_2026 = df["FINACTTOT"].notna() & (df["FINACTTOT"] > 0)
df["target_2026"] = "unknown"
print("  FY2026:")
df.loc[mask_2026, "target_2026"] = classify_with_fallback(
    df[mask_2026], "FINACTTOT", "GROSS_SQFT"
)
print(f"Target distribution:\n{df['target_2026'].value_counts()}")
print(f"Target proportions:\n{df['target_2026'].value_counts(normalize=True).round(3)}")
print(f"Unknowns: {(df['target_2026'] == 'unknown').sum()}")

# ── Drop binning helper columns (not needed downstream) ───────────────────────
df = df.drop(columns=["YRBUILT_BIN", "RESAREA_BIN", "MKTVAL_BIN"], errors="ignore")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nFinal shape: {df.shape}")
print(f"Config: ±{THRESHOLD*100:.0f}% threshold, min group size={MIN_GROUP_SIZE}")
print(f"Columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")