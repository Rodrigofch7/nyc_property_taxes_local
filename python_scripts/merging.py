import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SALES_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/sales_clean.parquet"
ASSESS_BASE = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"
OUTPUT_DIR  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"

# ── 1. Load & prep sales ──────────────────────────────────────────────────────
df_sales = pd.read_parquet(SALES_PATH)
df_sales = df_sales[df_sales["SALE_YEAR"].isin([2020, 2021, 2022, 2023, 2024])].copy()

df_sales["SALE_MONTH"] = df_sales["SALE DATE"].dt.month

# ── NYC fiscal year logic ─────────────────────────────────────────────────────
# NYC Assessment calendar:
#   - Assessments are finalized January 5th each year
#   - Fiscal year runs July 1 → June 30
#   - A sale in Jan–Jun 2023 is taxed under FY2023 assessment
#   - A sale in Jul–Dec 2023 is taxed under FY2024 assessment
#
# So: FISCAL_YEAR = SALE_YEAR     if SALE_MONTH in [1..6]
#     FISCAL_YEAR = SALE_YEAR + 1 if SALE_MONTH in [7..12]
#     Capped at 2024 (latest assessment data available)
df_sales["FISCAL_YEAR"] = np.where(
    df_sales["SALE_MONTH"] >= 7,
    df_sales["SALE_YEAR"] + 1,
    df_sales["SALE_YEAR"]
)
df_sales["FISCAL_YEAR"] = df_sales["FISCAL_YEAR"].clip(upper=2024)

print("Sale year distribution:")
print(df_sales["SALE_YEAR"].value_counts().sort_index())
print("\nFiscal year distribution (assessment year in effect at sale):")
print(df_sales["FISCAL_YEAR"].value_counts().sort_index())

# Cross-tab to verify the mapping makes sense
print("\nSale year → Fiscal year cross-tab:")
print(pd.crosstab(df_sales["SALE_YEAR"], df_sales["FISCAL_YEAR"]))

# ── 2. Load assessments for each fiscal year ──────────────────────────────────
# All fiscal years get the same _FY{year} suffix — no special cases, no ambiguity
# FINACTTOT  = taxable assessed value (what drives the actual tax bill)
# FINMKTTOT  = city's implied full market value (used for fairness analysis)
# FINACTLAND = assessed value of land only

ASSESS_COLS = ["BBL", "FINACTTOT", "FINMKTTOT", "FINACTLAND", "UNITS", "COOP_APTS", "BLDG_CLASS"]

print("\nLoading assessment files...")
all_fy_dfs = {}
for fy in [2020, 2021, 2022, 2023, 2024]:
    df_fy = pd.read_parquet(
        f"{ASSESS_BASE}/assessment_FY{fy}.parquet",
        columns=ASSESS_COLS
    ).drop_duplicates(subset="BBL").rename(columns={
        "FINACTTOT":  f"FINACTTOT_FY{fy}",
        "FINMKTTOT":  f"FINMKTTOT_FY{fy}",
        "FINACTLAND": f"FINACTLAND_FY{fy}",
    })
    all_fy_dfs[fy] = df_fy
    print(f"  FY{fy}: {len(df_fy):,} properties loaded")

# Start with FY2024 as base (keep UNITS, COOP_APTS, BLDG_CLASS from most recent)
df_base = all_fy_dfs[2024].copy()

# Merge all prior years in
for fy in [2020, 2021, 2022, 2023]:
    df_base = df_base.merge(
        all_fy_dfs[fy][[
            "BBL",
            f"FINACTTOT_FY{fy}",
            f"FINMKTTOT_FY{fy}",
            f"FINACTLAND_FY{fy}",
        ]],
        on="BBL",
        how="left"
    )

# Confirm all expected columns exist
expected_cols = [
    f"{field}_FY{fy}"
    for fy in [2020, 2021, 2022, 2023, 2024]
    for field in ["FINACTTOT", "FINMKTTOT", "FINACTLAND"]
]
missing = [c for c in expected_cols if c not in df_base.columns]
if missing:
    raise ValueError(f"Missing assessment columns after merge: {missing}")

print(f"\nAssessment base table: {df_base.shape}")
print("All 15 FY columns present (FINACTTOT, FINMKTTOT, FINACTLAND × 5 years)")

# ── 3. Merge sales + assessment base ─────────────────────────────────────────
df = df_sales.merge(df_base, on="BBL", how="left")

print(f"\nShape after merging sales + assessments: {df.shape}")
print(f"BBL match rate: {df['FINACTTOT_FY2024'].notna().mean():.1%}")

# ── 4. Assign each sale its correct year's assessment ─────────────────────────
# For every sale row, pick the FY column that matches the fiscal year
# in effect at the time of that sale.
#
# Example:
#   Sale date = 2022-03-15 → FISCAL_YEAR = 2022 → use FINACTTOT_FY2022
#   Sale date = 2022-09-20 → FISCAL_YEAR = 2023 → use FINACTTOT_FY2023
#
# This is the core of the matching — every row gets the assessment
# that was ACTUALLY governing its tax bill on the day it sold.

fiscal_years = [2020, 2021, 2022, 2023, 2024]
conditions   = [df["FISCAL_YEAR"] == fy for fy in fiscal_years]

df["AV_TAXABLE_AT_SALE"] = np.select(
    conditions,
    [df[f"FINACTTOT_FY{fy}"] for fy in fiscal_years],
    default=np.nan
)
df["AV_FULLMKT_AT_SALE"] = np.select(
    conditions,
    [df[f"FINMKTTOT_FY{fy}"] for fy in fiscal_years],
    default=np.nan
)
df["AV_LAND_AT_SALE"] = np.select(
    conditions,
    [df[f"FINACTLAND_FY{fy}"] for fy in fiscal_years],
    default=np.nan
)

# ── 4a. Verify the year matching is correct ───────────────────────────────────
print("\nYear-assessment matching verification:")
print(f"{'FY':<6} {'Rows':>8} {'AV_TAXABLE null%':>18} {'Matches source col':>20}")
print("-" * 56)
for fy in fiscal_years:
    subset   = df[df["FISCAL_YEAR"] == fy]
    null_pct = subset["AV_TAXABLE_AT_SALE"].isna().mean() * 100
    # Every value in AV_TAXABLE_AT_SALE must equal FINACTTOT_FY{fy} for this subset
    match_rate = (
        subset["AV_TAXABLE_AT_SALE"] == subset[f"FINACTTOT_FY{fy}"]
    ).mean() * 100
    print(f"FY{fy}  {len(subset):>8,}  {null_pct:>17.1f}%  {match_rate:>19.1f}%")

overall_null = df["AV_TAXABLE_AT_SALE"].isna().mean()
print(f"\nOverall AV_TAXABLE_AT_SALE null rate: {overall_null:.1%}")
if overall_null > 0.03:
    print("WARNING: null rate above 3% — check FISCAL_YEAR distribution")
    print(df[df["AV_TAXABLE_AT_SALE"].isna()]["FISCAL_YEAR"].value_counts())

# ── 5. Clean numeric fields ───────────────────────────────────────────────────
for col in ["AV_TAXABLE_AT_SALE", "AV_FULLMKT_AT_SALE", "AV_LAND_AT_SALE", "SALE PRICE"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 6. Core fairness metrics ──────────────────────────────────────────────────
# Sales ratio = city's implied full market value / actual sale price
#   = 1.0 → city assessment matches market exactly (fair)
#   > 1.0 → city thinks property is worth MORE than it sold for → overtaxed
#   < 1.0 → city thinks property is worth LESS than it sold for → undertaxed
df["sales_ratio"] = df["AV_FULLMKT_AT_SALE"] / df["SALE PRICE"]

# Effective AV ratio = taxable AV / sale price
# This is what actually drives the tax bill
df["effective_av_ratio"] = df["AV_TAXABLE_AT_SALE"] / df["SALE PRICE"]

# Land share = how much of the assessed value is land vs building
df["land_share"] = df["AV_LAND_AT_SALE"] / df["AV_TAXABLE_AT_SALE"]

# ── 7. Filter to arm's-length sales ──────────────────────────────────────────
before = len(df)
df = df[
    (df["SALE PRICE"]  > 50_000) &
    (df["sales_ratio"] > 0.05)   &
    (df["sales_ratio"] < 10.0)
].copy()

print(f"\nFiltered {before - len(df):,} non-arm's-length / data-error rows")
print(f"Shape after filtering: {df.shape}")

# ── 8. Fairness labels ────────────────────────────────────────────────────────
p33 = df["sales_ratio"].quantile(0.33)
p66 = df["sales_ratio"].quantile(0.66)
print(f"\nGlobal sales ratio thresholds — p33: {p33:.3f} | p66: {p66:.3f}")

df["fairness_label"] = pd.cut(
    df["sales_ratio"],
    bins=[0, p33, p66, float("inf")],
    labels=["undertaxed", "fairly_assessed", "overtaxed"]
)

def label_within_class(group):
    p33 = group["sales_ratio"].quantile(0.33)
    p66 = group["sales_ratio"].quantile(0.66)
    return pd.cut(
        group["sales_ratio"],
        bins=[0, p33, p66, float("inf")],
        labels=["undertaxed", "fairly_assessed", "overtaxed"]
    )

df["fairness_label_within_class"] = (
    df.groupby("TAX CLASS AT TIME OF SALE", group_keys=False)
      .apply(label_within_class)
)

# ── 9. IAAO Coefficient of Dispersion (COD) by tax class ─────────────────────
def cod(s):
    med = s.median()
    return (s - med).abs().mean() / med * 100 if med != 0 else np.nan

summary = (
    df.groupby("TAX CLASS AT TIME OF SALE")["sales_ratio"]
      .agg(count="count", median_ratio="median", mean_ratio="mean", cod=cod)
      .round(3)
      .reset_index()
)
print("\n── Assessment Fairness by Tax Class ──")
print(summary.to_string(index=False))
print("\nCOD guide: <10 = excellent | 10–15 = acceptable | >20 = poor")

# ── 10. Final column inventory ────────────────────────────────────────────────
print("\nKey columns in output:")
key_cols = [
    "BBL", "SALE DATE", "SALE_YEAR", "FISCAL_YEAR",
    "SALE PRICE", "TAX CLASS AT TIME OF SALE",
    "AV_TAXABLE_AT_SALE", "AV_FULLMKT_AT_SALE", "AV_LAND_AT_SALE",
    "sales_ratio", "effective_av_ratio", "land_share",
    "fairness_label", "fairness_label_within_class"
]
print(df[key_cols].head(3).to_string())

# ── 11. Save ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_parquet(os.path.join(OUTPUT_DIR, "merged_fairness_2020_2024.parquet"), index=False)
summary.to_csv(os.path.join(OUTPUT_DIR, "cod_by_class.csv"), index=False)

print(f"\nSaved to {OUTPUT_DIR}")
print(f"Final shape: {df.shape}")