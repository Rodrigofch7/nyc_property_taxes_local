import pandas as pd
import numpy as np
import os

# ── Load data ─────────────────────────────────────────────────────────────────
df_sales  = pd.read_parquet("/mnt/c/Users/rodri/Downloads/sales_clean.parquet")
df_assess = pd.read_parquet("/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim/assessment_FY2024.parquet")

print(f"Sales shape: {df_sales.shape}")
print(f"Assessment shape: {df_assess.shape}")

# ── Filter sales to 2022-2024 ─────────────────────────────────────────────────
df_sales = df_sales[df_sales["SALE_YEAR"].isin([2022, 2023, 2024])].copy()
print(f"Sales 2022-2024 shape: {df_sales.shape}")
print(f"Sales by year:\n{df_sales['SALE_YEAR'].value_counts().sort_index()}")

# ── Check BBL format ──────────────────────────────────────────────────────────
print("\nBBL length sales:")
print(df_sales["BBL"].str.len().value_counts().sort_index())
print("\nBBL length assessment:")
print(df_assess["BBL"].str.len().value_counts().sort_index())

# ── Merge: keep all sales, bring in assessment data ───────────────────────────
# Using FY2024 assessment as cross-sectional snapshot for all years
# This is defensible because NYC assessments change slowly and lag the market
df_merged = df_sales.merge(df_assess, on="BBL", how="left", suffixes=("_sale", "_assess"))

print(f"\nMerged shape: {df_merged.shape}")
print(f"Rows with assessment data: {df_merged['FINACTTOT'].notna().sum()}")
print(f"Rows without assessment match: {df_merged['FINACTTOT'].isna().sum()}")
print(f"Match rate: {df_merged['FINACTTOT'].notna().mean():.1%}")

# ── Create assessment ratio ───────────────────────────────────────────────────
df_merged["FINACTTOT"] = pd.to_numeric(df_merged["FINACTTOT"], errors="coerce")
df_merged["UNITS"]     = pd.to_numeric(df_merged["UNITS"],     errors="coerce")
df_merged["COOP_APTS"] = pd.to_numeric(df_merged["COOP_APTS"], errors="coerce")

# Use per-unit assessed value when units > 1
df_merged["FINACTTOT_per_unit"] = np.where(
    df_merged["UNITS"].notna() & (df_merged["UNITS"] > 1),
    df_merged["FINACTTOT"] / df_merged["UNITS"],
    np.where(
        df_merged["COOP_APTS"].notna() & (df_merged["COOP_APTS"] > 1),
        df_merged["FINACTTOT"] / df_merged["COOP_APTS"],
        df_merged["FINACTTOT"]
    )
)

df_merged["assessment_ratio"] = df_merged["FINACTTOT_per_unit"] / df_merged["SALE PRICE"]

# ── Filter ────────────────────────────────────────────────────────────────────
df_merged = df_merged[
    df_merged["assessment_ratio"].notna() &
    (df_merged["assessment_ratio"] > 0) &
    (df_merged["SALE PRICE"] > 50000)
].copy()

print(f"\nShape after cleaning: {df_merged.shape}")
print("\nAssessment ratio summary:")
print(df_merged["assessment_ratio"].describe())
print("\nPercentiles:")
print(df_merged["assessment_ratio"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

# ── Labels using empirical percentiles by tax class ───────────────────────────
def assign_label(group):
    p25 = group["assessment_ratio"].quantile(0.25)
    p75 = group["assessment_ratio"].quantile(0.75)
    return pd.cut(
        group["assessment_ratio"],
        bins=[0, p25, p75, float("inf")],
        labels=["undervalued", "fairly_valued", "overvalued"]
    )

df_merged["label"] = df_merged.groupby(
    "TAX CLASS AT TIME OF SALE", group_keys=False
).apply(assign_label)

df_merged = df_merged.dropna(subset=["label"])

print("\nLabel distribution:")
print(df_merged["label"].value_counts())
print("\nLabel proportions:")
print(df_merged["label"].value_counts(normalize=True).round(3))
print("\nLabel by year:")
print(df_merged.groupby("SALE_YEAR")["label"].value_counts().unstack())

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"
os.makedirs(output_path, exist_ok=True)

out_file = os.path.join(output_path, "merged_2022_2024.parquet")
df_merged.to_parquet(out_file, index=False)
print(f"\nSaved to: {out_file}")
print(f"Final shape: {df_merged.shape}")