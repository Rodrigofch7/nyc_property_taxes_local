import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SALES_PATH = "/mnt/c/Users/rodri/Downloads/sales_clean.parquet"
ASSESS_BASE = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"
OUTPUT_DIR = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"

# ── 1. Load Sales ─────────────────────────────────────────────────────────────
df_sales = pd.read_parquet(SALES_PATH)
df_sales = df_sales[df_sales["SALE_YEAR"].isin([2020, 2021, 2022, 2023, 2024])].copy()

# ── 2. Build Multi-Year Assessment Table ──────────────────────────────────────
# Start with FY2024 as the base
df_assess_wide = pd.read_parquet(f"{ASSESS_BASE}/assessment_FY2024.parquet")

# Add previous years (FINACTTOT only)
years = [2020, 2021, 2022, 2023]
for y in years:
    df_prev = pd.read_parquet(
        f"{ASSESS_BASE}/assessment_FY{y}.parquet",
        columns=["BBL", "FINACTTOT"]
    ).drop_duplicates(subset="BBL")
    
    df_prev = df_prev.rename(columns={"FINACTTOT": f"FINACTTOT_FY{y}"})
    df_assess_wide = df_assess_wide.merge(df_prev, on="BBL", how="left")

# ── 3. Merge Sales with the Multi-Year Assessment Table ───────────────────────
df_merged = df_sales.merge(df_assess_wide, on="BBL", how="left", suffixes=("_sale", "_assess"))

# ── 4. Logic: Map SALE_YEAR to the correct Fiscal Year ────────────────────────
# We create a single 'active_assessment' column based on the year the sale happened
conditions = [
    (df_merged["SALE_YEAR"] == 2020),
    (df_merged["SALE_YEAR"] == 2021),
    (df_merged["SALE_YEAR"] == 2022),
    (df_merged["SALE_YEAR"] == 2023),
    (df_merged["SALE_YEAR"] == 2024)
]
choices = [
    df_merged["FINACTTOT_FY2020"],
    df_merged["FINACTTOT_FY2021"],
    df_merged["FINACTTOT_FY2022"],
    df_merged["FINACTTOT_FY2023"],
    df_merged["FINACTTOT"]  # This is the FY2024 value from the base file
]

df_merged["CURRENT_FINACTTOT"] = np.select(conditions, choices, default=np.nan)

# ── 5. Create assessment ratio (Using the year-specific assessment) ───────────
df_merged["CURRENT_FINACTTOT"] = pd.to_numeric(df_merged["CURRENT_FINACTTOT"], errors="coerce")
df_merged["UNITS"] = pd.to_numeric(df_merged["UNITS"], errors="coerce")
df_merged["COOP_APTS"] = pd.to_numeric(df_merged["COOP_APTS"], errors="coerce")

df_merged["FINACTTOT_per_unit"] = np.where(
    df_merged["UNITS"] > 1,
    df_merged["CURRENT_FINACTTOT"] / df_merged["UNITS"],
    np.where(
        df_merged["COOP_APTS"] > 1,
        df_merged["CURRENT_FINACTTOT"] / df_merged["COOP_APTS"],
        df_merged["CURRENT_FINACTTOT"]
    )
)

df_merged["assessment_ratio"] = df_merged["FINACTTOT_per_unit"] / df_merged["SALE PRICE"]

# ── 6. Filter & Label (Same as your original logic) ───────────────────────────
df_merged = df_merged[
    (df_merged["assessment_ratio"] > 0) & 
    (df_merged["SALE PRICE"] > 50000)
].copy()

def assign_label(group):
    p25 = group["assessment_ratio"].quantile(0.25)
    p75 = group["assessment_ratio"].quantile(0.75)
    return pd.cut(group["assessment_ratio"], bins=[0, p25, p75, float("inf")], labels=["undervalued", "fairly_valued", "overvalued"])

df_merged["label"] = df_merged.groupby("TAX CLASS AT TIME OF SALE", group_keys=False).apply(assign_label)

# ── 7. Save ───────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_file = os.path.join(OUTPUT_DIR, "merged_2020_2024_dynamic.parquet")
df_merged.to_parquet(out_file, index=False)

print(f"Final shape with dynamic assessment years: {df_merged.shape}")