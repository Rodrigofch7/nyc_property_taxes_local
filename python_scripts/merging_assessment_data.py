import pandas as pd

# ── Path (WSL format) ────────────────────────────────────────────────
BASE_PATH = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"

# ── Load FY2026 as the base (most recent assessment) ─────────────────
df_2026 = pd.read_parquet(f"{BASE_PATH}/assessment_FY2026.parquet")
print(f"FY2026 base shape: {df_2026.shape}")

# ── Years you want to add as historical columns ───────────────────────
years = [2020, 2021, 2022, 2023, 2024, 2025]

# ── Merge each year onto 2026 ─────────────────────────────────────────
df_final = df_2026.copy()

for y in years:
    parquet_file = f"{BASE_PATH}/assessment_FY{y}.parquet"
    try:
        df_prev = pd.read_parquet(
            parquet_file,
            columns=["BBL", "FINACTTOT", "FINACTLAND", "FINMKTTOT"]
        )

        # ensure unique BBL
        df_prev = df_prev.drop_duplicates(subset="BBL")

        # rename columns
        df_prev = df_prev.rename(columns={
            "FINACTTOT":  f"FINACTTOT_FY{y}",
            "FINACTLAND": f"FINACTLAND_FY{y}",
            "FINMKTTOT":  f"FINMKTTOT_FY{y}"
        })

        df_final = df_final.merge(df_prev, on="BBL", how="left")
        print(f"  Merged FY{y}: {df_prev.shape[0]:,} rows")

    except FileNotFoundError:
        print(f"  WARNING: FY{y} parquet not found — skipping")

# ── Done ─────────────────────────────────────────────────────────────
print(f"\nFinal shape: {df_final.shape}")
print(df_final.filter(like="FINACTTOT").head())

# ── Save ──────────────────────────────────────────────────────────────
output_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/assessment_wide.parquet"
df_final.to_parquet(output_path, index=False)
print(f"\nSaved to: {output_path}")