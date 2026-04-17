import pandas as pd

# ── Path (WSL format) ────────────────────────────────────────────────
BASE_PATH = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"

# ── Load FY2024 (DO NOT TOUCH STRUCTURE) ─────────────────────────────
df_2024 = pd.read_parquet(f"{BASE_PATH}/assessment_FY2024.parquet")

# ── Years you want to add ────────────────────────────────────────────
years = [2020, 2021, 2022, 2023]

# ── Merge each year onto 2024 ────────────────────────────────────────
df_final = df_2024.copy()

for y in years:
    df_prev = pd.read_parquet(
        f"{BASE_PATH}/assessment_FY{y}.parquet",
        columns=["BBL", "FINACTTOT"]   # only what we need
    )

    # ensure unique BBL (important)
    df_prev = df_prev.drop_duplicates(subset="BBL")

    # rename column
    df_prev = df_prev.rename(columns={"FINACTTOT": f"FINACTTOT_FY{y}"})

    # merge into main dataset
    df_final = df_final.merge(df_prev, on="BBL", how="left")

# ── Done ─────────────────────────────────────────────────────────────
print("Final shape:", df_final.shape)
print(df_final.filter(like="FINACTTOT").head())
print(df_final.columns.tolist())

# ── Save output ───────────────────────────────────────────────────────
output_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/merged_2020_2024.parquet"

df_final.to_parquet(output_path, index=False)

print(f"Saved file to: {output_path}")