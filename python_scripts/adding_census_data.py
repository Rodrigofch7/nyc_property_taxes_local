"""
adding_census_data.py
Downloads ACS 5-year census data at ZIP code (ZCTA) level for NYC
and merges it into processed_labeled_data.parquet.

Key variables:
  B19013_001E = Median household income
  B25064_001E = Median gross rent
  B25077_001E = Median home value
  B03002_003E = White alone (non-Hispanic)
  B03002_001E = Total population (for racial % calculation)
  B25003_002E = Owner-occupied housing units
  B25003_001E = Total occupied housing units
  B17001_002E = Below poverty level
  B01003_001E = Total population
  B15003_022E = Bachelor's degree
  B15003_001E = Total pop 25+ (for education %)
"""

import pandas as pd
import numpy as np
import requests
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
OUTPUT_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data_census.parquet"
CENSUS_CACHE = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/census_zcta.parquet"

# ── Census API ────────────────────────────────────────────────────────────────
# Get a free key at https://api.census.gov/data/key_signup.html
# Or leave as None — the API works without a key but rate-limits more aggressively
CENSUS_API_KEY = 'fda60e79b0da81a8ac6472ff4250f47daa8c527b'

# ACS 5-year 2022 (most recent stable release)
ACS_YEAR = 2022


def fetch_census_data(year=ACS_YEAR, api_key=None):
    """
    Fetch ACS 5-year data at ZCTA (ZIP code tabulation area) level.
    Returns a DataFrame with one row per ZIP code.
    """
    if os.path.exists(CENSUS_CACHE):
        print(f"  Loading cached census data from {CENSUS_CACHE}")
        return pd.read_parquet(CENSUS_CACHE)

    print(f"  Fetching ACS {year} 5-year data from Census API...")

    variables = [
        "B19013_001E",  # median household income
        "B25064_001E",  # median gross rent
        "B25077_001E",  # median home value
        "B03002_003E",  # white alone non-hispanic
        "B03002_001E",  # total population (racial denom)
        "B25003_002E",  # owner occupied units
        "B25003_001E",  # total occupied housing units
        "B17001_002E",  # below poverty level
        "B01003_001E",  # total population
        "B15003_022E",  # bachelor's degree
        "B15003_001E",  # total pop 25+ (education denom)
    ]

    vars_str = ",".join(variables)
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"NAME,{vars_str}",
        "for": "zip code tabulation area:*",
    }
    if api_key:
        params["key"] = api_key

    print(f"  Calling: {base_url}")
    response = requests.get(base_url, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Rename columns
    df = df.rename(columns={
        "zip code tabulation area":  "ZIP_CODE",
        "B19013_001E":               "MEDIAN_INCOME",
        "B25064_001E":               "MEDIAN_RENT",
        "B25077_001E":               "MEDIAN_HOME_VALUE",
        "B03002_003E":               "POP_WHITE_ALONE",
        "B03002_001E":               "POP_TOTAL_RACE",
        "B25003_002E":               "OWNER_OCCUPIED",
        "B25003_001E":               "TOTAL_OCCUPIED",
        "B17001_002E":               "BELOW_POVERTY",
        "B01003_001E":               "TOTAL_POPULATION",
        "B15003_022E":               "BACHELORS_DEGREE",
        "B15003_001E":               "POP_25_PLUS",
    })

    # Convert to numeric (Census returns strings, -666666666 = missing)
    num_cols = [c for c in df.columns if c not in ["NAME", "ZIP_CODE"]]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-666666666, np.nan)
        df[col] = df[col].replace(-666666666.0, np.nan)

    # ── Derived features ──────────────────────────────────────────────────────
    # % white (racial composition)
    df["PCT_WHITE"] = (df["POP_WHITE_ALONE"] / df["POP_TOTAL_RACE"].clip(lower=1) * 100).round(2)

    # % owner occupied
    df["PCT_OWNER_OCCUPIED"] = (df["OWNER_OCCUPIED"] / df["TOTAL_OCCUPIED"].clip(lower=1) * 100).round(2)

    # % below poverty
    df["PCT_POVERTY"] = (df["BELOW_POVERTY"] / df["TOTAL_POPULATION"].clip(lower=1) * 100).round(2)

    # % with bachelor's degree
    df["PCT_BACHELORS"] = (df["BACHELORS_DEGREE"] / df["POP_25_PLUS"].clip(lower=1) * 100).round(2)

    # Log median income (handles skew)
    df["LOG_MEDIAN_INCOME"] = np.log1p(df["MEDIAN_INCOME"].clip(lower=0))

    # Keep only useful columns
    keep = [
        "ZIP_CODE",
        "MEDIAN_INCOME", "LOG_MEDIAN_INCOME",
        "MEDIAN_RENT", "MEDIAN_HOME_VALUE",
        "PCT_WHITE", "PCT_OWNER_OCCUPIED",
        "PCT_POVERTY", "PCT_BACHELORS",
        "TOTAL_POPULATION",
    ]
    df = df[keep]
    print(f"  Fetched {len(df):,} ZCTAs")

    # Cache for future runs
    df.to_parquet(CENSUS_CACHE, index=False)
    print(f"  Cached to {CENSUS_CACHE}")
    return df


def clean_zip(series):
    """Standardize ZIP codes to 5-digit strings."""
    return series.astype(str).str.strip().str[:5].str.zfill(5)


def merge_census(df_prop, df_census):
    """Left join property data with census data on ZIP_CODE."""
    df_prop["ZIP_CODE_CLEAN"]    = clean_zip(df_prop["ZIP_CODE"])
    df_census["ZIP_CODE_CLEAN"]  = clean_zip(df_census["ZIP_CODE"])

    before = len(df_prop)
    df_merged = df_prop.merge(
        df_census.drop(columns=["ZIP_CODE"]),
        on="ZIP_CODE_CLEAN",
        how="left"
    )
    df_merged = df_merged.drop(columns=["ZIP_CODE_CLEAN"])

    matched = df_merged["MEDIAN_INCOME"].notna().sum()
    print(f"  Properties before merge : {before:,}")
    print(f"  Properties after merge  : {len(df_merged):,}")
    print(f"  Matched with census data: {matched:,} ({matched/before:.1%})")
    print(f"  Unmatched               : {before - matched:,}")
    return df_merged


if __name__ == "__main__":

    # 1. Load property data
    print("Loading property data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Shape: {df.shape}")

    # 2. Fetch census data
    print("\nFetching census data...")
    df_census = fetch_census_data()
    print(f"  Census shape: {df_census.shape}")
    print(f"  Census columns: {df_census.columns.tolist()}")
    print(f"\nSample census data:\n{df_census.head(3)}")

    # 3. Merge
    print("\nMerging on ZIP_CODE...")
    df_final = merge_census(df, df_census)

    # 4. Check new columns
    new_cols = [c for c in df_final.columns if c not in df.columns]
    print(f"\nNew census columns added: {new_cols}")
    print(f"\nCensus feature stats:")
    print(df_final[new_cols].describe().round(2))

    # 5. Check nulls in new columns
    null_pct = df_final[new_cols].isnull().mean().round(3) * 100
    print(f"\nNull % in census columns:")
    print(null_pct)

    # 6. Save
    df_final.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Final shape: {df_final.shape}")
    print(f"\nAll columns ({len(df_final.columns)}):")
    for col in df_final.columns:
        print(f"  {col}")