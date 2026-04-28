import pandas as pd
import numpy as np
import requests
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
PLUTO_PATH     = "/mnt/c/Users/rodri/Downloads/Primary_Land_Use_Tax_Lot_Output__PLUTO_.csv"
OUTPUT_PATH    = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data_census.parquet"
CENSUS_CACHE   = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/census_tract.parquet"
PLUTO_CACHE    = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/pluto_slim.parquet"

CENSUS_API_KEY = "fda60e79b0da81a8ac6472ff4250f47daa8c527b"
ACS_YEAR       = 2022

NYC_COUNTIES = {
    "1": "061",  # Manhattan
    "2": "005",  # Bronx
    "3": "047",  # Brooklyn
    "4": "081",  # Queens
    "5": "085",  # Staten Island
}

# ── Helper: Robust Tract Logic ──────────────────────────────────────────────
def fix_tract_robust(val):
    """Standardizes PLUTO tract2010 to 6-digit Census format."""
    t = str(val).strip().replace(".0", "")
    if t in ("nan", "", "None", "0"): 
        return None
    
    if "." in t:
        parts = t.split(".")
        main = parts[0]
        dec = parts[1].ljust(2, '0')[:2]
        t = main + dec
    
    try:
        t_int = int(float(t))
    except: 
        return None

    if t_int <= 9999:
        return str(t_int * 100).zfill(6)
    
    return str(t_int).zfill(6)

# ── Load PLUTO ────────────────────────────────────────────────────────────────
def load_pluto(pluto_path, cache_path):
    if os.path.exists(cache_path):
        print(f"  Loading cached PLUTO from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Loading PLUTO from {pluto_path}...")
    pluto = pd.read_csv(
        pluto_path,
        usecols=["BBL", "tract2010", "borocode", "schooldist", "policeprct", "firecomp"],
        dtype=str,
        low_memory=False
    )

    pluto["BBL"] = pluto["BBL"].astype(str).str.strip().str.replace(".0", "", regex=False)
    pluto["borocode"] = pluto["borocode"].astype(str).str.strip().str.replace(".0", "", regex=False)
    pluto["county_fips"] = pluto["borocode"].map(NYC_COUNTIES)

    pluto["tract_fixed"] = pluto["tract2010"].apply(fix_tract_robust)
    
    # Create GEOID and GEOID_BASE
    pluto["GEOID"] = "36" + pluto["county_fips"] + pluto["tract_fixed"]
    pluto["GEOID_BASE"] = pluto["GEOID"].str[:9] + "00"

    pluto = pluto.drop_duplicates(subset="BBL")
    pluto.to_parquet(cache_path, index=False)
    return pluto

# ── Fetch ACS tract-level data ────────────────────────────────────────────────
def fetch_census_tract(year=ACS_YEAR, api_key=None):
    if os.path.exists(CENSUS_CACHE):
        print(f"  Loading cached census data from {CENSUS_CACHE}")
        return pd.read_parquet(CENSUS_CACHE)

    variables = {
        "B19013_001E": "MEDIAN_INCOME",
        "B25064_001E": "MEDIAN_RENT",
        "B25077_001E": "MEDIAN_HOME_VALUE",
        "B03002_003E": "POP_WHITE_ALONE",
        "B03002_001E": "POP_TOTAL_RACE",
        "B25003_002E": "OWNER_OCCUPIED",
        "B25003_001E": "TOTAL_OCCUPIED",
        "B17001_002E": "BELOW_POVERTY",
        "B01003_001E": "TOTAL_POPULATION",
        "B15003_022E": "BACHELORS_DEGREE",
        "B15003_001E": "POP_25_PLUS",
    }
    
    vars_str = ",".join(variables.keys())
    all_tracts = []

    for boro, county in NYC_COUNTIES.items():
        print(f"  Fetching tracts for county {county}...")
        url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,{vars_str}&for=tract:*&in=state:36%26in=county:{county}"
        if api_key: url += f"&key={api_key}"

        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            data = r.json()
            all_tracts.append(pd.DataFrame(data[1:], columns=data[0]))

    df = pd.concat(all_tracts, ignore_index=True)
    df["GEOID"] = "36" + df["county"].str.zfill(3) + df["tract"].str.zfill(6)
    df = df.rename(columns=variables)

    num_cols = list(variables.values())
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([-666666666, -666666666.0], np.nan)

    df["PCT_WHITE"] = (df["POP_WHITE_ALONE"] / df["POP_TOTAL_RACE"].clip(lower=1) * 100).round(2)
    df["PCT_OWNER_OCCUPIED"] = (df["OWNER_OCCUPIED"] / df["TOTAL_OCCUPIED"].clip(lower=1) * 100).round(2)
    df["PCT_POVERTY"] = (df["BELOW_POVERTY"] / df["TOTAL_POPULATION"].clip(lower=1) * 100).round(2)
    df["PCT_BACHELORS"] = (df["BACHELORS_DEGREE"] / df["POP_25_PLUS"].clip(lower=1) * 100).round(2)

    keep = ["GEOID", "MEDIAN_INCOME", "MEDIAN_RENT", "MEDIAN_HOME_VALUE", 
            "PCT_WHITE", "PCT_OWNER_OCCUPIED", "PCT_POVERTY", "PCT_BACHELORS", "TOTAL_POPULATION"]
    df = df[keep]
    df.to_parquet(CENSUS_CACHE, index=False)
    return df

# ── Aggregate to Base Tract ───────────────────────────────────────────────────
def aggregate_to_base_tract(df_census):
    """Groups census sub-tracts into base tracts."""
    df_census = df_census.copy()
    df_census["GEOID_BASE"] = df_census["GEOID"].str[:9] + "00"
    
    agg = df_census.groupby("GEOID_BASE").agg({
        "MEDIAN_INCOME": "median",
        "MEDIAN_RENT": "median",
        "MEDIAN_HOME_VALUE": "median",
        "PCT_WHITE": "mean",
        "PCT_OWNER_OCCUPIED": "mean",
        "PCT_POVERTY": "mean",
        "PCT_BACHELORS": "mean",
        "TOTAL_POPULATION": "sum"
    }).reset_index()
    return agg

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Step 1: Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    pluto = load_pluto(PLUTO_PATH, PLUTO_CACHE)
    df_census = fetch_census_tract(api_key=CENSUS_API_KEY)
    df_census_base = aggregate_to_base_tract(df_census)

    print("\nStep 2: Joining PLUTO to properties...")
    df["BBL"] = df["BBL"].astype(str).str.strip()
    pluto_slim = pluto[["BBL", "GEOID", "GEOID_BASE", "schooldist", "policeprct", "firecomp"]].copy()
    df = df.merge(pluto_slim, on="BBL", how="left")

    print("Step 3: Two-Step Census Join (Exact -> Base Fallback)...")
    # A. Exact Match Join
    df = df.merge(df_census, on="GEOID", how="left")
    
    exact_match_count = df["MEDIAN_INCOME"].notna().sum()
    print(f"  Matches after Exact Join: {exact_match_count:,}")

    # B. Fallback Join
    census_cols = [c for c in df_census.columns if c != "GEOID"]
    
    # Rename columns in base df to avoid collision, but keep GEOID_BASE for the join
    df_base = df_census_base.copy()
    rename_dict = {c: c + "_base" for c in census_cols}
    df_base = df_base.rename(columns=rename_dict)
    
    # Merge on GEOID_BASE
    df = df.merge(df_base, on="GEOID_BASE", how="left")
    
    # Fill only the NaNs
    for col in census_cols:
        df[col] = df[col].fillna(df[col + "_base"])
        df.drop(columns=[col + "_base"], inplace=True)

    # Step 4: Cleanup & District Conversion
    for col in ["schooldist", "policeprct", "firecomp"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(".0", "", regex=False), errors="coerce")

    # Final Log-Transforms
    for col in ["MEDIAN_INCOME", "MEDIAN_RENT", "MEDIAN_HOME_VALUE"]:
        df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))

    print(f"\nFinal Match Statistics:")
    matched = df["MEDIAN_INCOME"].notna().sum()
    print(f"  Properties with Census Data: {matched:,} / {len(df):,}")
    print(f"  Coverage: {matched/len(df):.1%}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")