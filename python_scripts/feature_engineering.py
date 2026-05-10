"""
feature_engineering.py
=======================
Shared feature engineering pipeline for NYC property tax models.
Import and call engineer_features() from any model script.

Usage:
    from feature_engineering import load_data, engineer_features, prepare_xy
"""

import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder

HISTORICAL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

DEFAULT_SALES_PATH = (
    "/home/rodrigofrancachaves/project-nyc_property_taxes/data/sales_clean.parquet"
)


# ── Sales data loader & feature builder ───────────────────────────────────────
def load_sales_features(
    df: pd.DataFrame,
    sales_path: str = DEFAULT_SALES_PATH,
    cutoff_year: int = 2025,       # only use sales strictly before FY2026
    min_price: int = 50_000,       # drop likely non-arm's-length transfers
) -> pd.DataFrame:
    """
    Join sales_clean.parquet onto df (which must have a BBL column) and return
    a DataFrame of per-property sale features indexed like df.

    Features produced
    -----------------
    HAS_RECENT_SALE         : 1 if any qualifying sale exists, else 0
    LAST_SALE_PRICE         : most recent qualifying sale price (0 if none)
    LAST_SALE_YEAR          : year of most recent sale (0 if none)
    YEARS_SINCE_SALE        : 2026 - LAST_SALE_YEAR  (26 if never sold)
    SALE_PRICE_PER_SQFT     : LAST_SALE_PRICE / GROSS_SQFT (0 if missing)
    SALE_TO_ASSESS_RATIO    : LAST_SALE_PRICE / FINACTTOT_FY2025 (0 if missing)
    LOG_LAST_SALE_PRICE     : log1p of LAST_SALE_PRICE
    N_SALES                 : number of qualifying sales in the window
    MAX_SALE_PRICE          : highest qualifying sale price
    MEDIAN_SALE_PRICE       : median qualifying sale price
    SALE_PRICE_TREND        : slope of sale prices over time (0 if <2 sales)
    ZIP_MEDIAN_SALE_PRICE   : median sale price per sqft in the same zip
    ZIP_SALE_VOLUME         : number of sales in the same zip (market activity)
    SALE_VS_ZIP_MEDIAN      : SALE_PRICE_PER_SQFT / ZIP_MEDIAN_SALE_PRICE
    """
    import os
    if not os.path.exists(sales_path):
        print(f"  WARNING: sales file not found at {sales_path} — skipping sales features")
        return pd.DataFrame(index=df.index)

    print(f"  Loading sales data from {sales_path}...")
    sales = pd.read_parquet(sales_path)

    # Filter to arm's-length, pre-cutoff sales only
    sales = sales[
        (sales["SALE PRICE"] >= min_price) &
        (sales["SALE_YEAR"]  <= cutoff_year)
    ].copy()

    # Normalise BBL to string for joining
    sales["BBL"] = sales["BBL"].astype(str).str.strip()
    df_bbl       = df["BBL"].astype(str).str.strip() if "BBL" in df.columns else None

    if df_bbl is None:
        print("  WARNING: BBL column not found in df — skipping sales features")
        return pd.DataFrame(index=df.index)

    print(f"  Sales after filtering: {len(sales):,}")

    # ── Per-property aggregates ───────────────────────────────────────────────
    grp = sales.groupby("BBL")

    agg = grp["SALE PRICE"].agg(
        N_SALES        = "count",
        LAST_SALE_PRICE= "last",     # files are sorted by year already
        MAX_SALE_PRICE = "max",
        MEDIAN_SALE_PRICE = "median",
    ).reset_index()

    last_year = grp["SALE_YEAR"].last().reset_index().rename(
        columns={"SALE_YEAR": "LAST_SALE_YEAR"}
    )
    agg = agg.merge(last_year, on="BBL", how="left")

    # Sale price trend: vectorized OLS slope (avoids memory-hungry groupby.apply)
    # For each BBL: slope = cov(year, price) / var(year)
    s = sales[["BBL", "SALE_YEAR", "SALE PRICE"]].copy()
    s["SALE_YEAR"]   = s["SALE_YEAR"].astype(float)
    s["SALE PRICE"]  = s["SALE PRICE"].astype(float)
    grp_s            = s.groupby("BBL")
    mean_year  = grp_s["SALE_YEAR"].transform("mean")
    mean_price = grp_s["SALE PRICE"].transform("mean")
    s["xc"] = s["SALE_YEAR"]  - mean_year
    s["yc"] = s["SALE PRICE"] - mean_price
    numer  = s.groupby("BBL").apply(lambda g: (g["xc"] * g["yc"]).sum())
    denom  = s.groupby("BBL").apply(lambda g: (g["xc"] ** 2).sum())
    slopes_s = (numer / denom.replace(0, np.nan)).fillna(0).reset_index()
    slopes_s.columns = ["BBL", "SALE_PRICE_TREND"]
    # Zero out BBLs with only one sale (slope is meaningless)
    single_sale = grp_s["SALE_YEAR"].count()[grp_s["SALE_YEAR"].count() < 2].index
    slopes_s.loc[slopes_s["BBL"].isin(single_sale), "SALE_PRICE_TREND"] = 0.0
    agg = agg.merge(slopes_s, on="BBL", how="left")
    del s, numer, denom, slopes_s, single_sale
    gc.collect()

    # ── ZIP-level aggregates (market context) ─────────────────────────────────
    # Use gross sqft from sales file when available
    sales["PRICE_PER_SQFT_RAW"] = (
        sales["SALE PRICE"] /
        pd.to_numeric(sales["GROSS SQUARE FEET"], errors="coerce").clip(lower=1)
    )
    # Only use rows where sqft is plausible
    sales_with_sqft = sales[
        pd.to_numeric(sales["GROSS SQUARE FEET"], errors="coerce").fillna(0) > 100
    ]

    if "ZIP CODE" in sales.columns:
        zip_stats = (
            sales_with_sqft.groupby("ZIP CODE")["PRICE_PER_SQFT_RAW"]
            .agg(ZIP_MEDIAN_SALE_PRICE="median", ZIP_SALE_VOLUME="count")
            .reset_index()
            .rename(columns={"ZIP CODE": "ZIP_CODE_SALES"})
        )
    else:
        zip_stats = None

    # ── Join onto df ──────────────────────────────────────────────────────────
    result = pd.DataFrame({"BBL": df_bbl.values}, index=df.index)
    result = result.merge(agg, on="BBL", how="left")

    # Derive clean features
    feat = pd.DataFrame(index=df.index)
    feat["HAS_RECENT_SALE"]   = result["N_SALES"].notna().astype(int)
    feat["N_SALES"]           = result["N_SALES"].fillna(0).astype(int)
    feat["LAST_SALE_PRICE"]   = result["LAST_SALE_PRICE"].fillna(0)
    feat["LOG_LAST_SALE_PRICE"] = np.log1p(feat["LAST_SALE_PRICE"])
    feat["LAST_SALE_YEAR"]    = result["LAST_SALE_YEAR"].fillna(0).astype(int)
    feat["YEARS_SINCE_SALE"]  = np.where(
        feat["LAST_SALE_YEAR"] > 0,
        2026 - feat["LAST_SALE_YEAR"],
        26,                          # sentinel: never sold in our window
    )
    feat["MAX_SALE_PRICE"]    = result["MAX_SALE_PRICE"].fillna(0)
    feat["MEDIAN_SALE_PRICE"] = result["MEDIAN_SALE_PRICE"].fillna(0)
    feat["SALE_PRICE_TREND"]  = result["SALE_PRICE_TREND"].fillna(0)

    # Price per sqft — use df's GROSS_SQFT for consistency
    gross_sqft = pd.to_numeric(df["GROSS_SQFT"], errors="coerce").fillna(0).clip(lower=1)
    feat["SALE_PRICE_PER_SQFT"] = (feat["LAST_SALE_PRICE"] / gross_sqft).clip(upper=50_000)

    # Sale-to-assessment ratio — most direct overvaluation signal
    if "FINACTTOT_FY2025" in df.columns:
        assess = pd.to_numeric(df["FINACTTOT_FY2025"], errors="coerce").fillna(0).clip(lower=1)
        feat["SALE_TO_ASSESS_RATIO"] = (
            feat["LAST_SALE_PRICE"] / assess
        ).clip(0, 50)
        feat["LOG_SALE_TO_ASSESS"] = np.log1p(feat["SALE_TO_ASSESS_RATIO"])

    # ZIP-level join
    if zip_stats is not None and "ZIP_CODE" in df.columns:
        zip_col = df["ZIP_CODE"].astype(str).str.strip()
        zip_map_median = zip_stats.set_index("ZIP_CODE_SALES")["ZIP_MEDIAN_SALE_PRICE"]
        zip_map_volume = zip_stats.set_index("ZIP_CODE_SALES")["ZIP_SALE_VOLUME"]
        feat["ZIP_MEDIAN_SALE_PRICE"] = zip_col.map(zip_map_median).fillna(0)
        feat["ZIP_SALE_VOLUME"]       = zip_col.map(zip_map_volume).fillna(0)
        feat["SALE_VS_ZIP_MEDIAN"] = (
            feat["SALE_PRICE_PER_SQFT"] /
            feat["ZIP_MEDIAN_SALE_PRICE"].clip(lower=1)
        ).clip(0, 10)
        # For unsold properties: still give them neighborhood context
        feat.loc[feat["HAS_RECENT_SALE"] == 0, "SALE_VS_ZIP_MEDIAN"] = np.nan

    matched = feat["HAS_RECENT_SALE"].sum()
    print(f"  BBL match rate: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")
    return feat


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(path: str, drop_unknowns: bool = True) -> pd.DataFrame:
    """Load processed parquet and optionally drop rows where target is 'unknown'."""
    print("Loading data...")
    df = pd.read_parquet(path)
    print(f"  Loaded shape: {df.shape}")
    if drop_unknowns and "target_2026" in df.columns:
        df = df[df["target_2026"] != "unknown"].copy()
        print(f"  Shape after dropping unknowns: {df.shape}")
    print(f"\nTarget distribution:\n{df['target_2026'].value_counts()}")
    print(f"\nTarget proportions:\n{df['target_2026'].value_counts(normalize=True).round(3)}")
    return df


# ── OLS projection ────────────────────────────────────────────────────────────
def project_next_year(
    df: pd.DataFrame,
    col_list: list[str],
    series_name: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Vectorized per-property OLS trend extrapolation over col_list (consecutive years).
    Returns three leak-free Series for FY2026:
      PROJ_<series_name>_FY2026       – projected dollar value
      PROJ_RATIO_<series_name>_FY2026 – projected / last known (momentum ratio)
      PROJ_RESID_<series_name>_FY2026 – projected − last known (dollar gap)
    """
    n     = len(col_list)
    x     = np.arange(n, dtype=np.float64)
    x_c   = x - x.mean()
    denom = float(x_c @ x_c)

    Y          = df[col_list].fillna(0).to_numpy(dtype=np.float64)  # (N, n)
    slopes     = (x_c @ Y.T) / denom
    intercepts = Y.mean(axis=1)

    x_next    = n - x.mean()
    projected = np.clip(intercepts + slopes * x_next, 0, None)
    last_known = Y[:, -1]

    ratio = np.clip(np.where(last_known > 0, projected / last_known, 1.0), 0, 5)
    resid = (projected - last_known).clip(-1e7, 1e7)

    idx = df.index
    return (
        pd.Series(projected, index=idx, name=f"PROJ_{series_name}_FY2026"),
        pd.Series(ratio,     index=idx, name=f"PROJ_RATIO_{series_name}_FY2026"),
        pd.Series(resid,     index=idx, name=f"PROJ_RESID_{series_name}_FY2026"),
    )


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(
    df: pd.DataFrame,
    historical_years: list[int] = HISTORICAL_YEARS,
    sales_path: str | None = DEFAULT_SALES_PATH,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Build all engineered features.

    Parameters
    ----------
    df              : property DataFrame with BBL column for sales join
    historical_years: fiscal years to build time-series features over
    sales_path      : path to sales_clean.parquet; pass None to skip

    Returns
    -------
    df          : DataFrame with new columns appended
    features    : ordered list of feature column names for X
    le_dict     : dict of {original_col: LabelEncoder} for categoricals
    """
    print("\nEngineering features...")

    # ── Numeric coercions ─────────────────────────────────────────────────────
    base_numeric = [
        "GROSS_SQFT", "LAND_AREA", "NUM_BLDGS", "YRBUILT",
        "UNITS", "COOP_APTS", "BLD_STORY", "LOT_FRT", "LOT_DEP",
        "FINACTTOT", "FINACTLAND", "FINMKTTOT", "PYACTTOT",
    ]
    hist_numeric = (
        [f"FINACTTOT_FY{y}"  for y in historical_years] +
        [f"FINACTLAND_FY{y}" for y in historical_years] +
        [f"FINMKTTOT_FY{y}"  for y in historical_years]
    )
    for col in base_numeric + hist_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    new_cols: dict = {}

    # ── Property structure features ───────────────────────────────────────────
    new_cols["BUILDING_AGE"]   = (2026 - df["YRBUILT"]).clip(lower=0, upper=200)
    new_cols["LOG_GROSS_SQFT"] = np.log1p(df["GROSS_SQFT"].fillna(0))
    new_cols["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"].fillna(0))
    new_cols["LOG_PYACTTOT"]   = np.log1p(df["PYACTTOT"].fillna(0))
    new_cols["SQFT_PER_UNIT"]  = (df["GROSS_SQFT"] / df["UNITS"].clip(lower=1)).clip(upper=50_000)
    new_cols["COVERAGE_RATIO"] = (df["GROSS_SQFT"] / df["LAND_AREA"].clip(lower=1)).clip(upper=50)
    new_cols["LOT_AREA"]       = df["LOT_FRT"] * df["LOT_DEP"]
    new_cols["BUILDING_ERA"]   = pd.cut(
        df["YRBUILT"],
        bins=[0, 1900, 1940, 1960, 1980, 2000, 2010, 2030],
        labels=[1, 2, 3, 4, 5, 6, 7],
    ).astype(float).fillna(0)

    # ── Assessment snapshot features (FY2025 — no leakage) ───────────────────
    assess_per_sqft                  = df["FINACTTOT_FY2025"].fillna(0) / df["GROSS_SQFT"].clip(lower=1)
    new_cols["ASSESS_PER_SQFT"]      = assess_per_sqft
    new_cols["LOG_ASSESS_PER_SQFT"]  = np.log1p(assess_per_sqft)
    new_cols["LAND_TO_TOTAL"]        = (df["FINACTLAND_FY2025"].fillna(0) / df["FINACTTOT_FY2025"].clip(lower=1)).clip(0, 1)
    mkt_to_assess                    = (df["FINMKTTOT_FY2025"].fillna(0)  / df["FINACTTOT_FY2025"].clip(lower=1)).clip(0, 20)
    new_cols["MKT_TO_ASSESS"]        = mkt_to_assess
    new_cols["LOG_MKT_TO_ASSESS"]    = np.log1p(mkt_to_assess)

    # ── ZIP_CODE neighborhood aggregates (replaces raw label encoding signal) ──
    # Computed on the full df before train/test split — use only FY2025 values
    # to avoid leakage from the target year
    if "ZIP_CODE" in df.columns and "FINACTTOT_FY2025" in df.columns:
        zip_assess = df.groupby("ZIP_CODE")["FINACTTOT_FY2025"].agg(["mean", "median", "std"])
        zip_assess.columns = ["ZIP_MEAN_ASSESS", "ZIP_MEDIAN_ASSESS", "ZIP_ASSESS_STD"]
        zip_assess["ZIP_ASSESS_STD"] = zip_assess["ZIP_ASSESS_STD"].fillna(0)

        new_cols["ZIP_MEAN_ASSESS"]   = df["ZIP_CODE"].map(zip_assess["ZIP_MEAN_ASSESS"])
        new_cols["ZIP_MEDIAN_ASSESS"] = df["ZIP_CODE"].map(zip_assess["ZIP_MEDIAN_ASSESS"])
        new_cols["ZIP_ASSESS_STD"]    = df["ZIP_CODE"].map(zip_assess["ZIP_ASSESS_STD"])

        # How does this property's assessment compare to its zip's median?
        # >1 = assessed above neighborhood median → overvaluation signal
        new_cols["ASSESS_VS_ZIP_MEDIAN"] = (
            df["FINACTTOT_FY2025"].fillna(0) /
            df["ZIP_CODE"].map(zip_assess["ZIP_MEDIAN_ASSESS"]).clip(lower=1)
        ).clip(0, 10)

    # ── BLDG_CLASS aggregates (building class drives NYC tax treatment) ────────
    if "BLDG_CLASS" in df.columns and "FINACTTOT_FY2025" in df.columns:
        cls_assess = df.groupby("BLDG_CLASS")["FINACTTOT_FY2025"].agg(["mean", "median"])
        cls_assess.columns = ["BLDG_CLASS_MEAN_ASSESS", "BLDG_CLASS_MEDIAN_ASSESS"]

        new_cols["BLDG_CLASS_MEAN_ASSESS"]   = df["BLDG_CLASS"].map(cls_assess["BLDG_CLASS_MEAN_ASSESS"])
        new_cols["BLDG_CLASS_MEDIAN_ASSESS"] = df["BLDG_CLASS"].map(cls_assess["BLDG_CLASS_MEDIAN_ASSESS"])

        # How does this property compare to others of the same building class?
        new_cols["ASSESS_VS_BLDG_CLASS_MEDIAN"] = (
            df["FINACTTOT_FY2025"].fillna(0) /
            df["BLDG_CLASS"].map(cls_assess["BLDG_CLASS_MEDIAN_ASSESS"]).clip(lower=1)
        ).clip(0, 10)

    # ── Historical column lists ───────────────────────────────────────────────
    finacttot_cols  = [c for c in [f"FINACTTOT_FY{y}"  for y in historical_years] if c in df.columns]
    finactland_cols = [c for c in [f"FINACTLAND_FY{y}" for y in historical_years] if c in df.columns]
    finmkttot_cols  = [c for c in [f"FINMKTTOT_FY{y}"  for y in historical_years] if c in df.columns]

    # ── Log transforms of historical values ──────────────────────────────────
    log_acttot_cols, log_actland_cols, log_mkttot_cols = [], [], []
    for col in finacttot_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_acttot_cols.append(name)
    for col in finactland_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_actland_cols.append(name)
    for col in finmkttot_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_mkttot_cols.append(name)

    # ── YoY % change ─────────────────────────────────────────────────────────
    def yoy_changes(cols: list[str], prefix: str) -> list[str]:
        names = []
        for i in range(1, len(cols)):
            yr   = historical_years[i]
            name = f"{prefix}_FY{yr}"
            new_cols[name] = (
                (df[cols[i]].fillna(0) - df[cols[i - 1]].fillna(0)) /
                df[cols[i - 1]].fillna(1).clip(lower=1)
            ).clip(-1, 5)
            names.append(name)
        return names

    yoy_cols      = yoy_changes(finacttot_cols,  "ASSESS_YOY")
    yoy_land_cols = yoy_changes(finactland_cols, "LAND_YOY")
    yoy_mkt_cols  = yoy_changes(finmkttot_cols,  "MKT_YOY")

    # ── Trend & volatility ────────────────────────────────────────────────────
    if yoy_cols:
        new_cols["ASSESS_VOLATILITY"] = (
            pd.DataFrame({k: new_cols[k] for k in yoy_cols}).std(axis=1).fillna(0)
        )
    else:
        new_cols["ASSESS_VOLATILITY"] = 0.0

    if len(finacttot_cols) >= 2:
        new_cols["ASSESS_TREND"] = (
            (df[finacttot_cols[-1]].fillna(0) - df[finacttot_cols[0]].fillna(0)) /
            df[finacttot_cols[0]].fillna(1).clip(lower=1)
        ).clip(-1, 10)
    else:
        new_cols["ASSESS_TREND"] = 0.0

    # ── OLS projections to FY2026 ─────────────────────────────────────────────
    proj_feature_names: list[str] = []
    for col_list, series_name in [
        (finacttot_cols,  "FINACTTOT"),
        (finactland_cols, "FINACTLAND"),
        (finmkttot_cols,  "FINMKTTOT"),
    ]:
        if len(col_list) >= 2:
            proj, ratio, resid = project_next_year(df, col_list, series_name)
            new_cols[proj.name]  = proj
            new_cols[ratio.name] = ratio
            new_cols[resid.name] = resid
            proj_feature_names  += [proj.name, ratio.name, resid.name]

    # ── NEW: Market vs assessed gap per year ──────────────────────────────────
    # If market grows faster than assessed → likely undervalued signal
    gap_cols = []
    for i in range(1, len(historical_years)):
        yr = historical_years[i]
        mkt_yoy_col  = f"MKT_YOY_FY{yr}"
        act_yoy_col  = f"ASSESS_YOY_FY{yr}"
        if mkt_yoy_col in new_cols and act_yoy_col in new_cols:
            name = f"MKT_ASSESS_GAP_YOY_FY{yr}"
            new_cols[name] = (
                pd.Series(new_cols[mkt_yoy_col], index=df.index) -
                pd.Series(new_cols[act_yoy_col], index=df.index)
            ).clip(-5, 5)
            gap_cols.append(name)

    # ── NEW: Cumulative growth from 2020 baseline ─────────────────────────────
    cumul_cols = []
    if finacttot_cols:
        base_col = finacttot_cols[0]
        for col in finacttot_cols[1:]:
            yr   = col.split("FY")[1]
            name = f"CUMUL_GROWTH_FY{yr}"
            new_cols[name] = (
                (df[col].fillna(0) - df[base_col].fillna(0)) /
                df[base_col].fillna(1).clip(lower=1)
            ).clip(-1, 10)
            cumul_cols.append(name)

    cumul_mkt_cols = []
    if finmkttot_cols:
        base_mkt = finmkttot_cols[0]
        for col in finmkttot_cols[1:]:
            yr   = col.split("FY")[1]
            name = f"CUMUL_MKT_GROWTH_FY{yr}"
            new_cols[name] = (
                (df[col].fillna(0) - df[base_mkt].fillna(0)) /
                df[base_mkt].fillna(1).clip(lower=1)
            ).clip(-1, 10)
            cumul_mkt_cols.append(name)

    # ── NEW: Assessment acceleration (2nd derivative) ─────────────────────────
    accel_cols = []
    if len(yoy_cols) >= 2:
        for i in range(1, len(yoy_cols)):
            yr   = historical_years[i + 1]
            name = f"ASSESS_ACCEL_FY{yr}"
            new_cols[name] = (
                pd.Series(new_cols[yoy_cols[i]], index=df.index) -
                pd.Series(new_cols[yoy_cols[i - 1]], index=df.index)
            ).clip(-5, 5)
            accel_cols.append(name)

    # ── NEW: Assessed per sqft per year (trajectory of assessment intensity) ──
    psqft_cols = []
    for col in finacttot_cols:
        yr   = col.split("FY")[1]
        name = f"ASSESS_PER_SQFT_FY{yr}"
        new_cols[name] = (df[col].fillna(0) / df["GROSS_SQFT"].clip(lower=1))
        psqft_cols.append(name)

    # ── NEW: Land ratio per year ──────────────────────────────────────────────
    land_ratio_cols = []
    for yr in historical_years:
        act_col  = f"FINACTTOT_FY{yr}"
        land_col = f"FINACTLAND_FY{yr}"
        if act_col in df.columns and land_col in df.columns:
            name = f"LAND_RATIO_FY{yr}"
            new_cols[name] = (
                df[land_col].fillna(0) /
                df[act_col].fillna(1).clip(lower=1)
            ).clip(0, 1)
            land_ratio_cols.append(name)

    # ── NEW: Consistent classification score ──────────────────────────────────
    # How many years in a row was the property over/under/fairly valued?
    # This is a very strong signal — persistent patterns predict future patterns
    over_cols  = [c for c in df.columns if "overvalued_"    in c and any(str(y) in c for y in historical_years)]
    under_cols = [c for c in df.columns if "undervalued_"   in c and any(str(y) in c for y in historical_years)]
    fair_cols  = [c for c in df.columns if "fairly_valued_" in c and any(str(y) in c for y in historical_years)]

    if over_cols:
        new_cols["CONSISTENT_OVERVALUED"]  = df[over_cols].sum(axis=1)
        new_cols["CONSISTENT_UNDERVALUED"] = df[under_cols].sum(axis=1)
        new_cols["CONSISTENT_FAIR"]        = df[fair_cols].sum(axis=1)
        # Dominant class over history
        new_cols["DOMINANT_CLASS"] = (
            pd.concat([
                df[over_cols].sum(axis=1).rename("over"),
                df[under_cols].sum(axis=1).rename("under"),
                df[fair_cols].sum(axis=1).rename("fair"),
            ], axis=1)
        ).idxmax(axis=1).map({"over": 0, "under": 1, "fair": 2}).fillna(2)

    # ── NEW: NYC cap flag ─────────────────────────────────────────────────────
    # NYC caps Class 1 annual increases at ~6%; if YoY is near 6%, likely capped
    if yoy_cols:
        last_yoy = yoy_cols[-1]
        new_cols["ASSESS_AT_CAP"] = pd.Series(new_cols[last_yoy], index=df.index).between(0.04, 0.07).astype(int)
    else:
        new_cols["ASSESS_AT_CAP"] = 0

    # ── NEW: Market trend ─────────────────────────────────────────────────────
    if len(finmkttot_cols) >= 2:
        new_cols["MKT_TREND"] = (
            (df[finmkttot_cols[-1]].fillna(0) - df[finmkttot_cols[0]].fillna(0)) /
            df[finmkttot_cols[0]].fillna(1).clip(lower=1)
        ).clip(-1, 10)
    else:
        new_cols["MKT_TREND"] = 0.0

    # ── NEW: Spread between market trend and assess trend ─────────────────────
    # Large positive = market outpacing assessment = likely undervalued
    if "MKT_TREND" in new_cols and "ASSESS_TREND" in new_cols:
        new_cols["MKT_VS_ASSESS_TREND_SPREAD"] = (
            pd.Series(new_cols["MKT_TREND"],    index=df.index) -
            pd.Series(new_cols["ASSESS_TREND"], index=df.index)
        ).clip(-10, 10)

    # ── Interaction: building age × assessment intensity ─────────────────────
    # Older buildings in high-assessment areas behave differently from
    # older buildings in low-assessment areas — captures this joint effect
    if "BUILDING_AGE" in new_cols and "ASSESS_PER_SQFT" in new_cols:
        new_cols["AGE_X_ASSESS_PER_SQFT"] = (
            pd.Series(new_cols["BUILDING_AGE"],  index=df.index) *
            pd.Series(new_cols["ASSESS_PER_SQFT"], index=df.index)
        ).clip(upper=1e6)

    # ── Categorical encoding ──────────────────────────────────────────────────
    categorical_cols = ["BORO", "BLDG_CLASS", "ZIP_CODE", "ZONING"]
    le_dict: dict = {}
    encoded_cat_cols: list[str] = []
    for col in categorical_cols:
        if col in df.columns:
            le   = LabelEncoder()
            name = f"{col}_CODE"
            new_cols[name] = le.fit_transform(df[col].fillna("Unknown").astype(str))
            le_dict[col]   = le
            encoded_cat_cols.append(name)

    # ── Historical status flags ───────────────────────────────────────────────
    historical_status_cols = [
        c for c in df.columns
        if any(str(yr) in c for yr in historical_years)
        and any(x in c for x in ["overvalued", "undervalued", "fairly_valued"])
    ]

    # ── Sales data join ───────────────────────────────────────────────────────
    sales_feat_cols: list[str] = []
    if sales_path is not None:
        sales_feats = load_sales_features(df, sales_path=sales_path)
        if not sales_feats.empty:
            for col in sales_feats.columns:
                new_cols[col] = sales_feats[col].values
                sales_feat_cols.append(col)

    # ── Concat all new columns at once (avoids DataFrame fragmentation) ───────
    new_df = pd.DataFrame(new_cols, index=df.index)
    df     = pd.concat([df, new_df], axis=1)
    del new_cols, new_df
    gc.collect()

    # ── Final feature list ────────────────────────────────────────────────────
    scalar_extras = []
    for name in [
        "CONSISTENT_OVERVALUED", "CONSISTENT_UNDERVALUED", "CONSISTENT_FAIR",
        "DOMINANT_CLASS", "ASSESS_AT_CAP", "MKT_TREND", "MKT_VS_ASSESS_TREND_SPREAD",
        # ZIP neighborhood aggregates
        "ZIP_MEAN_ASSESS", "ZIP_MEDIAN_ASSESS", "ZIP_ASSESS_STD", "ASSESS_VS_ZIP_MEDIAN",
        # Building class aggregates
        "BLDG_CLASS_MEAN_ASSESS", "BLDG_CLASS_MEDIAN_ASSESS", "ASSESS_VS_BLDG_CLASS_MEDIAN",
        # Interaction
        "AGE_X_ASSESS_PER_SQFT",
    ]:
        if name in df.columns:
            scalar_extras.append(name)
    # Sales features (dynamic — whatever load_sales_features produced)
    sales_feat_cols = [c for c in sales_feat_cols if c in df.columns]

    features = (
        encoded_cat_cols +
        [
            # Size & structure
            "LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS", "UNITS",
            "COOP_APTS", "BLD_STORY", "LOT_AREA",
            "SQFT_PER_UNIT", "COVERAGE_RATIO",
            # Age
            "BUILDING_AGE", "BUILDING_ERA",
            # Prior year total assessment
            "LOG_PYACTTOT",
            # Assessment ratios (FY2025-based)
            "ASSESS_PER_SQFT", "LOG_ASSESS_PER_SQFT",
            "LAND_TO_TOTAL", "MKT_TO_ASSESS", "LOG_MKT_TO_ASSESS",
            # Trend & volatility over FY2020-2025
            "ASSESS_TREND", "ASSESS_VOLATILITY",
        ] +
        scalar_extras +                # consistency scores, cap flag, mkt trend
        proj_feature_names +           # OLS-projected FY2026 values
        historical_status_cols +       # binary labels per year
        log_acttot_cols +              # log assessed total per year
        log_actland_cols +             # log assessed land per year
        log_mkttot_cols +              # log market total per year
        yoy_cols +                     # YoY % change in assessed total
        yoy_land_cols +                # YoY % change in assessed land
        yoy_mkt_cols +                 # YoY % change in market total
        gap_cols +                     # market vs assessed gap per year
        cumul_cols +                   # cumulative assessed growth from 2020
        cumul_mkt_cols +               # cumulative market growth from 2020
        accel_cols +                   # assessment acceleration
        psqft_cols +                   # assessed per sqft per year
        land_ratio_cols +              # land/total ratio per year
        sales_feat_cols                # actual transaction price features
    )
    features = [f for f in features if f in df.columns]

    print(f"  Total features built: {len(features)}")
    return df, features, le_dict


# ── Prepare X / y ─────────────────────────────────────────────────────────────
def prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "target_2026",
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y, coercing all X columns to float."""
    X = df[features].copy()
    y = df[target_col].astype(str)
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X, y


# ── Stratified subsampling ────────────────────────────────────────────────────
def subsample(X, y, n: int, seed: int = 42):
    """
    Stratified subsample of size n from (X, y).
    Works with both DataFrames (returns DataFrame) and numpy arrays.
    """
    y_arr    = np.asarray(y)
    if len(y_arr) <= n:
        return X, y_arr

    rng              = np.random.default_rng(seed)
    classes, counts  = np.unique(y_arr, return_counts=True)
    fracs            = counts / counts.sum()
    idx = np.concatenate([
        rng.choice(
            np.where(y_arr == cls)[0],
            size=int(np.ceil(frac * n)),
            replace=False,
        )
        for cls, frac in zip(classes, fracs)
    ])
    rng.shuffle(idx)
    idx = idx[:n]

    if isinstance(X, pd.DataFrame):
        return X.iloc[idx], y_arr[idx]
    return np.asarray(X)[idx], y_arr[idx]