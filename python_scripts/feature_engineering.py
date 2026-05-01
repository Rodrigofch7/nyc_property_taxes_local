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
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Build all engineered features.

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

    # ── Concat all new columns at once (avoids DataFrame fragmentation) ───────
    new_df = pd.DataFrame(new_cols, index=df.index)
    df     = pd.concat([df, new_df], axis=1)
    del new_cols, new_df
    gc.collect()

    # ── Final feature list ────────────────────────────────────────────────────
    # All features below are derived only from structural data or FY2020-2025
    # historical values. No FY2026 actuals are used anywhere — no leakage.
    features = (
        encoded_cat_cols +
        [
            # Size & structure
            "LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS", "UNITS",
            "COOP_APTS", "BLD_STORY", "LOT_AREA",
            # Previously missing — safe, derived from GROSS_SQFT/UNITS/LAND_AREA
            "SQFT_PER_UNIT", "COVERAGE_RATIO",
            # Age
            "BUILDING_AGE",
            # Previously missing — derived from YRBUILT bins only
            "BUILDING_ERA",
            # Prior year total assessment (FY2025 = last known, no leakage)
            "LOG_PYACTTOT",
            # Assessment ratios (all FY2025-based)
            "ASSESS_PER_SQFT", "LOG_ASSESS_PER_SQFT",
            "LAND_TO_TOTAL",
            "MKT_TO_ASSESS",
            # Previously missing — log of MKT_TO_ASSESS, better for skewed dist
            "LOG_MKT_TO_ASSESS",
            # Trend & volatility over FY2020-2025
            "ASSESS_TREND", "ASSESS_VOLATILITY",
        ] +
        # OLS-projected FY2026 values (extrapolated from FY2020-2025, no leakage)
        proj_feature_names +
        # Historical valuation status flags (FY2020-2025)
        historical_status_cols +
        # Log-transformed historical assessment series (FY2020-2025)
        log_acttot_cols + log_actland_cols + log_mkttot_cols +
        # Year-over-year % changes (FY2020-2025)
        yoy_cols + yoy_land_cols + yoy_mkt_cols
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