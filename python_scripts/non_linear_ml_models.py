"""
non_linear_ml_models.py
Trains a non-linear classifier (HistGradientBoosting) to classify
NYC properties as undervalued, fairly_valued, or overvalued.

New features — projected FY2026 values (leak-free):
  For each of FINACTTOT, FINACTLAND, FINMKTTOT we:
    1. Fit a vectorized OLS linear trend over the 6 known years (FY2020–FY2025)
       per property — this is just a dot product, no per-row Python loops.
    2. Extrapolate one year forward to get PROJ_<series>_FY2026.
    3. Compute PROJ_RATIO_<series>_FY2026 = projected_2026 / last_known_2025,
       i.e. the model's expected rate of change into 2026 given historical momentum.
    4. Compute PROJ_RESID_<series>_FY2026 = projected - last_known (dollar gap).

  All three series are leak-free: no FY2026 actuals are used anywhere.

RAM optimizations:
  - Feature engineering uses a single pd.concat() (no fragmentation).
  - Permutation importance uses a small subsample to avoid OOM.
  - Explicit gc.collect() at every major boundary.
"""

import pandas as pd
import numpy as np
import os
import time
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
MODEL_DIR  = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
OUTPUT_DIR = "/home/rodrigofrancachaves/project-nyc_property_taxes/outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HGB_SUBSAMPLE    = 300_000
HISTORICAL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(path):
    print("Loading data...")
    df = pd.read_parquet(path)
    df = df[df["target_2026"] != "unknown"].copy()
    print(f"  Shape after dropping unknowns: {df.shape}")
    return df


# ── Vectorized OLS trend extrapolation ───────────────────────────────────────
def project_next_year(df, col_list, series_name):
    """
    Given a list of columns representing consecutive yearly values
    (col_list[0] = earliest year, col_list[-1] = most recent year),
    fit a per-property OLS linear trend and extrapolate one step forward.

    Returns three Series (all leak-free, derived only from col_list):
      PROJ_<series_name>_FY2026  — projected dollar value
      PROJ_RATIO_<series_name>_FY2026 — projected / last known  (momentum ratio)
      PROJ_RESID_<series_name>_FY2026 — projected - last known  (dollar gap)

    Implementation: vectorized normal equations over the n years.
    No Python loop over rows — runs in numpy on the full 1M-row array.
    """
    n = len(col_list)
    # Normalize x to [0, n-1] so the intercept is near the data center
    x = np.arange(n, dtype=np.float64)          # shape (n,)
    x_mean = x.mean()
    x_c = x - x_mean                            # centered x

    # Build Y matrix: shape (n_properties, n_years)
    Y = df[col_list].fillna(0).to_numpy(dtype=np.float64)  # (N, n)

    # OLS slope per property: slope = (x_c @ Y.T) / (x_c @ x_c)
    # x_c @ Y.T  →  dot of (n,) with (n, N)  →  (N,)
    denom  = float(x_c @ x_c)                   # scalar
    slopes = (x_c @ Y.T) / denom                # (N,)

    # Intercept per property: mean(Y) - slope * mean(x) [but x is centered so mean(x_c)=0]
    intercepts = Y.mean(axis=1)                  # (N,) = mean of Y per row

    # Extrapolate to year index n (one step beyond the last known year)
    x_next     = n - x_mean                     # scalar: next step in centered coords
    projected  = intercepts + slopes * x_next   # (N,)
    projected  = np.clip(projected, 0, None)    # values can't be negative

    last_known = Y[:, -1]                        # (N,) — FY2025 actuals

    ratio = np.where(
        last_known > 0,
        projected / last_known,
        1.0                                      # flat if last_known is zero
    )
    ratio = np.clip(ratio, 0, 5)                 # cap extreme extrapolations

    resid = (projected - last_known).clip(-1e7, 1e7)

    idx = df.index
    return (
        pd.Series(projected, index=idx, name=f"PROJ_{series_name}_FY2026"),
        pd.Series(ratio,     index=idx, name=f"PROJ_RATIO_{series_name}_FY2026"),
        pd.Series(resid,     index=idx, name=f"PROJ_RESID_{series_name}_FY2026"),
    )


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    print("\nEngineering features...")

    # ── Numeric conversions ───────────────────────────────────────────────────
    base_numeric = [
        "GROSS_SQFT", "LAND_AREA", "NUM_BLDGS", "YRBUILT",
        "UNITS", "COOP_APTS", "BLD_STORY", "LOT_FRT", "LOT_DEP",
        "FINACTTOT", "FINACTLAND", "FINMKTTOT", "PYACTTOT"
    ]
    hist_numeric = (
        [f"FINACTTOT_FY{y}"  for y in HISTORICAL_YEARS] +
        [f"FINACTLAND_FY{y}" for y in HISTORICAL_YEARS] +
        [f"FINMKTTOT_FY{y}"  for y in HISTORICAL_YEARS]
    )
    for col in base_numeric + hist_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    new_cols = {}

    # ── Basic property features ───────────────────────────────────────────────
    new_cols["BUILDING_AGE"]   = (2026 - df["YRBUILT"]).clip(lower=0, upper=200)
    new_cols["LOG_GROSS_SQFT"] = np.log1p(df["GROSS_SQFT"].fillna(0))
    new_cols["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"].fillna(0))
    new_cols["LOG_PYACTTOT"]   = np.log1p(df["PYACTTOT"].fillna(0))
    new_cols["SQFT_PER_UNIT"]  = (df["GROSS_SQFT"] / df["UNITS"].clip(lower=1)).clip(upper=50000)
    new_cols["COVERAGE_RATIO"] = (df["GROSS_SQFT"] / df["LAND_AREA"].clip(lower=1)).clip(upper=50)
    new_cols["LOT_AREA"]       = df["LOT_FRT"] * df["LOT_DEP"]
    new_cols["BUILDING_ERA"]   = pd.cut(
        df["YRBUILT"],
        bins=[0, 1900, 1940, 1960, 1980, 2000, 2010, 2030],
        labels=[1, 2, 3, 4, 5, 6, 7]
    ).astype(float).fillna(0)

    # ── Assessment snapshot features (FY2025 — no leakage) ───────────────────
    assess_per_sqft = df["FINACTTOT_FY2025"].fillna(0) / df["GROSS_SQFT"].clip(lower=1)
    new_cols["ASSESS_PER_SQFT"]     = assess_per_sqft
    new_cols["LOG_ASSESS_PER_SQFT"] = np.log1p(assess_per_sqft)
    new_cols["LAND_TO_TOTAL"]       = (df["FINACTLAND_FY2025"].fillna(0) / df["FINACTTOT_FY2025"].clip(lower=1)).clip(0, 1)
    mkt_to_assess                   = (df["FINMKTTOT_FY2025"].fillna(0)  / df["FINACTTOT_FY2025"].clip(lower=1)).clip(0, 20)
    new_cols["MKT_TO_ASSESS"]       = mkt_to_assess
    new_cols["LOG_MKT_TO_ASSESS"]   = np.log1p(mkt_to_assess)

    # ── Historical column lists ───────────────────────────────────────────────
    finacttot_cols  = [f"FINACTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINACTTOT_FY{y}"  in df.columns]
    finactland_cols = [f"FINACTLAND_FY{y}" for y in HISTORICAL_YEARS if f"FINACTLAND_FY{y}" in df.columns]
    finmkttot_cols  = [f"FINMKTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINMKTTOT_FY{y}"  in df.columns]

    # ── Log transforms of historical values ──────────────────────────────────
    log_acttot_cols, log_actland_cols, log_mkttot_cols = [], [], []
    for col in finacttot_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_acttot_cols.append(name)
    for col in finactland_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_actland_cols.append(name)
    for col in finmkttot_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_mkttot_cols.append(name)

    # ── YoY % change (FY2020→FY2025, 5 deltas per series) ───────────────────
    def yoy_changes(cols, prefix):
        names = []
        for i in range(1, len(cols)):
            name = f"{prefix}_FY{HISTORICAL_YEARS[i]}"
            new_cols[name] = (
                (df[cols[i]].fillna(0) - df[cols[i-1]].fillna(0)) /
                df[cols[i-1]].fillna(1).clip(lower=1)
            ).clip(-1, 5)
            names.append(name)
        return names

    yoy_cols      = yoy_changes(finacttot_cols,  "ASSESS_YOY")
    yoy_land_cols = yoy_changes(finactland_cols, "LAND_YOY")
    yoy_mkt_cols  = yoy_changes(finmkttot_cols,  "MKT_YOY")

    # ── Volatility and trend ──────────────────────────────────────────────────
    new_cols["ASSESS_VOLATILITY"] = (
        pd.DataFrame({k: new_cols[k] for k in yoy_cols}).std(axis=1).fillna(0)
    )
    new_cols["ASSESS_TREND"] = (
        (df[finacttot_cols[-1]].fillna(0) - df[finacttot_cols[0]].fillna(0)) /
        df[finacttot_cols[0]].fillna(1).clip(lower=1)
    ).clip(-1, 10) if len(finacttot_cols) >= 2 else 0.0

    # ── Projected FY2026 features (vectorized OLS extrapolation) ─────────────
    # Each call adds 3 columns: projected value, momentum ratio, dollar residual.
    # All derived from FY2020–FY2025 only — zero leakage.
    print("  Computing projected FY2026 features (vectorized OLS)...")
    proj_feature_names = []

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
            print(f"    {series_name}: {proj.name}, {ratio.name}, {resid.name}")

    # ── Categorical encoding ──────────────────────────────────────────────────
    categorical_cols = ["BORO", "BLDG_CLASS", "ZIP_CODE", "ZONING"]
    le_dict = {}
    encoded_cat_cols = []
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            name = f"{col}_CODE"
            new_cols[name] = le.fit_transform(df[col].fillna("Unknown").astype(str))
            le_dict[col] = le
            encoded_cat_cols.append(name)

    # ── Single concat — eliminates DataFrame fragmentation ───────────────────
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    del new_cols, new_df
    gc.collect()

    # ── Historical valuation status columns ──────────────────────────────────
    historical_status_cols = [
        c for c in df.columns
        if any(str(yr) in c for yr in HISTORICAL_YEARS)
        and any(x in c for x in ["overvalued", "undervalued", "fairly_valued"])
    ]

    # ── Final feature list ────────────────────────────────────────────────────
    features = (
        encoded_cat_cols +
        [
            "LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS", "UNITS", "COOP_APTS",
            "BLD_STORY", "LOT_AREA", "BUILDING_AGE", "LOG_PYACTTOT",
            "LOG_ASSESS_PER_SQFT", "LAND_TO_TOTAL", "MKT_TO_ASSESS",
            "ASSESS_TREND", "ASSESS_VOLATILITY",
        ] +
        proj_feature_names +       # ← the 9 new projected FY2026 features
        historical_status_cols +
        log_acttot_cols + log_actland_cols + log_mkttot_cols +
        yoy_cols + yoy_land_cols + yoy_mkt_cols
    )
    features = [f for f in features if f in df.columns]

    print(f"  Total features: {len(features)}")
    print(f"  Projected FY2026 features added: {proj_feature_names}")
    return df, features, le_dict


# ── Prepare X and y ───────────────────────────────────────────────────────────
def prepare_xy(df, features, target_col="target_2026"):
    X = df[features].copy()
    y = df[target_col].astype(str)
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X, y


# ── Subsample (stratified, numpy — no extra DataFrame copy) ──────────────────
def subsample(X, y, n, seed=42):
    if len(X) <= n:
        return np.asarray(X), np.asarray(y)
    rng   = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    idx = np.concatenate([
        rng.choice(np.where(y_arr == cls)[0],
                   size=int(np.ceil((cnt / len(y)) * n)),
                   replace=False)
        for cls, cnt in zip(classes, counts)
    ])
    rng.shuffle(idx)
    idx = idx[:n]
    return np.asarray(X)[idx], y_arr[idx]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df, features, le_dict = engineer_features(df)
    X, y = prepare_xy(df, features)
    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nModel: HistGradientBoosting (non-linear, RAM-safe)")
    X_sub, y_sub = subsample(X_train, y_train, HGB_SUBSAMPLE)

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        {
            "max_iter":          [200, 300],
            "max_depth":         [5, 7, None],
            "learning_rate":     [0.05, 0.1, 0.2],
            "l2_regularization": [0.0, 0.1],
        },
        n_iter=10, cv=cv5, scoring="f1_macro",
        n_jobs=1, verbose=1, random_state=42, refit=True
    )

    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best params: {search.best_params_} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")
    del X_sub, y_sub
    gc.collect()

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = search.best_estimator_.predict(X_test.to_numpy())
    print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 Macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred, labels=["undervalued", "fairly_valued", "overvalued"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["undervalued", "fairly_valued", "overvalued"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("HistGradientBoosting — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hgb_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Feature importance (permutation, RAM-safe subsample) ─────────────────
    print("\nCalculating Permutation Importance (subsample n=3000)...")
    X_imp, y_imp = subsample(X_test, y_test, n=3000)
    r = permutation_importance(
        search.best_estimator_, X_imp, y_imp,
        n_repeats=3, random_state=42, n_jobs=1   # n_jobs=1 to avoid RAM spike
    )
    feat_imp = pd.DataFrame({
        "Feature":    features,
        "Importance": r.importances_mean,
        "Std":        r.importances_std,
    }).sort_values("Importance", ascending=False)

    feat_imp.to_csv(os.path.join(OUTPUT_DIR, "hgb_feature_importance.csv"), index=False)
    print(f"\nTop 15 features:\n{feat_imp.head(15).to_string(index=False)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "hgb_model.pkl"))
    joblib.dump(features,               os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict,                os.path.join(MODEL_DIR, "label_encoders.pkl"))

    print(f"\nResults saved to : {OUTPUT_DIR}")
    print(f"Model saved to   : {MODEL_DIR}")