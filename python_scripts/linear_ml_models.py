"""
linear_ml_models.py
Trains multiple linear classifiers to classify NYC properties as
undervalued, fairly_valued, or overvalued.

Linear models:
  1. SGDClassifier (modified_huber, L2) — fast, scales well
  2. SGDClassifier (modified_huber, L1) — sparse coefficients
  3. SGDClassifier (modified_huber, ElasticNet) — hybrid regularization

New features — projected FY2026 values (leak-free):
  For each of FINACTTOT, FINACTLAND, FINMKTTOT we:
    1. Fit a vectorized OLS linear trend over FY2020–FY2025 per property.
    2. Extrapolate one year forward → PROJ_<series>_FY2026.
    3. Compute PROJ_RATIO_<series>_FY2026 = projected / FY2025 (momentum).
    4. Compute PROJ_RESID_<series>_FY2026 = projected - FY2025 (dollar gap).
  All three series are leak-free: no FY2026 actuals used anywhere.

RAM optimizations:
  - engineer_features() batches all new columns into a single pd.concat().
  - X_train / X_test held as numpy arrays after scaling.
  - Subsampling uses numpy index draws — no second full-copy split.
  - Explicit del + gc.collect() at every major boundary.

Primary metric: macro F1 (treats all classes equally).
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
    cross_val_score, GridSearchCV
)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

LINEAR_SUBSAMPLE = 100_000
HISTORICAL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(path):
    print("Loading data...")
    df = pd.read_parquet(path)
    print(f"  Loaded shape: {df.shape}")
    df = df[df["target_2026"] != "unknown"].copy()
    print(f"  Shape after dropping unknowns: {df.shape}")
    print(f"\nTarget distribution:\n{df['target_2026'].value_counts()}")
    print(f"\nTarget proportions:\n{df['target_2026'].value_counts(normalize=True).round(3)}")
    return df


# ── Vectorized OLS trend extrapolation ───────────────────────────────────────
def project_next_year(df, col_list, series_name):
    """
    Given columns representing consecutive yearly values (FY2020–FY2025),
    fit a per-property OLS linear trend and extrapolate one step forward.

    Returns three Series (all leak-free, derived only from col_list):
      PROJ_<series_name>_FY2026  — projected dollar value
      PROJ_RATIO_<series_name>_FY2026 — projected / FY2025  (momentum ratio)
      PROJ_RESID_<series_name>_FY2026 — projected - FY2025  (dollar gap)

    Fully vectorized via numpy normal equations — no Python loop over rows.
    """
    n     = len(col_list)
    x     = np.arange(n, dtype=np.float64)
    x_c   = x - x.mean()                        # centered x, shape (n,)
    denom = float(x_c @ x_c)                    # scalar

    Y          = df[col_list].fillna(0).to_numpy(dtype=np.float64)  # (N, n)
    slopes     = (x_c @ Y.T) / denom            # (N,)
    intercepts = Y.mean(axis=1)                  # (N,)

    x_next    = n - x.mean()                    # next step in centered coords
    projected = np.clip(intercepts + slopes * x_next, 0, None)  # (N,)
    last_known = Y[:, -1]                        # FY2025 actuals, (N,)

    ratio = np.clip(
        np.where(last_known > 0, projected / last_known, 1.0),
        0, 5
    )
    resid = (projected - last_known).clip(-1e7, 1e7)

    idx = df.index
    return (
        pd.Series(projected, index=idx, name=f"PROJ_{series_name}_FY2026"),
        pd.Series(ratio,     index=idx, name=f"PROJ_RATIO_{series_name}_FY2026"),
        pd.Series(resid,     index=idx, name=f"PROJ_RESID_{series_name}_FY2026"),
    )


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    """
    Build all engineered columns and return (df_with_features, feature_list, le_dict).

    RAM fix: accumulate every new column in a plain dict, then do a single
    pd.concat at the end — eliminates DataFrame fragmentation.
    """
    print("\nEngineering features...")

    # ── Numeric conversions (in-place on existing columns only) ───────────────
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
    assess_per_sqft             = df["FINACTTOT_FY2025"].fillna(0) / df["GROSS_SQFT"].clip(lower=1)
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

    # ── YoY % change ─────────────────────────────────────────────────────────
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

    # ── Market vs assessed gap ────────────────────────────────────────────────
    gap_cols = []
    for i in range(1, len(HISTORICAL_YEARS)):
        y = HISTORICAL_YEARS[i]
        mkt_yoy = f"MKT_YOY_FY{y}"; act_yoy = f"ASSESS_YOY_FY{y}"
        if mkt_yoy in new_cols and act_yoy in new_cols:
            name = f"MKT_ASSESS_GAP_YOY_FY{y}"
            new_cols[name] = (new_cols[mkt_yoy] - new_cols[act_yoy]).clip(-5, 5)
            gap_cols.append(name)

    # ── Cumulative growth from 2020 baseline ──────────────────────────────────
    def cumul_growth(cols, base_col, prefix):
        names = []
        if base_col not in df.columns:
            return names
        for y, col in zip(HISTORICAL_YEARS[1:], cols[1:]):
            if col in df.columns:
                name = f"{prefix}_FY{y}"
                new_cols[name] = (
                    (df[col].fillna(0) - df[base_col].fillna(0)) /
                    df[base_col].fillna(1).clip(lower=1)
                ).clip(-1, 10)
                names.append(name)
        return names

    cumul_cols      = cumul_growth(finacttot_cols,  "FINACTTOT_FY2020",  "CUMUL_GROWTH")
    cumul_land_cols = cumul_growth(finactland_cols, "FINACTLAND_FY2020", "CUMUL_LAND_GROWTH")
    cumul_mkt_cols  = cumul_growth(finmkttot_cols,  "FINMKTTOT_FY2020",  "CUMUL_MKT_GROWTH")

    # ── Acceleration (2nd derivative) ─────────────────────────────────────────
    def acceleration(yoy, prefix):
        names = []
        for i in range(2, len(yoy)):
            name = f"{prefix}_FY{HISTORICAL_YEARS[i + 1]}"
            new_cols[name] = (new_cols[yoy[i]] - new_cols[yoy[i-1]]).clip(-5, 5)
            names.append(name)
        return names

    accel_cols      = acceleration(yoy_cols,      "ASSESS_ACCEL")
    land_accel_cols = acceleration(yoy_land_cols, "LAND_ACCEL")
    mkt_accel_cols  = acceleration(yoy_mkt_cols,  "MKT_ACCEL")

    # ── Volatility ────────────────────────────────────────────────────────────
    new_cols["ASSESS_VOLATILITY"] = pd.DataFrame({k: new_cols[k] for k in yoy_cols}).std(axis=1).fillna(0) if yoy_cols else 0.0
    new_cols["LAND_VOLATILITY"]   = pd.DataFrame({k: new_cols[k] for k in yoy_land_cols}).std(axis=1).fillna(0) if yoy_land_cols else 0.0
    new_cols["MKT_VOLATILITY"]    = pd.DataFrame({k: new_cols[k] for k in yoy_mkt_cols}).std(axis=1).fillna(0) if yoy_mkt_cols else 0.0

    # ── Overall trends ────────────────────────────────────────────────────────
    def overall_trend(cols):
        avail = [c for c in cols if c in df.columns]
        if len(avail) >= 2:
            return ((df[avail[-1]].fillna(0) - df[avail[0]].fillna(0)) /
                    df[avail[0]].fillna(1).clip(lower=1)).clip(-1, 10)
        return 0.0

    new_cols["ASSESS_TREND"] = overall_trend(finacttot_cols)
    new_cols["LAND_TREND"]   = overall_trend(finactland_cols)
    new_cols["MKT_TREND"]    = overall_trend(finmkttot_cols)

    # ── Assessment cap flag ───────────────────────────────────────────────────
    new_cols["ASSESS_AT_CAP"] = new_cols[yoy_cols[-1]].between(0.04, 0.07).astype(int) if yoy_cols else 0

    # ── Land ratio per year ───────────────────────────────────────────────────
    land_ratio_cols = []
    for y in HISTORICAL_YEARS:
        act_col = f"FINACTTOT_FY{y}"; land_col = f"FINACTLAND_FY{y}"
        if act_col in df.columns and land_col in df.columns:
            name = f"LAND_RATIO_FY{y}"
            new_cols[name] = (df[land_col].fillna(0) / df[act_col].fillna(1).clip(lower=1)).clip(0, 1)
            land_ratio_cols.append(name)
    new_cols["LAND_RATIO_TREND"] = (new_cols[land_ratio_cols[-1]] - new_cols[land_ratio_cols[0]]).clip(-1, 1) if len(land_ratio_cols) >= 2 else 0.0

    # ── Market/assessed ratio per year ────────────────────────────────────────
    mkt_ratio_cols = []
    for y in HISTORICAL_YEARS:
        mkt_col = f"FINMKTTOT_FY{y}"; act_col = f"FINACTTOT_FY{y}"
        if mkt_col in df.columns and act_col in df.columns:
            name = f"MKT_ASSESS_RATIO_FY{y}"
            new_cols[name] = (df[act_col].fillna(0) / df[mkt_col].fillna(1).clip(lower=1)).clip(0, 5)
            mkt_ratio_cols.append(name)
    new_cols["MKT_RATIO_TREND"] = (new_cols[mkt_ratio_cols[-1]] - new_cols[mkt_ratio_cols[0]]).clip(-5, 5) if len(mkt_ratio_cols) >= 2 else 0.0

    # ── Assessed per sqft per year ────────────────────────────────────────────
    psqft_cols = []
    for y in HISTORICAL_YEARS:
        act_col = f"FINACTTOT_FY{y}"
        if act_col in df.columns:
            name = f"ASSESS_PER_SQFT_FY{y}"
            new_cols[name] = (df[act_col].fillna(0) / df["GROSS_SQFT"].clip(lower=1))
            psqft_cols.append(name)
    new_cols["PSQFT_TREND"] = (new_cols[psqft_cols[-1]] - new_cols[psqft_cols[0]]).clip(-10000, 10000) if len(psqft_cols) >= 2 else 0.0

    # ── Consistency scores ────────────────────────────────────────────────────
    over_cols  = [c for c in df.columns if "overvalued_"    in c and any(str(y) in c for y in HISTORICAL_YEARS)]
    under_cols = [c for c in df.columns if "undervalued_"   in c and any(str(y) in c for y in HISTORICAL_YEARS)]
    fair_cols  = [c for c in df.columns if "fairly_valued_" in c and any(str(y) in c for y in HISTORICAL_YEARS)]
    new_cols["CONSISTENT_OVERVALUED"]  = df[over_cols].sum(axis=1)  if over_cols  else 0
    new_cols["CONSISTENT_UNDERVALUED"] = df[under_cols].sum(axis=1) if under_cols else 0
    new_cols["CONSISTENT_FAIR"]        = df[fair_cols].sum(axis=1)  if fair_cols  else 0

    # ── Projected FY2026 features (vectorized OLS extrapolation) ─────────────
    # Derived from FY2020–FY2025 only — zero leakage into the 2026 target.
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

    # ── Historical status columns ─────────────────────────────────────────────
    historical_status_cols = [
        c for c in df.columns
        if any(str(yr) in c for yr in HISTORICAL_YEARS)
        and any(x in c for x in ["overvalued", "undervalued", "fairly_valued"])
    ]

    # ── Final feature list ────────────────────────────────────────────────────
    features = (
        encoded_cat_cols +
        [
            "LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS",
            "UNITS", "COOP_APTS", "BLD_STORY",
            "LOT_FRT", "LOT_DEP", "LOT_AREA",
            "BUILDING_AGE", "BUILDING_ERA",
            "SQFT_PER_UNIT", "COVERAGE_RATIO",
            "LOG_PYACTTOT",
            "LOG_ASSESS_PER_SQFT",
            "LAND_TO_TOTAL",
            "MKT_TO_ASSESS", "LOG_MKT_TO_ASSESS",
            "ASSESS_TREND", "LAND_TREND", "MKT_TREND",
            "ASSESS_VOLATILITY", "LAND_VOLATILITY", "MKT_VOLATILITY",
            "ASSESS_AT_CAP",
            "LAND_RATIO_TREND", "MKT_RATIO_TREND", "PSQFT_TREND",
            "CONSISTENT_OVERVALUED", "CONSISTENT_UNDERVALUED", "CONSISTENT_FAIR",
        ] +
        proj_feature_names +           # ← 9 new projected FY2026 features
        historical_status_cols +
        log_acttot_cols + log_actland_cols + log_mkttot_cols +
        yoy_cols + yoy_land_cols + yoy_mkt_cols +
        gap_cols +
        cumul_cols + cumul_land_cols + cumul_mkt_cols +
        accel_cols + land_accel_cols + mkt_accel_cols +
        land_ratio_cols + mkt_ratio_cols + psqft_cols
    )
    features = [f for f in features if f in df.columns]

    print(f"  Total features: {len(features)}")
    print(f"  Projected FY2026 features added: {proj_feature_names}")
    return df, features, le_dict


# ── Prepare X and y ───────────────────────────────────────────────────────────
def prepare_xy(df, features, target_col="target_2026"):
    print("\nPreparing X and y...")
    X = df[features].copy()
    y = df[target_col].astype(str)
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    print(f"  X shape: {X.shape}")
    print(f"  y distribution:\n{y.value_counts()}")
    return X, y


# ── Subsample helper (numpy — no extra DataFrame copy) ────────────────────────
def subsample(X, y, n, seed=42):
    if len(X) <= n:
        return np.asarray(X), np.asarray(y)
    rng   = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    fracs = counts / counts.sum()
    idx = np.concatenate([
        rng.choice(np.where(y_arr == cls)[0], size=int(np.ceil(frac * n)), replace=False)
        for cls, frac in zip(classes, fracs)
    ])
    rng.shuffle(idx)
    idx = idx[:n]
    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    return X_arr[idx], y_arr[idx]


# ── Generic evaluation ────────────────────────────────────────────────────────
def evaluate(name, model, X_test, y_test, X_train, y_train, subsample_n):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")
    print(f"\n  Test Accuracy    : {acc:.4f}")
    print(f"  Test F1 Macro    : {f1m:.4f}  ← primary metric")
    print(f"  Test F1 Weighted : {f1w:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    cm = confusion_matrix(
        y_test, y_pred,
        labels=["undervalued", "fairly_valued", "overvalued"]
    )
    print(f"Confusion Matrix:\n{cm}")
    X_cv, y_cv = subsample(X_train, y_train, subsample_n, seed=99)
    cv_scores = cross_val_score(
        model, X_cv, y_cv,
        cv=StratifiedKFold(5, shuffle=True, random_state=99),
        scoring="f1_macro", n_jobs=-1
    )
    print(f"  CV F1 Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return {
        "Model":            name,
        "Test Accuracy":    round(acc, 4),
        "Test F1 Macro":    round(f1m, 4),
        "Test F1 Weighted": round(f1w, 4),
        "CV F1 Macro":      round(cv_scores.mean(), 4),
        "CV F1 Std":        round(cv_scores.std(), 4)
    }, cm


# ── Plot coefficients ─────────────────────────────────────────────────────────
def plot_coefficients(model, features, name, output_dir, top_n=20):
    classes = model.classes_
    coef_df = pd.DataFrame(model.coef_, index=classes, columns=features)
    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 8))
    if len(classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        top  = coef_df.loc[cls].abs().nlargest(top_n).index
        vals = coef_df.loc[cls][top].sort_values()
        colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]
        vals.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Top {top_n} coefficients\nClass: {cls}", fontsize=12)
        ax.set_xlabel("Coefficient value")
        ax.axvline(0, color="black", linewidth=0.8)
    plt.suptitle(f"{name} — Coefficients by Class", fontsize=13)
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    out  = os.path.join(output_dir, f"{slug}_coefficients.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    coef_df.T.reset_index().rename(columns={"index": "feature"}).to_csv(
        out.replace(".png", ".csv"), index=False
    )
    print(f"  Coefficients saved: {out}")


# ── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_cm(cm, name, output_dir):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["undervalued", "fairly_valued", "overvalued"]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    out  = os.path.join(output_dir, f"{slug}_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {out}")


# ── Train one SGD linear model ────────────────────────────────────────────────
def train_sgd(name, penalty, extra_params, X_train_sc, y_train_r,
              X_test_sc, y_test_r, features, cv5, output_dir, model_dir):
    print(f"\n{'='*60}")
    print(f"Model: SGDClassifier — {name}")
    X_sub, y_sub = subsample(X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    search = GridSearchCV(
        SGDClassifier(loss="modified_huber", penalty=penalty,
                      class_weight="balanced", max_iter=1000, tol=1e-3,
                      random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10,
                      **extra_params),
        {"alpha": [0.0001, 0.001, 0.01, 0.1]},
        cv=cv5, scoring="f1_macro", n_jobs=1, verbose=1, refit=True
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best params: {search.best_params_} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")
    del X_sub, y_sub
    gc.collect()

    res, cm = evaluate(f"SGD {name}", search.best_estimator_,
                       X_test_sc, y_test_r, X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    res["Best Params"] = str(search.best_params_)
    plot_coefficients(search.best_estimator_, features, f"SGD {name}", output_dir)
    plot_cm(cm, f"SGD {name}", output_dir)

    slug = name.lower().replace(" ", "_")
    joblib.dump(search.best_estimator_, os.path.join(model_dir, f"sgd_{slug}.pkl"))
    best = search.best_estimator_
    del search
    gc.collect()
    return res, best


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
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── Scale and immediately convert to numpy to save RAM ────────────────────
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    y_train_r  = y_train.to_numpy()
    y_test_r   = y_test.to_numpy()

    del X_train, X_test, y_train, y_test, X, y
    gc.collect()

    baseline = pd.Series(y_train_r).value_counts(normalize=True).max()
    print(f"\nBaseline (majority class): {baseline:.4f}")

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    # ── Linear models ─────────────────────────────────────────────────────────
    res, _ = train_sgd("L2", "l2", {},
                       X_train_sc, y_train_r, X_test_sc, y_test_r,
                       features, cv5, OUTPUT_DIR, MODEL_DIR)
    all_results.append(res)

    res, _ = train_sgd("L1", "l1", {},
                       X_train_sc, y_train_r, X_test_sc, y_test_r,
                       features, cv5, OUTPUT_DIR, MODEL_DIR)
    all_results.append(res)

    res, _ = train_sgd("ElasticNet", "elasticnet", {"l1_ratio": 0.85},
                       X_train_sc, y_train_r, X_test_sc, y_test_r,
                       features, cv5, OUTPUT_DIR, MODEL_DIR)
    all_results.append(res)

    # ── Save shared artefacts ─────────────────────────────────────────────────
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "linear_features.pkl"))
    joblib.dump(le_dict,  os.path.join(MODEL_DIR, "label_encoders.pkl"))

    results_df = pd.DataFrame(all_results).sort_values("Test F1 Macro", ascending=False)
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON (ranked by macro F1)")
    print(f"Baseline (majority class): {baseline:.4f}")
    print(results_df[[
        "Model", "Test Accuracy", "Test F1 Macro",
        "Test F1 Weighted", "CV F1 Macro", "CV F1 Std"
    ]].to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "linear_model_results.csv"), index=False)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"All models saved to:  {MODEL_DIR}")