"""
linear_model.py
Trains multiple linear classifiers and one non-linear model to classify
NYC properties as undervalued, fairly_valued, or overvalued.

Linear models:
  1. SGDClassifier (modified_huber, L2) — fast, scales well
  2. SGDClassifier (modified_huber, L1) — sparse coefficients
  3. SGDClassifier (modified_huber, ElasticNet) — hybrid regularization
  4. Passive Aggressive Classifier — online linear learner

Non-linear:
  5. HistGradientBoostingClassifier — best non-linear, RAM-safe

All models use subsampling and balanced class weights.
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
    cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

LINEAR_SUBSAMPLE = 100_000
HGB_SUBSAMPLE    = 300_000
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

    # ── Basic property features ───────────────────────────────────────────────
    df["BUILDING_AGE"]    = (2026 - df["YRBUILT"]).clip(lower=0, upper=200)
    df["LOG_GROSS_SQFT"]  = np.log1p(df["GROSS_SQFT"].fillna(0))
    df["LOG_LAND_AREA"]   = np.log1p(df["LAND_AREA"].fillna(0))
    df["LOG_PYACTTOT"]    = np.log1p(df["PYACTTOT"].fillna(0))
    df["SQFT_PER_UNIT"]   = (df["GROSS_SQFT"] / df["UNITS"].clip(lower=1)).clip(upper=50000)
    df["COVERAGE_RATIO"]  = (df["GROSS_SQFT"] / df["LAND_AREA"].clip(lower=1)).clip(upper=50)
    df["LOT_AREA"]        = df["LOT_FRT"] * df["LOT_DEP"]
    df["BUILDING_ERA"]    = pd.cut(
        df["YRBUILT"],
        bins=[0, 1900, 1940, 1960, 1980, 2000, 2010, 2030],
        labels=[1, 2, 3, 4, 5, 6, 7]
    ).astype(float).fillna(0)

    # ── Current assessment features ───────────────────────────────────────────
    df["ASSESS_PER_SQFT"]     = (df["FINACTTOT"].fillna(0) / df["GROSS_SQFT"].clip(lower=1))
    df["LOG_ASSESS_PER_SQFT"] = np.log1p(df["ASSESS_PER_SQFT"])
    df["LAND_TO_TOTAL"]       = (df["FINACTLAND"].fillna(0) / df["FINACTTOT"].clip(lower=1)).clip(0, 1)
    df["MKT_TO_ASSESS"]       = (df["FINMKTTOT"].fillna(0)  / df["FINACTTOT"].clip(lower=1)).clip(0, 20)
    df["LOG_MKT_TO_ASSESS"]   = np.log1p(df["MKT_TO_ASSESS"])

    # ── Log transforms for all historical columns ─────────────────────────────
    finacttot_cols  = [f"FINACTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINACTTOT_FY{y}"  in df.columns]
    finactland_cols = [f"FINACTLAND_FY{y}" for y in HISTORICAL_YEARS if f"FINACTLAND_FY{y}" in df.columns]
    finmkttot_cols  = [f"FINMKTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINMKTTOT_FY{y}"  in df.columns]

    for col in finacttot_cols + finactland_cols + finmkttot_cols:
        df[f"LOG_{col}"] = np.log1p(df[col].fillna(0))

    log_acttot_cols  = [f"LOG_{c}" for c in finacttot_cols]
    log_actland_cols = [f"LOG_{c}" for c in finactland_cols]
    log_mkttot_cols  = [f"LOG_{c}" for c in finmkttot_cols]

    # ── YoY % change in assessed total ───────────────────────────────────────
    yoy_cols = []
    for i in range(1, len(HISTORICAL_YEARS)):
        y_curr   = HISTORICAL_YEARS[i]
        y_prev   = HISTORICAL_YEARS[i - 1]
        curr_col = f"FINACTTOT_FY{y_curr}"
        prev_col = f"FINACTTOT_FY{y_prev}"
        if curr_col in df.columns and prev_col in df.columns:
            col_name = f"ASSESS_YOY_FY{y_curr}"
            df[col_name] = (
                (df[curr_col].fillna(0) - df[prev_col].fillna(0)) /
                df[prev_col].fillna(1).clip(lower=1)
            ).clip(-1, 5)
            yoy_cols.append(col_name)

    # ── YoY % change in assessed LAND value ──────────────────────────────────
    yoy_land_cols = []
    for i in range(1, len(HISTORICAL_YEARS)):
        y_curr   = HISTORICAL_YEARS[i]
        y_prev   = HISTORICAL_YEARS[i - 1]
        curr_col = f"FINACTLAND_FY{y_curr}"
        prev_col = f"FINACTLAND_FY{y_prev}"
        if curr_col in df.columns and prev_col in df.columns:
            col_name = f"LAND_YOY_FY{y_curr}"
            df[col_name] = (
                (df[curr_col].fillna(0) - df[prev_col].fillna(0)) /
                df[prev_col].fillna(1).clip(lower=1)
            ).clip(-1, 5)
            yoy_land_cols.append(col_name)

    # ── YoY % change in market value ─────────────────────────────────────────
    yoy_mkt_cols = []
    for i in range(1, len(HISTORICAL_YEARS)):
        y_curr   = HISTORICAL_YEARS[i]
        y_prev   = HISTORICAL_YEARS[i - 1]
        curr_col = f"FINMKTTOT_FY{y_curr}"
        prev_col = f"FINMKTTOT_FY{y_prev}"
        if curr_col in df.columns and prev_col in df.columns:
            col_name = f"MKT_YOY_FY{y_curr}"
            df[col_name] = (
                (df[curr_col].fillna(0) - df[prev_col].fillna(0)) /
                df[prev_col].fillna(1).clip(lower=1)
            ).clip(-1, 5)
            yoy_mkt_cols.append(col_name)

    # ── Gap: market growth vs assessed growth per year ────────────────────────
    # Positive = market growing faster than assessed = likely undervalued
    gap_cols = []
    for i in range(1, len(HISTORICAL_YEARS)):
        y_curr  = HISTORICAL_YEARS[i]
        mkt_yoy = f"MKT_YOY_FY{y_curr}"
        act_yoy = f"ASSESS_YOY_FY{y_curr}"
        if mkt_yoy in df.columns and act_yoy in df.columns:
            col_name = f"MKT_ASSESS_GAP_YOY_FY{y_curr}"
            df[col_name] = (df[mkt_yoy] - df[act_yoy]).clip(-5, 5)
            gap_cols.append(col_name)

    # ── Cumulative growth from 2020 baseline ──────────────────────────────────
    cumul_cols = []
    base_col = "FINACTTOT_FY2020"
    if base_col in df.columns:
        for y in HISTORICAL_YEARS[1:]:
            curr_col = f"FINACTTOT_FY{y}"
            if curr_col in df.columns:
                col_name = f"CUMUL_GROWTH_FY{y}"
                df[col_name] = (
                    (df[curr_col].fillna(0) - df[base_col].fillna(0)) /
                    df[base_col].fillna(1).clip(lower=1)
                ).clip(-1, 10)
                cumul_cols.append(col_name)

    cumul_land_cols = []
    base_land = "FINACTLAND_FY2020"
    if base_land in df.columns:
        for y in HISTORICAL_YEARS[1:]:
            curr_col = f"FINACTLAND_FY{y}"
            if curr_col in df.columns:
                col_name = f"CUMUL_LAND_GROWTH_FY{y}"
                df[col_name] = (
                    (df[curr_col].fillna(0) - df[base_land].fillna(0)) /
                    df[base_land].fillna(1).clip(lower=1)
                ).clip(-1, 10)
                cumul_land_cols.append(col_name)

    cumul_mkt_cols = []
    base_mkt = "FINMKTTOT_FY2020"
    if base_mkt in df.columns:
        for y in HISTORICAL_YEARS[1:]:
            curr_col = f"FINMKTTOT_FY{y}"
            if curr_col in df.columns:
                col_name = f"CUMUL_MKT_GROWTH_FY{y}"
                df[col_name] = (
                    (df[curr_col].fillna(0) - df[base_mkt].fillna(0)) /
                    df[base_mkt].fillna(1).clip(lower=1)
                ).clip(-1, 10)
                cumul_mkt_cols.append(col_name)

    # ── Acceleration (2nd derivative of growth) ───────────────────────────────
    accel_cols = []
    for i in range(2, len(yoy_cols)):
        col_name = f"ASSESS_ACCEL_FY{HISTORICAL_YEARS[i + 1]}"
        df[col_name] = (df[yoy_cols[i]] - df[yoy_cols[i - 1]]).clip(-5, 5)
        accel_cols.append(col_name)

    land_accel_cols = []
    for i in range(2, len(yoy_land_cols)):
        col_name = f"LAND_ACCEL_FY{HISTORICAL_YEARS[i + 1]}"
        df[col_name] = (df[yoy_land_cols[i]] - df[yoy_land_cols[i - 1]]).clip(-5, 5)
        land_accel_cols.append(col_name)

    mkt_accel_cols = []
    for i in range(2, len(yoy_mkt_cols)):
        col_name = f"MKT_ACCEL_FY{HISTORICAL_YEARS[i + 1]}"
        df[col_name] = (df[yoy_mkt_cols[i]] - df[yoy_mkt_cols[i - 1]]).clip(-5, 5)
        mkt_accel_cols.append(col_name)

    # ── Volatility ────────────────────────────────────────────────────────────
    if yoy_cols:
        df["ASSESS_VOLATILITY"] = df[yoy_cols].std(axis=1).fillna(0)
    else:
        df["ASSESS_VOLATILITY"] = 0.0

    if yoy_land_cols:
        df["LAND_VOLATILITY"] = df[yoy_land_cols].std(axis=1).fillna(0)
    else:
        df["LAND_VOLATILITY"] = 0.0

    if yoy_mkt_cols:
        df["MKT_VOLATILITY"] = df[yoy_mkt_cols].std(axis=1).fillna(0)
    else:
        df["MKT_VOLATILITY"] = 0.0

    # ── Overall trends ────────────────────────────────────────────────────────
    avail = sorted([c for c in finacttot_cols if c in df.columns])
    df["ASSESS_TREND"] = (
        (df[avail[-1]].fillna(0) - df[avail[0]].fillna(0)) /
        df[avail[0]].fillna(1).clip(lower=1)
    ).clip(-1, 10) if len(avail) >= 2 else 0.0

    avail_land = sorted([c for c in finactland_cols if c in df.columns])
    df["LAND_TREND"] = (
        (df[avail_land[-1]].fillna(0) - df[avail_land[0]].fillna(0)) /
        df[avail_land[0]].fillna(1).clip(lower=1)
    ).clip(-1, 10) if len(avail_land) >= 2 else 0.0

    avail_mkt = sorted([c for c in finmkttot_cols if c in df.columns])
    df["MKT_TREND"] = (
        (df[avail_mkt[-1]].fillna(0) - df[avail_mkt[0]].fillna(0)) /
        df[avail_mkt[0]].fillna(1).clip(lower=1)
    ).clip(-1, 10) if len(avail_mkt) >= 2 else 0.0

    # ── Assessment cap flag ───────────────────────────────────────────────────
    df["ASSESS_AT_CAP"] = 0
    if yoy_cols:
        df["ASSESS_AT_CAP"] = df[yoy_cols[-1]].between(0.04, 0.07).astype(int)

    # ── Land ratio per year ───────────────────────────────────────────────────
    land_ratio_cols = []
    for y in HISTORICAL_YEARS:
        act_col  = f"FINACTTOT_FY{y}"
        land_col = f"FINACTLAND_FY{y}"
        if act_col in df.columns and land_col in df.columns:
            col_name = f"LAND_RATIO_FY{y}"
            df[col_name] = (
                df[land_col].fillna(0) /
                df[act_col].fillna(1).clip(lower=1)
            ).clip(0, 1)
            land_ratio_cols.append(col_name)

    if len(land_ratio_cols) >= 2:
        df["LAND_RATIO_TREND"] = (
            df[land_ratio_cols[-1]].fillna(0) -
            df[land_ratio_cols[0]].fillna(0)
        ).clip(-1, 1)
    else:
        df["LAND_RATIO_TREND"] = 0.0

    # ── Market/assessed ratio per year ────────────────────────────────────────
    mkt_ratio_cols = []
    for y in HISTORICAL_YEARS:
        mkt_col = f"FINMKTTOT_FY{y}"
        act_col = f"FINACTTOT_FY{y}"
        if mkt_col in df.columns and act_col in df.columns:
            col_name = f"MKT_ASSESS_RATIO_FY{y}"
            df[col_name] = (
                df[act_col].fillna(0) /
                df[mkt_col].fillna(1).clip(lower=1)
            ).clip(0, 5)
            mkt_ratio_cols.append(col_name)

    if len(mkt_ratio_cols) >= 2:
        df["MKT_RATIO_TREND"] = (
            df[mkt_ratio_cols[-1]].fillna(0) -
            df[mkt_ratio_cols[0]].fillna(0)
        ).clip(-5, 5)
    else:
        df["MKT_RATIO_TREND"] = 0.0

    # ── Assessed per sqft per year ────────────────────────────────────────────
    psqft_cols = []
    for y in HISTORICAL_YEARS:
        act_col = f"FINACTTOT_FY{y}"
        if act_col in df.columns:
            col_name = f"ASSESS_PER_SQFT_FY{y}"
            df[col_name] = (
                df[act_col].fillna(0) / df["GROSS_SQFT"].clip(lower=1)
            )
            psqft_cols.append(col_name)

    if len(psqft_cols) >= 2:
        df["PSQFT_TREND"] = (
            df[psqft_cols[-1]].fillna(0) -
            df[psqft_cols[0]].fillna(0)
        ).clip(-10000, 10000)
    else:
        df["PSQFT_TREND"] = 0.0

    # ── Consistency scores ────────────────────────────────────────────────────
    over_cols  = [c for c in df.columns if "overvalued_"    in c and any(str(y) in c for y in HISTORICAL_YEARS)]
    under_cols = [c for c in df.columns if "undervalued_"   in c and any(str(y) in c for y in HISTORICAL_YEARS)]
    fair_cols  = [c for c in df.columns if "fairly_valued_" in c and any(str(y) in c for y in HISTORICAL_YEARS)]

    df["CONSISTENT_OVERVALUED"]  = df[over_cols].sum(axis=1)  if over_cols  else 0
    df["CONSISTENT_UNDERVALUED"] = df[under_cols].sum(axis=1) if under_cols else 0
    df["CONSISTENT_FAIR"]        = df[fair_cols].sum(axis=1)  if fair_cols  else 0

    # ── Encode categoricals ───────────────────────────────────────────────────
    categorical_cols = ["BORO", "BLDG_CLASS", "ZIP_CODE", "ZONING"]
    le_dict = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_CODE"] = le.fit_transform(
                df[col].fillna("Unknown").astype(str)
            )
            le_dict[col] = le
    encoded_cat_cols = [f"{c}_CODE" for c in categorical_cols if c in df.columns]

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
        historical_status_cols +
        log_acttot_cols +
        log_actland_cols +
        log_mkttot_cols +
        yoy_cols +
        yoy_land_cols +
        yoy_mkt_cols +
        gap_cols +
        cumul_cols +
        cumul_land_cols +
        cumul_mkt_cols +
        accel_cols +
        land_accel_cols +
        mkt_accel_cols +
        land_ratio_cols +
        mkt_ratio_cols +
        psqft_cols
    )

    features = [f for f in features if f in df.columns]

    print(f"  Total features: {len(features)}")
    print(f"  Encoded categoricals:     {len(encoded_cat_cols)}")
    print(f"  Historical status:        {len(historical_status_cols)}")
    print(f"  Log assessed total:       {len(log_acttot_cols)}")
    print(f"  Log assessed land:        {len(log_actland_cols)}")
    print(f"  Log market total:         {len(log_mkttot_cols)}")
    print(f"  Assessed YoY:             {len(yoy_cols)}")
    print(f"  Land YoY:                 {len(yoy_land_cols)}")
    print(f"  Market YoY:               {len(yoy_mkt_cols)}")
    print(f"  Mkt vs assessed gap YoY:  {len(gap_cols)}")
    print(f"  Cumulative assessed:      {len(cumul_cols)}")
    print(f"  Cumulative land:          {len(cumul_land_cols)}")
    print(f"  Cumulative market:        {len(cumul_mkt_cols)}")
    print(f"  Assessed acceleration:    {len(accel_cols)}")
    print(f"  Land acceleration:        {len(land_accel_cols)}")
    print(f"  Market acceleration:      {len(mkt_accel_cols)}")
    print(f"  Land ratio per year:      {len(land_ratio_cols)}")
    print(f"  Market ratio per year:    {len(mkt_ratio_cols)}")
    print(f"  Assessed per sqft:        {len(psqft_cols)}")

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


# ── Subsample helper ──────────────────────────────────────────────────────────
def subsample(X, y, n, seed=42):
    if len(X) > n:
        X_s, _, y_s, _ = train_test_split(
            X, y, train_size=n, random_state=seed, stratify=y
        )
        return X_s, y_s
    return X, y


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
    classes  = model.classes_
    coef_df  = pd.DataFrame(model.coef_, index=classes, columns=features)
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
    return coef_df


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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    df = load_data(DATA_PATH)
    df, features, le_dict = engineer_features(df)
    X, y = prepare_xy(df, features)
    del df; gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=features, index=X_train.index
    ).reset_index(drop=True)
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=features, index=X_test.index
    ).reset_index(drop=True)
    y_train_r = y_train.reset_index(drop=True)
    y_test_r  = y_test.reset_index(drop=True)

    baseline = y.value_counts(normalize=True).max()
    print(f"\nBaseline (majority class): {baseline:.4f}")

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    # ── Model 1: SGD L2 ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Model 1: SGDClassifier — L2 (Ridge)")
    X_sub, y_sub = subsample(X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    search = GridSearchCV(
        SGDClassifier(loss="modified_huber", penalty="l2",
                      class_weight="balanced", max_iter=1000, tol=1e-3,
                      random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10),
        {"alpha": [0.0001, 0.001, 0.01, 0.1]},
        cv=cv5, scoring="f1_macro", n_jobs=1, verbose=1, refit=True
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best alpha: {search.best_params_['alpha']} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")
    res, cm = evaluate("SGD L2", search.best_estimator_,
                       X_test_sc, y_test_r, X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    res["Best Params"] = str(search.best_params_)
    all_results.append(res)
    plot_coefficients(search.best_estimator_, features, "SGD L2", OUTPUT_DIR)
    plot_cm(cm, "SGD L2", OUTPUT_DIR)
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "sgd_l2.pkl"))
    del X_sub, y_sub, search; gc.collect()

    # ── Model 2: SGD L1 ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Model 2: SGDClassifier — L1 (LASSO)")
    X_sub, y_sub = subsample(X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    search = GridSearchCV(
        SGDClassifier(loss="modified_huber", penalty="l1",
                      class_weight="balanced", max_iter=1000, tol=1e-3,
                      random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10),
        {"alpha": [0.0001, 0.001, 0.01, 0.1]},
        cv=cv5, scoring="f1_macro", n_jobs=1, verbose=1, refit=True
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best alpha: {search.best_params_['alpha']} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")
    res, cm = evaluate("SGD L1", search.best_estimator_,
                       X_test_sc, y_test_r, X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    res["Best Params"] = str(search.best_params_)
    all_results.append(res)
    plot_coefficients(search.best_estimator_, features, "SGD L1", OUTPUT_DIR)
    plot_cm(cm, "SGD L1", OUTPUT_DIR)
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "sgd_l1.pkl"))
    del X_sub, y_sub, search; gc.collect()

    # ── Model 3: SGD ElasticNet ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Model 3: SGDClassifier — ElasticNet (L1+L2)")
    X_sub, y_sub = subsample(X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    search = GridSearchCV(
        SGDClassifier(loss="modified_huber", penalty="elasticnet",
                      class_weight="balanced", max_iter=1000, tol=1e-3,
                      random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10),
        {"alpha": [0.0001, 0.001, 0.01], "l1_ratio": [0.15, 0.5, 0.85]},
        cv=cv5, scoring="f1_macro", n_jobs=1, verbose=1, refit=True
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best params: {search.best_params_} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")
    res, cm = evaluate("SGD ElasticNet", search.best_estimator_,
                       X_test_sc, y_test_r, X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    res["Best Params"] = str(search.best_params_)
    all_results.append(res)
    plot_coefficients(search.best_estimator_, features, "SGD ElasticNet", OUTPUT_DIR)
    plot_cm(cm, "SGD ElasticNet", OUTPUT_DIR)
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "sgd_elasticnet.pkl"))
    del X_sub, y_sub, search; gc.collect()

    # ── Model 4: Passive Aggressive ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Model 4: SGDClassifier — Passive Aggressive style (hinge loss)")
    X_sub, y_sub = subsample(X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    search = GridSearchCV(
        SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            class_weight="balanced", max_iter=1000, tol=1e-3,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10
        ),
        {"eta0": [0.001, 0.01, 0.1, 1.0]},
        cv=cv5, scoring="f1_macro", n_jobs=1, verbose=1, refit=True
    )

    # ── Model 5: HistGradientBoosting ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Model 5: HistGradientBoosting (non-linear, RAM-safe)")
    X_sub, y_sub = subsample(X_train, y_train, HGB_SUBSAMPLE)
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        {"max_iter": [200, 300], "max_depth": [5, 7, None],
         "learning_rate": [0.05, 0.1, 0.2], "min_samples_leaf": [20, 40],
         "l2_regularization": [0.0, 0.1]},
        n_iter=10, cv=cv5, scoring="f1_macro",
        n_jobs=1, verbose=1, random_state=42, refit=True
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best params: {search.best_params_} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")
    res, cm = evaluate("HistGradientBoosting", search.best_estimator_,
                       X_test, y_test_r, X_train, y_train, HGB_SUBSAMPLE)
    res["Best Params"] = str(search.best_params_)
    all_results.append(res)
    plot_cm(cm, "HistGradientBoosting", OUTPUT_DIR)
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "hgb.pkl"))

    print("\nCalculating Permutation Importance...")
    imp_sub_n = min(5_000, len(X_test))
    X_imp, _, y_imp, _ = train_test_split(
        X_test, y_test_r, train_size=imp_sub_n, stratify=y_test_r, random_state=42
    )
    perm_imp = permutation_importance(
        search.best_estimator_, X_imp, y_imp,
        n_repeats=3, random_state=42, n_jobs=1
    )
    feat_imp = pd.DataFrame({
        "Feature":    features,
        "Importance": perm_imp.importances_mean,
        "Std":        perm_imp.importances_std
    }).sort_values("Importance", ascending=False)
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, "hgb_feature_importance.csv"), index=False)
    print(f"\nTop 20 HGB features:")
    print(feat_imp.head(20).to_string(index=False))
    del X_sub, y_sub, X_imp, y_imp, search; gc.collect()

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict,  os.path.join(MODEL_DIR, "label_encoders.pkl"))

    results_df = pd.DataFrame(all_results).sort_values("Test F1 Macro", ascending=False)
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON (ranked by macro F1)")
    print(f"Baseline (majority class): {baseline:.4f}")
    print(results_df[[
        "Model", "Test Accuracy", "Test F1 Macro",
        "Test F1 Weighted", "CV F1 Macro", "CV F1 Std"
    ]].to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "all_model_results.csv"), index=False)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"All models saved to:  {MODEL_DIR}")