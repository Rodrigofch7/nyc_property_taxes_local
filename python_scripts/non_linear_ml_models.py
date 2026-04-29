"""
ml_models_nonlinear.py
Trains a non-linear classifier (HistGradientBoosting) to classify 
NYC properties as undervalued, fairly_valued, or overvalued.

RAM optimizations:
  - Feature engineering uses a single pd.concat() to prevent fragmentation.
  - Permutation importance uses a subsample to avoid OOM.
  - Explicit gc.collect() at boundaries.
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
    cross_val_score, RandomizedSearchCV
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


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    print("\nEngineering features...")

    # Numeric conversions
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

    # Basic property features
    new_cols["BUILDING_AGE"]    = (2026 - df["YRBUILT"]).clip(lower=0, upper=200)
    new_cols["LOG_GROSS_SQFT"]  = np.log1p(df["GROSS_SQFT"].fillna(0))
    new_cols["LOG_LAND_AREA"]   = np.log1p(df["LAND_AREA"].fillna(0))
    new_cols["LOG_PYACTTOT"]    = np.log1p(df["PYACTTOT"].fillna(0))
    new_cols["SQFT_PER_UNIT"]   = (df["GROSS_SQFT"] / df["UNITS"].clip(lower=1)).clip(upper=50000)
    new_cols["COVERAGE_RATIO"]  = (df["GROSS_SQFT"] / df["LAND_AREA"].clip(lower=1)).clip(upper=50)
    new_cols["LOT_AREA"]        = df["LOT_FRT"] * df["LOT_DEP"]
    new_cols["BUILDING_ERA"]    = pd.cut(
        df["YRBUILT"],
        bins=[0, 1900, 1940, 1960, 1980, 2000, 2010, 2030],
        labels=[1, 2, 3, 4, 5, 6, 7]
    ).astype(float).fillna(0)

    # Assessment features
    assess_per_sqft = df["FINACTTOT"].fillna(0) / df["GROSS_SQFT"].clip(lower=1)
    new_cols["ASSESS_PER_SQFT"]     = assess_per_sqft
    new_cols["LOG_ASSESS_PER_SQFT"] = np.log1p(assess_per_sqft)
    new_cols["LAND_TO_TOTAL"]       = (df["FINACTLAND"].fillna(0) / df["FINACTTOT"].clip(lower=1)).clip(0, 1)
    mkt_to_assess                   = (df["FINMKTTOT"].fillna(0) / df["FINACTTOT"].clip(lower=1)).clip(0, 20)
    new_cols["MKT_TO_ASSESS"]       = mkt_to_assess
    new_cols["LOG_MKT_TO_ASSESS"]   = np.log1p(mkt_to_assess)

    # Historical helper structures
    finacttot_cols  = [f"FINACTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINACTTOT_FY{y}"  in df.columns]
    finactland_cols = [f"FINACTLAND_FY{y}" for y in HISTORICAL_YEARS if f"FINACTLAND_FY{y}" in df.columns]
    finmkttot_cols  = [f"FINMKTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINMKTTOT_FY{y}" in df.columns]

    log_acttot_cols, log_actland_cols, log_mkttot_cols = [], [], []
    for col in finacttot_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_acttot_cols.append(name)
    for col in finactland_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_actland_cols.append(name)
    for col in finmkttot_cols:
        name = f"LOG_{col}"; new_cols[name] = np.log1p(df[col].fillna(0)); log_mkttot_cols.append(name)

    def yoy_changes(cols, prefix):
        names = []
        for i in range(1, len(cols)):
            name = f"{prefix}_FY{HISTORICAL_YEARS[i]}"
            new_cols[name] = ((df[cols[i]].fillna(0) - df[cols[i-1]].fillna(0)) / 
                              df[cols[i-1]].fillna(1).clip(lower=1)).clip(-1, 5)
            names.append(name)
        return names

    yoy_cols      = yoy_changes(finacttot_cols,  "ASSESS_YOY")
    yoy_land_cols = yoy_changes(finactland_cols, "LAND_YOY")
    yoy_mkt_cols  = yoy_changes(finmkttot_cols,  "MKT_YOY")

    # Trends and Volatility
    new_cols["ASSESS_VOLATILITY"] = pd.DataFrame({k: new_cols[k] for k in yoy_cols}).std(axis=1).fillna(0)
    new_cols["ASSESS_TREND"] = ((df[finacttot_cols[-1]].fillna(0) - df[finacttot_cols[0]].fillna(0)) / 
                                 df[finacttot_cols[0]].fillna(1).clip(lower=1)).clip(-1, 10) if len(finacttot_cols) >= 2 else 0.0

    # Categorical encoding
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

    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    del new_cols, new_df
    gc.collect()

    historical_status_cols = [c for c in df.columns if any(str(yr) in c for yr in HISTORICAL_YEARS) 
                              and any(x in c for x in ["overvalued", "undervalued", "fairly_valued"])]

    features = (encoded_cat_cols + ["LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS", "UNITS", "COOP_APTS", 
                                    "BLD_STORY", "LOT_AREA", "BUILDING_AGE", "LOG_PYACTTOT", "LOG_ASSESS_PER_SQFT", 
                                    "LAND_TO_TOTAL", "MKT_TO_ASSESS", "ASSESS_TREND", "ASSESS_VOLATILITY"] + 
                historical_status_cols + log_acttot_cols + log_actland_cols + log_mkttot_cols + 
                yoy_cols + yoy_land_cols + yoy_mkt_cols)
    
    features = [f for f in features if f in df.columns]
    return df, features, le_dict


# ── Prepare X and y ───────────────────────────────────────────────────────────
def prepare_xy(df, features, target_col="target_2026"):
    X = df[features].copy()
    y = df[target_col].astype(str)
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X, y


# ── Subsample ─────────────────────────────────────────────────────────────────
def subsample(X, y, n, seed=42):
    if len(X) <= n: return np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    idx = np.concatenate([rng.choice(np.where(y_arr == cls)[0], size=int(np.ceil((cnt/len(y))*n)), replace=False) 
                          for cls, cnt in zip(classes, counts)])
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print(f"\nModel: HistGradientBoosting (non-linear, RAM-safe)")
    X_sub, y_sub = subsample(X_train, y_train, HGB_SUBSAMPLE)
    
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        {"max_iter": [200, 300], "max_depth": [5, 7, None], 
         "learning_rate": [0.05, 0.1, 0.2], "l2_regularization": [0.0, 0.1]},
        n_iter=5, cv=cv5, scoring="f1_macro", n_jobs=1, verbose=1, random_state=42, refit=True
    )
    
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best params: {search.best_params_} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")

    # Evaluation - Convert to numpy to avoid feature name warnings
    y_pred = search.best_estimator_.predict(X_test.to_numpy())
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Feature Importance - Using Permutation to avoid constructor/versioning issues
    print("\nCalculating Permutation Importance (RAM-safe subsample)...")
    X_imp, y_imp = subsample(X_test, y_test, n=5000)
    r = permutation_importance(search.best_estimator_, X_imp, y_imp, n_repeats=5, random_state=42, n_jobs=-1)

    feat_imp = pd.DataFrame({
        "Feature": features, 
        "Importance": r.importances_mean
    }).sort_values("Importance", ascending=False)
    
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, "hgb_feature_importance.csv"), index=False)
    print(f"\nTop 10 features:\n{feat_imp.head(10).to_string(index=False)}")

    # Save
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "hgb_model.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    
    print(f"\nResults and model saved to {OUTPUT_DIR} and {MODEL_DIR}")