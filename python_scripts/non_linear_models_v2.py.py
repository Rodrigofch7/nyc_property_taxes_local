"""
non_linear_ml_models.py
Trains a non-linear classifier (LightGBM) to classify
NYC properties as undervalued, fairly_valued, or overvalued.
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
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
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
    n = len(col_list)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    x_c = x - x_mean

    Y = df[col_list].fillna(0).to_numpy(dtype=np.float64)

    denom  = float(x_c @ x_c)
    slopes = (x_c @ Y.T) / denom
    intercepts = Y.mean(axis=1)

    x_next     = n - x_mean
    projected  = intercepts + slopes * x_next
    projected  = np.clip(projected, 0, None)

    last_known = Y[:, -1]
    ratio = np.where(last_known > 0, projected / last_known, 1.0)
    ratio = np.clip(ratio, 0, 5)
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

    assess_per_sqft = df["FINACTTOT_FY2025"].fillna(0) / df["GROSS_SQFT"].clip(lower=1)
    new_cols["ASSESS_PER_SQFT"]     = assess_per_sqft
    new_cols["LOG_ASSESS_PER_SQFT"] = np.log1p(assess_per_sqft)
    new_cols["LAND_TO_TOTAL"]       = (df["FINACTLAND_FY2025"].fillna(0) / df["FINACTTOT_FY2025"].clip(lower=1)).clip(0, 1)
    mkt_to_assess                   = (df["FINMKTTOT_FY2025"].fillna(0)  / df["FINACTTOT_FY2025"].clip(lower=1)).clip(0, 20)
    new_cols["MKT_TO_ASSESS"]       = mkt_to_assess
    new_cols["LOG_MKT_TO_ASSESS"]   = np.log1p(mkt_to_assess)

    finacttot_cols  = [f"FINACTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINACTTOT_FY{y}"  in df.columns]
    finactland_cols = [f"FINACTLAND_FY{y}" for y in HISTORICAL_YEARS if f"FINACTLAND_FY{y}" in df.columns]
    finmkttot_cols  = [f"FINMKTTOT_FY{y}"  for y in HISTORICAL_YEARS if f"FINMKTTOT_FY{y}"  in df.columns]

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
            new_cols[name] = ((df[cols[i]].fillna(0) - df[cols[i-1]].fillna(0)) / df[cols[i-1]].fillna(1).clip(lower=1)).clip(-1, 5)
            names.append(name)
        return names

    yoy_cols      = yoy_changes(finacttot_cols,  "ASSESS_YOY")
    yoy_land_cols = yoy_changes(finactland_cols, "LAND_YOY")
    yoy_mkt_cols  = yoy_changes(finmkttot_cols,  "MKT_YOY")

    new_cols["ASSESS_VOLATILITY"] = pd.DataFrame({k: new_cols[k] for k in yoy_cols}).std(axis=1).fillna(0)
    new_cols["ASSESS_TREND"] = ((df[finacttot_cols[-1]].fillna(0) - df[finacttot_cols[0]].fillna(0)) / df[finacttot_cols[0]].fillna(1).clip(lower=1)).clip(-1, 10) if len(finacttot_cols) >= 2 else 0.0

    proj_feature_names = []
    for col_list, series_name in [(finacttot_cols, "FINACTTOT"), (finactland_cols, "FINACTLAND"), (finmkttot_cols, "FINMKTTOT")]:
        if len(col_list) >= 2:
            proj, ratio, resid = project_next_year(df, col_list, series_name)
            new_cols[proj.name]  = proj
            new_cols[ratio.name] = ratio
            new_cols[resid.name] = resid
            proj_feature_names += [proj.name, ratio.name, resid.name]

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
    del new_cols, new_df; gc.collect()

    historical_status_cols = [c for c in df.columns if any(str(yr) in c for yr in HISTORICAL_YEARS) and any(x in c for x in ["overvalued", "undervalued", "fairly_valued"])]

    features = encoded_cat_cols + ["LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS", "UNITS", "COOP_APTS", "BLD_STORY", "LOT_AREA", "BUILDING_AGE", "LOG_PYACTTOT", "LOG_ASSESS_PER_SQFT", "LAND_TO_TOTAL", "MKT_TO_ASSESS", "ASSESS_TREND", "ASSESS_VOLATILITY"] + proj_feature_names + historical_status_cols + log_acttot_cols + log_actland_cols + log_mkttot_cols + yoy_cols + yoy_land_cols + yoy_mkt_cols
    features = [f for f in features if f in df.columns]

    return df, features, le_dict


def prepare_xy(df, features, target_col="target_2026"):
    X = df[features].copy()
    y = df[target_col].astype(str)
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X, y


def subsample(X, y, n, seed=42):
    if len(X) <= n:
        return X, np.asarray(y)          # X is already a DataFrame here
    rng = np.random.default_rng(seed)
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
    return X.iloc[idx], y_arr[idx]       # .iloc keeps column names intact
 

# ── Execution ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df, features, le_dict = engineer_features(df)
    X, y = prepare_xy(df, features)
    del df; gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print(f"\n{'='*60}\nModel: LightGBM")
    
    # Define CV and Subsample
    cv5 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X_sub, y_sub = subsample(X_train, y_train, HGB_SUBSAMPLE)

    search = RandomizedSearchCV(
        LGBMClassifier(random_state=42, class_weight="balanced", n_jobs=-1, verbose=-1),
        {
            "n_estimators": [300, 500],
            "max_depth": [5, 9, -1],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 127],
            "reg_lambda": [0.1, 1.0]
        },
        n_iter=10, cv=cv5, scoring="f1_macro", n_jobs=1, verbose=0, random_state=42, refit=True
    )

    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best params : {search.best_params_}")
    print(f"  CV F1 Macro : {search.best_score_:.4f}")
    print(f"  Time        : {time.time()-t0:.0f}s")

    # Final Evaluation
    y_pred = search.best_estimator_.predict(X_test)
    print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 Macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["undervalued", "fairly_valued", "overvalued"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["undervalued", "fairly_valued", "overvalued"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    plt.title("LightGBM — Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "lgbm_confusion_matrix.png"))
    plt.close()

    # Feature Importance (LGBM Built-in)
    feat_imp = pd.DataFrame({"Feature": features, "Importance": search.best_estimator_.feature_importances_}).sort_values("Importance", ascending=False)
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv"), index=False)
    print(f"\nTop 15 Features:\n{feat_imp.head(15).to_string(index=False)}")

    # Save Artifacts
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    print(f"\nOutputs saved to {OUTPUT_DIR}")