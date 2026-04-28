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
DATA_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data_census.parquet"
MODEL_DIR  = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
OUTPUT_DIR = "/home/rodrigofrancachaves/project-nyc_property_taxes/outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LINEAR_SUBSAMPLE = 100_000
HGB_SUBSAMPLE    = 300_000


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
    numeric_cols = (
        ["GROSS_SQFT", "LAND_AREA", "NUM_BLDGS", "YRBUILT",
         "UNITS", "COOP_APTS", "BLD_STORY", "LOT_FRT", "LOT_DEP",
         "FINACTTOT", "PYACTTOT",
         # census columns
         "MEDIAN_INCOME", "LOG_MEDIAN_INCOME", "MEDIAN_RENT",
         "MEDIAN_HOME_VALUE", "PCT_WHITE", "PCT_OWNER_OCCUPIED",
         "PCT_POVERTY", "PCT_BACHELORS", "TOTAL_POPULATION"] +
        [f"FINACTTOT_FY{y}" for y in [2020, 2021, 2022, 2023, 2024, 2025]]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Property features ─────────────────────────────────────────────────────
    df["BUILDING_AGE"]   = (2026 - df["YRBUILT"]).clip(lower=0, upper=200)
    df["LOG_GROSS_SQFT"] = np.log1p(df["GROSS_SQFT"].fillna(0))
    df["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"].fillna(0))
    df["LOG_PYACTTOT"]   = np.log1p(df["PYACTTOT"].fillna(0))

    # ── Historical assessment log transforms ──────────────────────────────────
    historical_value_cols = [c for c in df.columns if "FINACTTOT_FY" in c]
    for col in historical_value_cols:
        df[f"LOG_{col}"] = np.log1p(df[col].fillna(0))
    log_value_cols = [f"LOG_{c}" for c in historical_value_cols if c in df.columns]

    # ── Assessment trend ──────────────────────────────────────────────────────
    fy_available = sorted([c for c in historical_value_cols if c in df.columns])
    if len(fy_available) >= 2:
        df["ASSESS_TREND"] = (
            (df[fy_available[-1]].fillna(0) - df[fy_available[0]].fillna(0)) /
            df[fy_available[0]].fillna(1).clip(lower=1)
        ).clip(-1, 10)
    else:
        df["ASSESS_TREND"] = 0.0

    # ── Census log transforms ─────────────────────────────────────────────────
    df["LOG_MEDIAN_RENT"]       = np.log1p(df["MEDIAN_RENT"].fillna(0))
    df["LOG_MEDIAN_HOME_VALUE"] = np.log1p(df["MEDIAN_HOME_VALUE"].fillna(0))
    df["LOG_TOTAL_POPULATION"]  = np.log1p(df["TOTAL_POPULATION"].fillna(0))

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

    # ── Historical classification status columns ───────────────────────────────
    historical_status_cols = [
        c for c in df.columns
        if any(str(yr) in c for yr in [2020, 2021, 2022, 2023, 2024, 2025])
        and any(x in c for x in ["overvalued", "undervalued", "fairly_valued"])
    ]

    # ── Census features ───────────────────────────────────────────────────────
    census_features = [
        "LOG_MEDIAN_INCOME",        # income level of ZIP area
        "LOG_MEDIAN_RENT",          # rent level
        "LOG_MEDIAN_HOME_VALUE",    # home value context
        "PCT_WHITE",                # racial composition
        "PCT_OWNER_OCCUPIED",       # owner vs renter
        "PCT_POVERTY",              # poverty rate
        "PCT_BACHELORS",            # education level
        "LOG_TOTAL_POPULATION",     # density proxy
    ]
    census_features = [f for f in census_features if f in df.columns]

    # ── Final feature list ────────────────────────────────────────────────────
    features = (
        encoded_cat_cols +
        [
            "LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS",
            "UNITS", "COOP_APTS", "BLD_STORY",
            "LOT_FRT", "LOT_DEP", "BUILDING_AGE",
            "LOG_PYACTTOT", "ASSESS_TREND",
        ] +
        historical_status_cols +
        log_value_cols +
        census_features
    )
    features = [f for f in features if f in df.columns]
    print(f"  Total features: {len(features)}")
    print(f"  Census features included: {census_features}")
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
        "Model": name,
        "Test Accuracy": round(acc, 4),
        "Test F1 Macro": round(f1m, 4),
        "Test F1 Weighted": round(f1w, 4),
        "CV F1 Macro": round(cv_scores.mean(), 4),
        "CV F1 Std": round(cv_scores.std(), 4)
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

    # 1. Load and prepare
    df = load_data(DATA_PATH)
    df, features, le_dict = engineer_features(df)
    X, y = prepare_xy(df, features)
    del df; gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # Scale for linear models
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
        SGDClassifier(
            loss="modified_huber", penalty="l2",
            class_weight="balanced", max_iter=1000, tol=1e-3,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10
        ),
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
        SGDClassifier(
            loss="modified_huber", penalty="l1",
            class_weight="balanced", max_iter=1000, tol=1e-3,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10
        ),
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
        SGDClassifier(
            loss="modified_huber", penalty="elasticnet",
            class_weight="balanced", max_iter=1000, tol=1e-3,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10
        ),
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
    print("Model 4: Passive Aggressive Classifier (linear)")
    X_sub, y_sub = subsample(X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    search = GridSearchCV(
        PassiveAggressiveClassifier(
            class_weight="balanced", max_iter=1000, tol=1e-3,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10
        ),
        {"C": [0.001, 0.01, 0.1, 1.0]},
        cv=cv5, scoring="f1_macro", n_jobs=1, verbose=1, refit=True
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  Best C: {search.best_params_['C']} | CV F1: {search.best_score_:.4f} | {time.time()-t0:.0f}s")
    res, cm = evaluate("Passive Aggressive", search.best_estimator_,
                       X_test_sc, y_test_r, X_train_sc, y_train_r, LINEAR_SUBSAMPLE)
    res["Best Params"] = str(search.best_params_)
    all_results.append(res)
    plot_coefficients(search.best_estimator_, features, "Passive Aggressive", OUTPUT_DIR)
    plot_cm(cm, "Passive Aggressive", OUTPUT_DIR)
    joblib.dump(search.best_estimator_, os.path.join(MODEL_DIR, "passive_aggressive.pkl"))
    del X_sub, y_sub, search; gc.collect()

    # ── Model 5: HistGradientBoosting (non-linear) ────────────────────────────
    print(f"\n{'='*60}")
    print("Model 5: HistGradientBoosting (non-linear, RAM-safe)")
    X_sub, y_sub = subsample(X_train, y_train, HGB_SUBSAMPLE)
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "max_iter":          [200, 300],
            "max_depth":         [5, 7, None],
            "learning_rate":     [0.05, 0.1, 0.2],
            "min_samples_leaf":  [20, 40],
            "l2_regularization": [0.0, 0.1],
        },
        n_iter=10,
        cv=cv5,
        scoring="f1_macro",
        n_jobs=1,
        verbose=1,
        random_state=42,
        refit=True
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

    # Permutation importance for HGB
    print("\nCalculating Permutation Importance...")
    imp_sub_n = min(20_000, len(X_test))
    X_imp, _, y_imp, _ = train_test_split(
        X_test, y_test_r,
        train_size=imp_sub_n, stratify=y_test_r, random_state=42
    )
    perm_imp = permutation_importance(
        search.best_estimator_, X_imp, y_imp,
        n_repeats=5, random_state=42, n_jobs=-1
    )
    feat_imp = pd.DataFrame({
        "Feature":    features,
        "Importance": perm_imp.importances_mean,
        "Std":        perm_imp.importances_std
    }).sort_values("Importance", ascending=False)
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, "hgb_feature_importance.csv"), index=False)
    print(f"\nTop 15 HGB features (Permutation Importance):")
    print(feat_imp.head(15).to_string(index=False))
    del X_sub, y_sub, X_imp, y_imp, search; gc.collect()

    # ── Save shared artifacts ─────────────────────────────────────────────────
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict,  os.path.join(MODEL_DIR, "label_encoders.pkl"))

    # ── Final summary ─────────────────────────────────────────────────────────
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