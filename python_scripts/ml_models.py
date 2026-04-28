"""
linear_model.py
Trains a linear classifier (SGDClassifier with L2 regularization) to classify
NYC properties as undervalued, fairly_valued, or overvalued based on
peer-group assessment ratios.

Model choice:   SGDClassifier with modified_huber loss (linear model)
Regularizer:    L2 (Ridge) — chosen because features are correlated across
                years; L2 shrinks all coefficients proportionally rather than
                zeroing some out arbitrarily (as L1 would), giving stable
                interpretable weights.
Alpha:          Selected via cross-validation over a log-scale grid,
                optimizing macro F1.
"""

import pandas as pd
import numpy as np
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
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

# Subsample size — 100k is plenty for a linear model
SUBSAMPLE = 200_000


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(path):
    """Load processed labeled dataset and drop unknown targets."""
    print("Loading data...")
    df = pd.read_parquet(path)
    print(f"  Loaded shape: {df.shape}")
    df = df[df["target_2026"] != "unknown"].copy()
    print(f"  Shape after dropping unknown targets: {df.shape}")
    print(f"\nTarget distribution:\n{df['target_2026'].value_counts()}")
    print(f"\nTarget proportions:\n{df['target_2026'].value_counts(normalize=True).round(3)}")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    """
    Build feature matrix from structural, geographic, and
    historical assessment columns. No sale price used.
    """
    print("\nEngineering features...")

    numeric_cols = (
        ["GROSS_SQFT", "LAND_AREA", "NUM_BLDGS", "YRBUILT",
         "UNITS", "COOP_APTS", "BLD_STORY", "LOT_FRT", "LOT_DEP",
         "FINACTTOT", "PYACTTOT"] +
        [f"FINACTTOT_FY{y}" for y in [2020, 2021, 2022, 2023, 2024, 2025]]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Building age
    df["BUILDING_AGE"] = (2026 - df["YRBUILT"]).clip(lower=0, upper=200)

    # Log-transform skewed variables
    df["LOG_GROSS_SQFT"] = np.log1p(df["GROSS_SQFT"].fillna(0))
    df["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"].fillna(0))
    df["LOG_PYACTTOT"]   = np.log1p(df["PYACTTOT"].fillna(0))

    # Log-transform historical assessed values
    historical_value_cols = [c for c in df.columns if "FINACTTOT_FY" in c]
    for col in historical_value_cols:
        df[f"LOG_{col}"] = np.log1p(df[col].fillna(0))
    log_value_cols = [f"LOG_{c}" for c in historical_value_cols if c in df.columns]

    # Assessment trend
    fy_available = sorted([c for c in historical_value_cols if c in df.columns])
    if len(fy_available) >= 2:
        df["ASSESS_TREND"] = (
            (df[fy_available[-1]].fillna(0) - df[fy_available[0]].fillna(0)) /
            df[fy_available[0]].fillna(1).clip(lower=1)
        ).clip(-1, 10)
    else:
        df["ASSESS_TREND"] = 0.0

    # Encode categoricals
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

    # Historical classification status columns
    historical_status_cols = [
        c for c in df.columns
        if any(str(yr) in c for yr in [2020, 2021, 2022, 2023, 2024, 2025])
        and any(x in c for x in ["overvalued", "undervalued", "fairly_valued"])
    ]

    features = (
        encoded_cat_cols +
        [
            "LOG_GROSS_SQFT", "LOG_LAND_AREA", "NUM_BLDGS",
            "UNITS", "COOP_APTS", "BLD_STORY",
            "LOT_FRT", "LOT_DEP", "BUILDING_AGE",
            "LOG_PYACTTOT", "ASSESS_TREND",
        ] +
        historical_status_cols +
        log_value_cols
    )
    features = [f for f in features if f in df.columns]
    print(f"  Total features: {len(features)}")
    return df, features, le_dict


# ── Prepare X and y ───────────────────────────────────────────────────────────
def prepare_xy(df, features, target_col="target_2026"):
    """Build X matrix and y vector, impute missing with median."""
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


# ── Train SGD classifier ──────────────────────────────────────────────────────
def train_sgd(X_train, y_train):
    """
    Train SGDClassifier with modified_huber loss and L2 regularization.
    Tunes alpha via GridSearchCV optimizing macro F1.
    Subsamples to SUBSAMPLE rows for speed.
    """
    # Subsample
    if len(X_train) > SUBSAMPLE:
        print(f"\nSubsampling to {SUBSAMPLE:,} rows...")
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=SUBSAMPLE,
            random_state=42,
            stratify=y_train
        )
        print(f"  Subsample class distribution:\n{y_sub.value_counts()}")
    else:
        X_sub, y_sub = X_train, y_train

    print(f"\nTuning SGDClassifier (L2)...")
    print(f"  Searching alpha in: [0.0001, 0.001, 0.01, 0.1, 1.0]")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_model = SGDClassifier(
        loss="modified_huber",    # smooth hinge — gives probabilities
        penalty="l2",             # L2 regularization
        class_weight="balanced",  # handles class imbalance
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=-1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}

    search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        verbose=2,
        refit=True,
    )

    start = time.time()
    search.fit(X_sub, y_sub)
    elapsed = time.time() - start

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Best alpha : {search.best_params_['alpha']}")
    print(f"  Best CV F1 macro: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_test):
    """Compute test metrics and cross-validation F1."""
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Test Accuracy    : {acc:.4f}")
    print(f"  Test F1 Macro    : {f1m:.4f}  ← primary metric")
    print(f"  Test F1 Weighted : {f1w:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(
        y_test, y_pred,
        labels=["undervalued", "fairly_valued", "overvalued"]
    )
    print(f"Confusion Matrix:\n{cm}")

    # CV on subsample
    print("\nRunning cross-validation (subsampled)...")
    if len(X_train) > SUBSAMPLE:
        X_cv, _, y_cv, _ = train_test_split(
            X_train, y_train,
            train_size=SUBSAMPLE,
            random_state=99,
            stratify=y_train
        )
    else:
        X_cv, y_cv = X_train, y_train

    cv_scores = cross_val_score(
        model, X_cv, y_cv,
        cv=StratifiedKFold(5, shuffle=True, random_state=99),
        scoring="f1_macro", n_jobs=-1
    )
    print(f"  CV F1 Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return acc, f1m, f1w, cv_scores.mean(), cv_scores.std(), cm


# ── Plot coefficients ─────────────────────────────────────────────────────────
def plot_coefficients(model, features, output_dir, top_n=20):
    """Plot and save top-N coefficients per class."""
    print("\nPlotting coefficients...")
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

    plt.suptitle(
        "SGD Linear Classifier (L2) Coefficients by Class\n"
        "Red = positive (increases probability), Blue = negative",
        fontsize=13
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "linear_model_coefficients.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Coefficient plot saved to: {out_path}")

    coef_out = os.path.join(output_dir, "linear_model_coefficients.csv")
    coef_df.T.reset_index().rename(columns={"index": "feature"}).to_csv(
        coef_out, index=False
    )
    print(f"  Coefficient table saved to: {coef_out}")
    return coef_df


# ── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm, output_dir):
    """Save confusion matrix plot."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["undervalued", "fairly_valued", "overvalued"]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("SGD Linear Classifier — Confusion Matrix (Test Set)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "linear_model_confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Feature engineering
    df, features, le_dict = engineer_features(df)

    # 3. Prepare X and y
    X, y = prepare_xy(df, features)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # 5. Scale (required for SGD)
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=features, index=X_test.index
    )
    y_train = y_train.reset_index(drop=True)
    X_train_scaled = X_train_scaled.reset_index(drop=True)

    # 6. Train with alpha tuning
    model, best_params, best_cv_score = train_sgd(X_train_scaled, y_train)

    # 7. Evaluate
    baseline = y.value_counts(normalize=True).max()
    print(f"\nBaseline (majority class): {baseline:.4f}")
    acc, f1m, f1w, cv_mean, cv_std, cm = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_test
    )

    # 8. Plots
    coef_df = plot_coefficients(model, features, OUTPUT_DIR, top_n=20)
    plot_confusion_matrix(cm, OUTPUT_DIR)

    # 9. Save
    joblib.dump(model,    os.path.join(MODEL_DIR, "sgd_linear_classifier.pkl"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict,  os.path.join(MODEL_DIR, "label_encoders.pkl"))

    # 10. Summary
    print(f"\n{'='*60}")
    print("SGD LINEAR CLASSIFIER SUMMARY")
    print(f"{'='*60}")
    print(f"Loss             : modified_huber (smooth hinge)")
    print(f"Regularization   : L2 (Ridge)")
    print(f"Best alpha       : {best_params['alpha']}")
    print(f"Best CV F1 macro : {best_cv_score:.4f}")
    print(f"Test Accuracy    : {acc:.4f}")
    print(f"Test F1 Macro    : {f1m:.4f}")
    print(f"Test F1 Weighted : {f1w:.4f}")
    print(f"CV F1 Macro      : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Baseline         : {baseline:.4f}")
    print(f"\nTop 5 features per class:")
    for cls in model.classes_:
        top5 = coef_df.loc[cls].abs().nlargest(5).index.tolist()
        print(f"  {cls}: {top5}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Models saved to:      {MODEL_DIR}")