"""
non_linear_ml_models.py
=======================
Trains a LightGBM classifier to classify NYC properties as
undervalued, fairly_valued, or overvalued.

Features:
  - Progress bar via tqdm
  - Checkpointing: saves best params after every CV iteration
  - Resume: if interrupted, re-run and it picks up where it left off
  - Set FORCE_RETUNE = True to wipe cache and start fresh

Imports shared logic from:
  feature_engineering.py   – load_data, engineer_features, prepare_xy, subsample
  hyperparameter_tuning.py – tune_lgbm
"""

import os
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from lightgbm import LGBMClassifier
import joblib

from feature_engineering import load_data, engineer_features, prepare_xy, subsample
from hyperparameter_tuning import tune_lgbm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
MODEL_DIR  = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
OUTPUT_DIR = "/home/rodrigofrancachaves/project-nyc_property_taxes/outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSAMPLE_SIZE = 300_000

# ── Config ────────────────────────────────────────────────────────────────────
# n_iter: number of hyperparameter combinations to try
# Each combo = cv fits of LightGBM on 300k rows
# At ~5-8 min per combo: 20 combos ≈ 2-3 hours
N_ITER      = 20
CV_FOLDS    = 3

# Set True to wipe cache and start fresh
FORCE_RETUNE = False


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1m    = f1_score(y_test, y_pred, average="macro")
    f1w    = f1_score(y_test, y_pred, average="weighted")

    print(f"\nTest Accuracy    : {acc:.4f}")
    print(f"Test F1 Macro    : {f1m:.4f}  ← primary metric")
    print(f"Test F1 Weighted : {f1w:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred, labels=["undervalued", "fairly_valued", "overvalued"])
    return acc, f1m, f1w, cm


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, output_dir):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["undervalued", "fairly_valued", "overvalued"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    plt.title("LightGBM — Confusion Matrix")
    out = os.path.join(output_dir, "lgbm_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {out}")


def plot_feature_importance(model, features, output_dir, top_n=20):
    feat_imp = (
        pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
    )
    feat_imp.to_csv(os.path.join(output_dir, "lgbm_feature_importance.csv"), index=False)
    print(f"\nTop {top_n} Features:\n{feat_imp.head(top_n).to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.head(top_n).plot(
        kind="barh", x="Feature", y="Importance",
        ax=ax, legend=False, color="#4575b4",
    )
    ax.invert_yaxis()
    ax.set_title(f"LightGBM — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    out = os.path.join(output_dir, "lgbm_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load and engineer features
    df = load_data(DATA_PATH)
    df, features, le_dict = engineer_features(df)
    X, y = prepare_xy(df, features)
    del df
    gc.collect()

    print(f"\nFeature count: {len(features)}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # 3. Subsample for tuning (RAM-safe)
    print(f"\nSubsampling to {SUBSAMPLE_SIZE:,} rows for hyperparameter search...")
    X_sub, y_sub = subsample(X_train, y_train, SUBSAMPLE_SIZE)
    print(f"  Subsample shape: {X_sub.shape}")

    # 4. Tune with checkpointing
    print(f"\n{'='*60}")
    print(f"Model: LightGBM  |  n_iter={N_ITER}  |  cv={CV_FOLDS}-fold")
    print(f"Checkpoint saves after every iteration — safe to Ctrl+C and resume")
    print(f"{'='*60}")

    estimator = LGBMClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )

    best_model = tune_lgbm(
        estimator, X_sub, y_sub,
        model_key="lgbm",
        n_iter=N_ITER,
        cv=CV_FOLDS,
        force_retune=FORCE_RETUNE,
    )
    del X_sub, y_sub
    gc.collect()

    # 5. Evaluate on full test set
    print(f"\n{'='*60}")
    print("Evaluating on full test set...")
    acc, f1m, f1w, cm = evaluate(best_model, X_test, y_test)
    plot_confusion_matrix(cm, OUTPUT_DIR)
    plot_feature_importance(best_model, features, OUTPUT_DIR)

    # 6. Save artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    joblib.dump(features,   os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict,    os.path.join(MODEL_DIR, "label_encoders.pkl"))

    print(f"\n{'='*60}")
    print(f"Test Accuracy    : {acc:.4f}")
    print(f"Test F1 Macro    : {f1m:.4f}")
    print(f"Test F1 Weighted : {f1w:.4f}")
    print(f"\nOutputs saved to : {OUTPUT_DIR}")
    print(f"Models saved to  : {MODEL_DIR}")
    print(f"\nTo re-tune from scratch: set FORCE_RETUNE = True")
    print(f"To resume interrupted run: just re-run the script as-is")