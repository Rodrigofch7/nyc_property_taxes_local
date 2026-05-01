"""
non_linear_ml_models.py
=======================
Trains a LightGBM classifier to classify NYC properties as
undervalued, fairly_valued, or overvalued.

Imports shared logic from:
  feature_engineering.py    – load_data, engineer_features, prepare_xy, subsample
  hyperparameter_tuning.py  – tune_lgbm
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

HGB_SUBSAMPLE = 300_000


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
        kind="barh", x="Feature", y="Importance", ax=ax, legend=False, color="#4575b4",
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
    df = load_data(DATA_PATH)
    df, features, le_dict = engineer_features(df)
    X, y = prepare_xy(df, features)
    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── Subsample for tuning ──────────────────────────────────────────────────
    print(f"\nSubsampling to {HGB_SUBSAMPLE:,} rows for hyperparameter search...")
    X_sub, y_sub = subsample(X_train, y_train, HGB_SUBSAMPLE)

    # ── Tune ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nModel: LightGBM")
    base_estimator = LGBMClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    result = tune_lgbm(base_estimator, X_sub, y_sub, n_iter=10, cv=3, model_key="lgbm")
    del X_sub, y_sub
    gc.collect()

    # tune_lgbm returns a fitted estimator when loading from cache,
    # or a search object when CV was run
    if hasattr(result, "best_estimator_"):
        best_model  = result.best_estimator_
        best_params = result.best_params_
        cv_score    = result.best_score_
    else:
        best_model  = result
        best_params = result.get_params()
        cv_score    = None

    # ── Evaluate ──────────────────────────────────────────────────────────────
    acc, f1m, f1w, cm = evaluate(best_model, X_test, y_test)
    plot_confusion_matrix(cm, OUTPUT_DIR)
    plot_feature_importance(best_model, features, OUTPUT_DIR)

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(best_model, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    joblib.dump(features,   os.path.join(MODEL_DIR, "features.pkl"))
    joblib.dump(le_dict,    os.path.join(MODEL_DIR, "label_encoders.pkl"))

    print(f"\nBest params : {best_params}")
    if cv_score:
        print(f"CV F1 Macro : {cv_score:.4f}")
    print(f"\nOutputs saved to {OUTPUT_DIR}")
    print(f"Models saved to  {MODEL_DIR}")
