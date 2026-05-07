"""
linear_ml_models.py
===================
Trains SGDClassifier variants (L2, L1, ElasticNet) to classify
NYC properties as undervalued, fairly_valued, or overvalued.

Imports shared logic from:
  feature_engineering.py    – load_data, engineer_features, prepare_xy, subsample
  hyperparameter_tuning.py  – tune_sgd
"""

import os
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import joblib

from feature_engineering import load_data, engineer_features, prepare_xy, subsample
from hyperparameter_tuning import tune_sgd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
MODEL_DIR  = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
OUTPUT_DIR = "/home/rodrigofrancachaves/project-nyc_property_taxes/outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LINEAR_SUBSAMPLE = 100_000

# Options:
# False   → use cached params, skip CV (normal runs)
# True    → wipe cache, run fresh CV, save results regardless
# "safe"  → run fresh CV, only update cache if new params are better
FORCE_RETUNE = "safe"


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(name, model, X_test, y_test, X_train, y_train, subsample_n):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1m    = f1_score(y_test, y_pred, average="macro")
    f1w    = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Test Accuracy    : {acc:.4f}")
    print(f"  Test F1 Macro    : {f1m:.4f}  ← primary metric")
    print(f"  Test F1 Weighted : {f1w:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred, labels=["undervalued", "fairly_valued", "overvalued"])
    print(f"Confusion Matrix:\n{cm}")

    X_cv, y_cv = subsample(X_train, y_train, subsample_n, seed=99)
    cv_scores  = cross_val_score(
        model, X_cv, y_cv,
        cv=StratifiedKFold(10, shuffle=True, random_state=99),
        scoring="f1_macro", n_jobs=-1,
    )
    print(f"  CV F1 Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "Model":            name,
        "Test Accuracy":    round(acc, 4),
        "Test F1 Macro":    round(f1m, 4),
        "Test F1 Weighted": round(f1w, 4),
        "CV F1 Macro":      round(cv_scores.mean(), 4),
        "CV F1 Std":        round(cv_scores.std(),  4),
    }, cm


# ── Plot helpers ──────────────────────────────────────────────────────────────
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


def plot_cm(cm, name, output_dir):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["undervalued", "fairly_valued", "overvalued"],
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


# ── Train one SGD model ───────────────────────────────────────────────────────
def train_sgd(name, penalty, extra_params,
              X_train_sc, y_train_r, X_test_sc, y_test_r,
              features, cv5, output_dir, model_dir, force_retune=False):
    print(f"\n{'='*60}\nModel: SGDClassifier — {name}")

    X_sub, y_sub = subsample(X_train_sc, y_train_r, LINEAR_SUBSAMPLE)

    base_estimator = SGDClassifier(
        loss="modified_huber",
        penalty=penalty,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        **extra_params,
    )

    result = tune_sgd(base_estimator, X_sub, y_sub, penalty=penalty, cv=5,
                      model_key=f"sgd_{penalty.lower()}", force_retune=force_retune)
    del X_sub, y_sub
    gc.collect()

    # tune_sgd returns a fitted estimator when loading from cache,
    # or a search object when CV was run
    if hasattr(result, "best_estimator_"):
        best_model  = result.best_estimator_
        best_params = result.best_params_
    else:
        best_model  = result
        best_params = result.get_params()

    res, cm = evaluate(
        f"SGD {name}", best_model,
        X_test_sc, y_test_r, X_train_sc, y_train_r, LINEAR_SUBSAMPLE,
    )
    res["Best Params"] = str(best_params)

    plot_coefficients(best_model, features, f"SGD {name}", output_dir)
    plot_cm(cm, f"SGD {name}", output_dir)

    slug = name.lower().replace(" ", "_")
    joblib.dump(best_model, os.path.join(model_dir, f"sgd_{slug}.pkl"))
    gc.collect()
    return res, best_model


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

    print("\nScaling features...")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    y_train_r  = y_train.to_numpy()
    y_test_r   = y_test.to_numpy()
    del X_train, X_test, y_train, y_test, X, y
    gc.collect()

    baseline = pd.Series(y_train_r).value_counts(normalize=True).max()
    print(f"\nBaseline (majority class): {baseline:.4f}")

    cv5         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    res, _ = train_sgd("L2", "l2", {},
                       X_train_sc, y_train_r, X_test_sc, y_test_r,
                       features, cv5, OUTPUT_DIR, MODEL_DIR,
                       force_retune=FORCE_RETUNE)
    all_results.append(res)

    res, _ = train_sgd("L1", "l1", {},
                       X_train_sc, y_train_r, X_test_sc, y_test_r,
                       features, cv5, OUTPUT_DIR, MODEL_DIR,
                       force_retune=FORCE_RETUNE)
    all_results.append(res)

    res, _ = train_sgd("ElasticNet", "elasticnet", {"l1_ratio": 0.85},
                       X_train_sc, y_train_r, X_test_sc, y_test_r,
                       features, cv5, OUTPUT_DIR, MODEL_DIR,
                       force_retune=FORCE_RETUNE)
    all_results.append(res)

    # ── Save shared artifacts ─────────────────────────────────────────────────
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "linear_features.pkl"))
    joblib.dump(le_dict,  os.path.join(MODEL_DIR, "label_encoders.pkl"))

    results_df = pd.DataFrame(all_results).sort_values("Test F1 Macro", ascending=False)
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON (ranked by macro F1)")
    print(f"Baseline (majority class): {baseline:.4f}")
    print(results_df[[
        "Model", "Test Accuracy", "Test F1 Macro",
        "Test F1 Weighted", "CV F1 Macro", "CV F1 Std",
    ]].to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "linear_model_results.csv"), index=False)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"All models saved to:  {MODEL_DIR}")