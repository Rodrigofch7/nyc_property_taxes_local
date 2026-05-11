"""
diagnostics.py
==============
Quick diagnostic checks for the NYC property tax classification pipeline.
Runs fast — no model training, just data and feature analysis.

Checks:
  1. Zero / near-zero importance features
  2. SGD L1 vs L2/ElasticNet performance gap
  3. Feature collinearity (for linear models)
  4. Prediction confidence distribution (borderline properties)
  5. Peer group labeling sanity check
  6. Sales feature match rate and usefulness
"""

import sys
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Make feature_engineering importable ───────────────────────────────────────
sys.path.insert(0, "/home/rodrigofrancachaves/project-nyc_property_taxes/python_scripts")

# ── Paths — adjust if needed ──────────────────────────────────────────────────
OUTPUT_DIR = "/home/rodrigofrancachaves/project-nyc_property_taxes/outputs"
MODEL_DIR  = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
DATA_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
SALES_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/sales_clean.parquet"

SEP = "=" * 60


# ── Shared helper: load + engineer a sample ───────────────────────────────────
def _load_engineered_sample(sample_n, with_sales=True):
    """
    Loads processed_labeled_data, samples sample_n rows,
    runs engineer_features, and returns (df, features_built).
    Pass with_sales=False to skip the sales join (faster, fewer features).
    """
    from feature_engineering import load_data, engineer_features

    sales_path = SALES_PATH if (with_sales and os.path.exists(SALES_PATH)) else None

    df = load_data(DATA_PATH, drop_unknowns=True)
    df = df.sample(min(sample_n, len(df)), random_state=42)
    df, features_built, _ = engineer_features(df, sales_path=sales_path)
    return df, features_built


# ── 1. Zero / near-zero importance features ───────────────────────────────────
def check_zero_importance(threshold=500):
    print(f"\n{SEP}")
    print("CHECK 1: Zero / near-zero importance features (LightGBM)")
    print(SEP)

    imp_path = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv")
    if not os.path.exists(imp_path):
        print("  SKIP: lgbm_feature_importance.csv not found")
        return

    imp  = pd.read_csv(imp_path).sort_values("Importance")
    zero = imp[imp["Importance"] == 0]
    low  = imp[(imp["Importance"] > 0) & (imp["Importance"] < threshold)]

    print(f"\n  Zero importance ({len(zero)} features) → safe to DROP:")
    for _, row in zero.iterrows():
        print(f"    {row['Feature']}")

    print(f"\n  Near-zero importance < {threshold} ({len(low)} features) → consider dropping:")
    for _, row in low.iterrows():
        print(f"    {row['Feature']:45s}  importance={row['Importance']}")

    total = imp["Importance"].sum()
    top10 = imp.tail(10)["Importance"].sum()
    print(f"\n  Top 10 features account for {top10/total*100:.1f}% of total importance")


# ── 2. SGD model performance gap ─────────────────────────────────────────────
def check_model_comparison():
    print(f"\n{SEP}")
    print("CHECK 2: SGD model comparison")
    print(SEP)

    res_path = os.path.join(OUTPUT_DIR, "linear_model_results.csv")
    if not os.path.exists(res_path):
        print("  SKIP: linear_model_results.csv not found")
        return

    res  = pd.read_csv(res_path)[["Model", "Test F1 Macro", "CV F1 Macro", "CV F1 Std"]]
    print(f"\n{res.to_string(index=False)}")

    best = res["Test F1 Macro"].max()
    for _, row in res.iterrows():
        gap = best - row["Test F1 Macro"]
        if gap > 0.03:
            print(f"\n  ⚠  {row['Model']} is {gap:.3f} F1 below best → consider dropping")


# ── 3. Feature collinearity (correlation clusters) ────────────────────────────
def check_collinearity(sample_n=20_000, corr_threshold=0.97):
    print(f"\n{SEP}")
    print(f"CHECK 3: Feature collinearity (|corr| > {corr_threshold})")
    print(SEP)

    feat_path = os.path.join(MODEL_DIR, "linear_features.pkl")
    if not os.path.exists(feat_path):
        print("  SKIP: linear_features.pkl not found")
        return
    if not os.path.exists(DATA_PATH):
        print("  SKIP: processed_labeled_data.parquet not found")
        return

    try:
        from feature_engineering import load_data, engineer_features
    except ImportError:
        print("  SKIP: feature_engineering.py not importable")
        return

    features = joblib.load(feat_path)

    print(f"  Engineering features on {sample_n:,} row sample (skipping sales join)...")
    # Sales features aren't needed for collinearity — skip for speed
    df, _ = _load_engineered_sample(sample_n, with_sales=False)

    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"  Note: {len(missing)} features not in sample (sales features — expected)")

    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Correlation"})
        .query(f"Correlation > {corr_threshold}")
        .sort_values("Correlation", ascending=False)
    )

    if high_corr.empty:
        print(f"  ✓ No feature pairs above {corr_threshold} correlation")
    else:
        print(f"\n  {len(high_corr)} highly correlated pairs (keep one, drop the other):\n")
        print(high_corr.to_string(index=False))


# ── 4. Prediction confidence (borderline properties) ─────────────────────────
def check_confidence(sample_n=50_000, borderline_threshold=0.50):
    print(f"\n{SEP}")
    print("CHECK 4: Prediction confidence — borderline properties")
    print(SEP)

    model_path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
    feat_path  = os.path.join(MODEL_DIR, "features.pkl")
    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        print("  SKIP: lgbm_model.pkl or features.pkl not found")
        return
    if not os.path.exists(DATA_PATH):
        print("  SKIP: processed_labeled_data.parquet not found")
        return

    try:
        from feature_engineering import load_data, engineer_features
    except ImportError:
        print("  SKIP: feature_engineering.py not importable")
        return

    model    = joblib.load(model_path)
    features = joblib.load(feat_path)

    print(f"  Engineering features on {sample_n:,} row sample (including sales)...")
    df, _ = _load_engineered_sample(sample_n, with_sales=True)

    # Use only features the model was trained on, in the right order
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠  {len(missing)} training features missing from sample:")
        for f in missing:
            print(f"      {f}")
        print(f"  Filling missing features with 0...")
        for f in missing:
            df[f] = 0.0

    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    proba     = model.predict_proba(X)
    max_proba = proba.max(axis=1)

    print(f"\n  Confidence distribution (max class probability):")
    for cutoff in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pct = (max_proba < cutoff).mean() * 100
        print(f"    < {cutoff:.0%} confidence : {pct:5.1f}% of properties")

    borderline = (max_proba < borderline_threshold).mean() * 100
    print(f"\n  ⚠  {borderline:.1f}% of properties below {borderline_threshold:.0%} confidence")
    print(f"     These are borderline — flag separately rather than forcing a label")

    preds = model.predict(X)
    df2   = pd.DataFrame({"pred": preds, "max_proba": max_proba})
    print(f"\n  Mean confidence by predicted class:")
    print(df2.groupby("pred")["max_proba"].mean().round(3).to_string())


# ── 5. Peer group labeling sanity check ──────────────────────────────────────
def check_labeling_sanity(sample_n=50_000):
    print(f"\n{SEP}")
    print("CHECK 5: Peer group labeling sanity")
    print(SEP)

    if not os.path.exists(DATA_PATH):
        print("  SKIP: processed_labeled_data.parquet not found")
        return

    needed = ["FINACTTOT", "FINMKTTOT", "GROSS_SQFT", "target_2026"]
    try:
        df = pd.read_parquet(DATA_PATH, columns=needed)
    except Exception as e:
        print(f"  SKIP: could not load columns — {e}")
        return

    df = df[df["target_2026"] != "unknown"].dropna()
    df = df.sample(min(sample_n, len(df)), random_state=42)

    for col in ["FINACTTOT", "FINMKTTOT", "GROSS_SQFT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    df["assess_per_sqft"] = df["FINACTTOT"] / df["GROSS_SQFT"].clip(lower=1)
    df["mkt_to_assess"]   = df["FINMKTTOT"] / df["FINACTTOT"].clip(lower=1)

    print(f"\n  Assessment per sqft by label (FY2026 target):")
    print(df.groupby("target_2026")["assess_per_sqft"].describe()[
        ["mean", "50%", "std"]
    ].round(2).to_string())

    print(f"\n  Market-to-assessed ratio by label:")
    print(df.groupby("target_2026")["mkt_to_assess"].describe()[
        ["mean", "50%", "std"]
    ].round(2).to_string())

    means = df.groupby("target_2026")["assess_per_sqft"].mean()
    if "overvalued" in means and "undervalued" in means:
        if means["overvalued"] > means["undervalued"]:
            print(f"\n  ✓ Labels are directionally consistent:")
            print(f"    overvalued mean assess/sqft  = {means['overvalued']:.1f}")
            print(f"    undervalued mean assess/sqft = {means['undervalued']:.1f}")
        else:
            print(f"\n  ⚠  Labels may be inverted or noisy — check peer group logic")


# ── 6. Sales feature match rate and usefulness ────────────────────────────────
def check_sales_features():
    print(f"\n{SEP}")
    print("CHECK 6: Sales feature match rate and importance")
    print(SEP)

    imp_path = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv")
    if not os.path.exists(imp_path):
        print("  SKIP: lgbm_feature_importance.csv not found")
        return

    imp = pd.read_csv(imp_path)
    sales_features = [
        "HAS_RECENT_SALE", "N_SALES", "LAST_SALE_PRICE", "LOG_LAST_SALE_PRICE",
        "LAST_SALE_YEAR", "YEARS_SINCE_SALE", "MAX_SALE_PRICE", "MEDIAN_SALE_PRICE",
        "SALE_PRICE_TREND", "ZIP_MEDIAN_SALE_PRICE", "ZIP_SALE_VOLUME",
        "SALE_PRICE_PER_SQFT", "SALE_TO_ASSESS_RATIO", "LOG_SALE_TO_ASSESS",
        "SALE_VS_ZIP_MEDIAN",
    ]

    sales_imp   = imp[imp["Feature"].isin(sales_features)].sort_values("Importance", ascending=False)
    total_imp   = imp["Importance"].sum()
    sales_total = sales_imp["Importance"].sum()

    print(f"\n  Sales feature importances:")
    print(sales_imp.to_string(index=False))
    print(f"\n  Sales features combined = {sales_total/total_imp*100:.2f}% of total importance")

    # Try to get match rate from a small sample with sales join
    if os.path.exists(SALES_PATH):
        try:
            print(f"\n  Computing BBL match rate from 10k sample...")
            df, _ = _load_engineered_sample(10_000, with_sales=True)
            if "HAS_RECENT_SALE" in df.columns:
                match_rate = df["HAS_RECENT_SALE"].mean() * 100
                print(f"  BBL match rate (HAS_RECENT_SALE=1): {match_rate:.1f}%")
                if match_rate < 30:
                    print(f"  ⚠  Low match rate — sales block has limited coverage, consider dropping")
        except Exception as e:
            print(f"  Could not compute match rate: {e}")
    else:
        print(f"  Note: sales_clean.parquet not found — match rate check skipped")


# ── Run all checks ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nNYC PROPERTY TAX — DIAGNOSTIC REPORT")
    print(SEP)

    check_zero_importance()
    check_model_comparison()
    check_collinearity()
    check_confidence()
    check_labeling_sanity()
    check_sales_features()

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)