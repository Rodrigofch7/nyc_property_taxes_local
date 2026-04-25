import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
data_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"
df = pd.read_parquet(os.path.join(data_path, "merged_2020_2024.parquet"))

print(f"Loaded: {df.shape}")

# ─────────────────────────────────────────────
# Numeric conversion
# ─────────────────────────────────────────────
num_cols = [
    "BOROUGH", "GROSS_SQFT", "GROSS SQUARE FEET", "LAND_AREA",
    "YRBUILT", "NUM_BLDGS", "BLD_STORY", "UNITS",
    "LOT_FRT", "LOT_DEP", "FINACTTOT", "PYACTTOT"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ─────────────────────────────────────────────
# BETTER LABEL DEFINITION (IMPORTANT FIX)
# ─────────────────────────────────────────────
df["target_ratio"] = df["TAX CLASS AT TIME OF SALE"].astype(str).apply(
    lambda x: 0.06 if x.strip().startswith("1") else 0.45
)

df["ratio_vs_target"] = df["assessment_ratio"] / df["target_ratio"]

# 🔥 FIX: more robust bins (reduces noise in "fair")
df["label"] = pd.cut(
    df["ratio_vs_target"],
    bins=[0, 0.85, 1.15, np.inf],
    labels=["undervalued", "fairly_valued", "overvalued"]
)

df = df.dropna(subset=["label", "assessment_ratio"])

print("\nClass distribution:")
print(df["label"].value_counts())

# ─────────────────────────────────────────────
# Feature engineering (keep strong signals only)
# ─────────────────────────────────────────────
df["BUILDING_AGE"] = (df["SALE_YEAR"] - df["YRBUILT"]).clip(0, 200)

df["SQFT"] = df["GROSS SQUARE FEET"].fillna(df["GROSS_SQFT"])
df["LOG_SQFT"] = np.log1p(df["SQFT"])
df["LOG_LAND"] = np.log1p(df["LAND_AREA"])

df["COVERAGE_RATIO"] = df["SQFT"] / df["LAND_AREA"].clip(lower=1)
df["SQFT_PER_UNIT"] = df["SQFT"] / df["UNITS"].clip(lower=1)

df["ASSESS_PER_SQFT"] = df["FINACTTOT"] / df["SQFT"].clip(lower=1)

# Neighborhood signal (VERY important)
nbhd = df.groupby("NEIGHBORHOOD")["ASSESS_PER_SQFT"].median().reset_index()
nbhd.columns = ["NEIGHBORHOOD", "NBHD_MEDIAN"]

df = df.merge(nbhd, on="NEIGHBORHOOD", how="left")
df["VS_NEIGHBORHOOD"] = df["ASSESS_PER_SQFT"] / df["NBHD_MEDIAN"].clip(lower=1)

df["NBHD_PERCENTILE"] = df.groupby("NEIGHBORHOOD")["ASSESS_PER_SQFT"].rank(pct=True)

# ─────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────
df["NEIGHBORHOOD_CODE"] = LabelEncoder().fit_transform(df["NEIGHBORHOOD"].astype(str))
df["ZIP_CODE_CODE"] = LabelEncoder().fit_transform(df["ZIP CODE"].astype(str))

# ─────────────────────────────────────────────
# FEATURES (clean + high signal)
# ─────────────────────────────────────────────
FEATURES = [
    "BOROUGH",
    "NEIGHBORHOOD_CODE",
    "ZIP_CODE_CODE",
    "SALE_YEAR",
    "BUILDING_AGE",
    "LOG_SQFT",
    "LOG_LAND",
    "COVERAGE_RATIO",
    "SQFT_PER_UNIT",
    "ASSESS_PER_SQFT",
    "VS_NEIGHBORHOOD",
    "NBHD_PERCENTILE"
]

df_model = df[FEATURES + ["label"]].copy()

for c in FEATURES:
    df_model[c] = pd.to_numeric(df_model[c], errors="coerce")
    df_model[c] = df_model[c].fillna(df_model[c].median())

X = df_model[FEATURES]
y = df_model["label"].astype(str)

# ─────────────────────────────────────────────
# Train/test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────
# SAMPLE for tuning (speed + stability)
# ─────────────────────────────────────────────
X_tune = X_train.sample(50000, random_state=42)
y_tune = y_train.loc[X_tune.index]

# ─────────────────────────────────────────────
# CLASS WEIGHTS (IMPORTANT FIX)
# ─────────────────────────────────────────────
sample_weights = compute_sample_weight("balanced", y_tune)

# ─────────────────────────────────────────────
# MODEL + TUNING
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

model = HistGradientBoostingClassifier(random_state=42)

search = RandomizedSearchCV(
    model,
    param_distributions={
        "max_iter": [200, 300, 400],
        "max_depth": [5, 7, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_samples_leaf": [20, 40, 80],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    n_iter=10,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_tune, y_tune, sample_weight=sample_weights)

print("\nBest params:", search.best_params_)
print("Best CV F1 macro:", search.best_score_)

# ─────────────────────────────────────────────
# FINAL MODEL
# ─────────────────────────────────────────────
best_model = search.best_estimator_

# retrain with full data + weights
final_weights = compute_sample_weight("balanced", y_train)
best_model.fit(X_train, y_train, sample_weight=final_weights)

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
y_pred = best_model.predict(X_test)

print("\n" + "="*60)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 macro:", f1_score(y_test, y_pred, average="macro"))
print("Test F1 weighted:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred,
                      labels=["undervalued", "fairly_valued", "overvalued"]))