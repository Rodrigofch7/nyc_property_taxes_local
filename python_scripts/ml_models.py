import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"
df = pd.read_parquet(os.path.join(data_path, "merged_2024.parquet"))
print(f"Loaded shape: {df.shape}")

# ── Create labels using empirical percentiles by tax class ────────────────────
def assign_label(group):
    p25 = group["assessment_ratio"].quantile(0.25)
    p75 = group["assessment_ratio"].quantile(0.75)
    return pd.cut(
        group["assessment_ratio"],
        bins=[0, p25, p75, float("inf")],
        labels=["undervalued", "fairly_valued", "overvalued"]
    )

df["label"] = df.groupby("TAX CLASS AT TIME OF SALE", group_keys=False).apply(assign_label)
df = df.dropna(subset=["label"])

print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nLabel proportions:")
print(df["label"].value_counts(normalize=True).round(3))

# ── Feature engineering ───────────────────────────────────────────────────────
# Borough from BBL
df["BOROUGH"] = pd.to_numeric(df["BOROUGH"], errors="coerce")

# Building age
df["BUILDING_AGE"] = 2024 - pd.to_numeric(df["YRBUILT"], errors="coerce")
df["BUILDING_AGE"] = df["BUILDING_AGE"].clip(lower=0, upper=200)

# Price per sqft
df["GROSS SQUARE FEET"] = pd.to_numeric(df["GROSS SQUARE FEET"], errors="coerce")
df["PRICE_PER_SQFT"] = np.where(
    df["GROSS SQUARE FEET"] > 0,
    df["SALE PRICE"] / df["GROSS SQUARE FEET"],
    np.nan
)

# Assessed value per sqft
df["GROSS_SQFT"] = pd.to_numeric(df["GROSS_SQFT"], errors="coerce")
df["ASSESSED_PER_SQFT"] = np.where(
    df["GROSS_SQFT"] > 0,
    df["FINACTTOT_per_unit"] / df["GROSS_SQFT"],
    np.nan
)

# Log transforms for skewed variables
df["LOG_SALE_PRICE"]    = np.log1p(df["SALE PRICE"])
df["LOG_FINACTTOT"]     = np.log1p(df["FINACTTOT"])
df["LOG_GROSS_SQFT"]    = np.log1p(df["GROSS_SQFT"])
df["LOG_LAND_AREA"]     = np.log1p(pd.to_numeric(df["LAND_AREA"], errors="coerce"))

# Numeric features for model
FEATURES = [
    "BOROUGH",
    "LOG_SALE_PRICE",
    "LOG_FINACTTOT",
    "LOG_GROSS_SQFT",
    "LOG_LAND_AREA",
    "BUILDING_AGE",
    "PRICE_PER_SQFT",
    "ASSESSED_PER_SQFT",
    "NUM_BLDGS",
    "BLD_STORY",
    "UNITS",
    "LOT_FRT",
    "LOT_DEP",
]

# Encode categorical features
df["BLDG_CLASS_CODE"] = LabelEncoder().fit_transform(
    df["BLDG_CLASS"].fillna("Unknown")
)
df["TAX_CLASS_CODE"] = LabelEncoder().fit_transform(
    df["TAX CLASS AT TIME OF SALE"].fillna("Unknown").astype(str)
)
df["ZONING_CODE"] = LabelEncoder().fit_transform(
    df["ZONING"].fillna("Unknown")
)

FEATURES += ["BLDG_CLASS_CODE", "TAX_CLASS_CODE", "ZONING_CODE"]

# ── Prepare X and y ───────────────────────────────────────────────────────────
df_model = df[FEATURES + ["label"]].copy()

# Convert all features to numeric
for col in FEATURES:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

# Drop rows with any missing values
df_model = df_model.dropna()
print(f"\nShape after dropping nulls: {df_model.shape}")

X = df_model[FEATURES]
y = df_model["label"].astype(str)

print(f"\nFeatures: {FEATURES}")
print(f"Target distribution:\n{y.value_counts()}")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]:,}")
print(f"Test size:  {X_test.shape[0]:,}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Define models ─────────────────────────────────────────────────────────────
models = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":            DecisionTreeClassifier(random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors":      KNeighborsClassifier(n_neighbors=5),
}

# ── Train and evaluate ────────────────────────────────────────────────────────
results = []

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")

    # Use scaled data for distance-based and linear models
    if name in ["Logistic Regression", "K-Nearest Neighbors"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    results.append({
        "Model":    name,
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4)
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("MODEL COMPARISON SUMMARY")
print('='*50)
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print(results_df.to_string(index=False))

# ── Save results ──────────────────────────────────────────────────────────────
results_df.to_csv(
    os.path.join(data_path, "model_results.csv"),
    index=False
)
print(f"\nResults saved to: {data_path}/model_results.csv")