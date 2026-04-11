import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"
df = pd.read_parquet(os.path.join(data_path, "merged_2022_2024.parquet"))
print(f"Loaded shape: {df.shape}")

# ── Convert key columns to numeric ───────────────────────────────────────────
for col in ["BOROUGH", "GROSS_SQFT", "GROSS SQUARE FEET", "LAND_AREA",
            "YRBUILT", "NUM_BLDGS", "BLD_STORY", "UNITS", "LOT_FRT",
            "LOT_DEP", "FINACTTOT", "PYACTTOT", "PYACTLAND", "FINACTLAND"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Recompute labels using better thresholds ─────────────────────────────────
# Instead of arbitrary percentiles, use tax-class-specific targets:
# Class 1 target: 6% of market value
# Class 2/3/4 target: 45% of market value
# A property is fairly valued if within ±40% of its class target
# This gives more meaningful economic categories

df["target_ratio"] = df["TAX CLASS AT TIME OF SALE"].astype(str).apply(
    lambda x: 0.06 if x.strip().startswith("1") else 0.45
)

# Normalize assessment ratio relative to class target
df["ratio_vs_target"] = df["assessment_ratio"] / df["target_ratio"]

# Label based on how far from target:
# undervalued:   assessed at less than 70% of target → paying too little tax
# fairly_valued: assessed at 70-130% of target → roughly fair
# overvalued:    assessed at more than 130% of target → paying too much tax
df["label"] = pd.cut(
    df["ratio_vs_target"],
    bins=[0, 0.90, 1.10, float("inf")],
    labels=["undervalued", "fairly_valued", "overvalued"]
)
df = df.dropna(subset=["label", "assessment_ratio"])

print(f"\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nLabel proportions:")
print(df["label"].value_counts(normalize=True).round(3))

# ── Feature engineering ───────────────────────────────────────────────────────

# Building age
df["BUILDING_AGE"] = (df["SALE_YEAR"] - df["YRBUILT"]).clip(lower=0, upper=200)

# Building era
df["BUILDING_ERA"] = pd.cut(
    df["YRBUILT"],
    bins=[0, 1900, 1940, 1960, 1980, 2000, 2010, 2030],
    labels=["pre1900", "prewar", "postwar", "1960s", "1980s", "2000s", "modern"]
)

# Property size
df["SQFT"] = df["GROSS SQUARE FEET"].fillna(df["GROSS_SQFT"])
df["LOG_GROSS_SQFT"] = np.log1p(df["SQFT"])
df["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"])

# Lot characteristics
df["LOT_AREA"]       = df["LOT_FRT"] * df["LOT_DEP"]
df["COVERAGE_RATIO"] = df["SQFT"] / df["LAND_AREA"].clip(lower=1)
df["SQFT_PER_UNIT"]  = df["SQFT"] / df["UNITS"].clip(lower=1)

# Assessment trajectory (prior year features — no leakage)
df["PYACTTOT"]       = pd.to_numeric(df["PYACTTOT"], errors="coerce")
df["LOG_PYACTTOT"]   = np.log1p(df["PYACTTOT"])
df["ASSESS_CHANGE"]  = df["FINACTTOT"] - df["PYACTTOT"]
df["ASSESS_CHANGE_PCT"] = (df["ASSESS_CHANGE"] / df["PYACTTOT"].clip(lower=1)).clip(-1, 5)
df["ASSESS_GREW"]    = (df["FINACTTOT"] > df["PYACTTOT"]).astype(int)
df["ASSESS_AT_CAP"]  = df["ASSESS_CHANGE_PCT"].between(0.04, 0.065).astype(int)

# Land vs building value split
df["LAND_TO_TOTAL"]  = df["FINACTLAND"] / df["FINACTTOT"].clip(lower=1)

# Neighborhood-level aggregates (computed from training data only — slight simplification)
neighborhood_stats = df.groupby("NEIGHBORHOOD")["SALE PRICE"].agg(
    NEIGHBORHOOD_MEDIAN_PRICE="median",
    NEIGHBORHOOD_SALE_COUNT="count"
).reset_index()
df = df.merge(neighborhood_stats, on="NEIGHBORHOOD", how="left")
df["PRICE_VS_NEIGHBORHOOD"] = df["SALE PRICE"] / df["NEIGHBORHOOD_MEDIAN_PRICE"].clip(lower=1)
df["LOG_NEIGHBORHOOD_MEDIAN"] = np.log1p(df["NEIGHBORHOOD_MEDIAN_PRICE"])

# ZIP code aggregates
zip_stats = df.groupby("ZIP CODE")["SALE PRICE"].agg(
    ZIP_MEDIAN_PRICE="median",
    ZIP_SALE_COUNT="count"
).reset_index()
df = df.merge(zip_stats, on="ZIP CODE", how="left")
df["PRICE_VS_ZIP"] = df["SALE PRICE"] / df["ZIP_MEDIAN_PRICE"].clip(lower=1)

# Sale timing
if "SALE DATE" in df.columns:
    df["SALE DATE"] = pd.to_datetime(df["SALE DATE"], errors="coerce")
    df["SALE_MONTH"]   = df["SALE DATE"].dt.month
    df["SALE_QUARTER"] = df["SALE DATE"].dt.quarter

# Encode categoricals
df["BLDG_CLASS_CODE"]   = LabelEncoder().fit_transform(df["BLDG_CLASS"].fillna("Unknown"))
df["TAX_CLASS_CODE"]    = LabelEncoder().fit_transform(df["TAX CLASS AT TIME OF SALE"].fillna("Unknown").astype(str))
df["ZONING_CODE"]       = LabelEncoder().fit_transform(df["ZONING"].fillna("Unknown"))
df["NEIGHBORHOOD_CODE"] = LabelEncoder().fit_transform(df["NEIGHBORHOOD"].fillna("Unknown"))
df["ZIP_CODE_CODE"]     = LabelEncoder().fit_transform(df["ZIP CODE"].fillna("Unknown").astype(str))
df["BUILDING_ERA_CODE"] = LabelEncoder().fit_transform(df["BUILDING_ERA"].astype(str))

# ── Features ──────────────────────────────────────────────────────────────────
# NOTE: PRICE_VS_NEIGHBORHOOD and PRICE_VS_ZIP use SALE PRICE
# but they capture relative position vs neighbors, not absolute value.
# They are borderline — keeping them as they add location signal.
# LOG_PYACTTOT is prior year assessed value — not current, so no leakage.
FEATURES = [
    # Location
    "BOROUGH",
    "NEIGHBORHOOD_CODE",
    "ZIP_CODE_CODE",
    # Temporal
    "SALE_YEAR",
    "SALE_MONTH",
    "SALE_QUARTER",
    # Physical characteristics
    "BUILDING_AGE",
    "BUILDING_ERA_CODE",
    "LOG_GROSS_SQFT",
    "LOG_LAND_AREA",
    "LOT_AREA",
    "COVERAGE_RATIO",
    "SQFT_PER_UNIT",
    "NUM_BLDGS",
    "BLD_STORY",
    "UNITS",
    "LOT_FRT",
    "LOT_DEP",
    # Classification
    "BLDG_CLASS_CODE",
    "TAX_CLASS_CODE",
    "ZONING_CODE",
    # Assessment trajectory (prior year — no leakage)
    "LOG_PYACTTOT",
    "ASSESS_CHANGE_PCT",
    "ASSESS_GREW",
    "ASSESS_AT_CAP",
    "LAND_TO_TOTAL",
    # Neighborhood context
    "LOG_NEIGHBORHOOD_MEDIAN",
    "NEIGHBORHOOD_SALE_COUNT",
    "PRICE_VS_NEIGHBORHOOD",
    "PRICE_VS_ZIP",
    "ZIP_SALE_COUNT",
]

# Only keep features that exist in df
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"\nUsing {len(FEATURES)} features: {FEATURES}")

# ── Prepare X and y ───────────────────────────────────────────────────────────
df_model = df[FEATURES + ["label"]].copy()
for col in FEATURES:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

# Impute with median
null_counts = df_model[FEATURES].isnull().sum()
print(f"\nNull counts before imputation:")
print(null_counts[null_counts > 0])

for col in FEATURES:
    if df_model[col].isnull().any():
        median_val = df_model[col].median()
        df_model[col] = df_model[col].fillna(median_val)

df_model = df_model.dropna(subset=["label"])
print(f"\nShape after imputation: {df_model.shape}")

X = df_model[FEATURES]
y = df_model["label"].astype(str)

print(f"\nTarget distribution:\n{y.value_counts()}")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]:,}")
print(f"Test size:  {X_test.shape[0]:,}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=10),
}

results = []
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")

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

    results.append({"Model": name, "Accuracy": round(acc, 4), "F1 Score": round(f1, 4)})

# ── Feature importance ────────────────────────────────────────────────────────
rf_model = models["Random Forest"]
feat_imp = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(f"\n{'='*50}")
print("RANDOM FOREST FEATURE IMPORTANCE")
print('='*50)
print(feat_imp.to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("MODEL COMPARISON SUMMARY")
print('='*50)
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print(results_df.to_string(index=False))

baseline = y.value_counts(normalize=True).max()
print(f"\nBaseline accuracy (majority class): {baseline:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
results_df.to_csv(os.path.join(data_path, "model_results.csv"), index=False)
feat_imp.to_csv(os.path.join(data_path, "feature_importance.csv"), index=False)
print(f"\nResults saved to: {data_path}")