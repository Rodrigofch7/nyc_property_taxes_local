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
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Label proportions:\n{df['label'].value_counts(normalize=True).round(3)}")

# ── Feature engineering ───────────────────────────────────────────────────────
# Only use property characteristics — NOT sale price or assessed value
# to avoid data leakage

df["BOROUGH"]           = pd.to_numeric(df["BOROUGH"], errors="coerce")
df["GROSS_SQFT"]        = pd.to_numeric(df["GROSS_SQFT"], errors="coerce")
df["GROSS SQUARE FEET"] = pd.to_numeric(df["GROSS SQUARE FEET"], errors="coerce")
df["LAND_AREA"]         = pd.to_numeric(df["LAND_AREA"], errors="coerce")
df["YRBUILT"]           = pd.to_numeric(df["YRBUILT"], errors="coerce")
df["NUM_BLDGS"]         = pd.to_numeric(df["NUM_BLDGS"], errors="coerce")
df["BLD_STORY"]         = pd.to_numeric(df["BLD_STORY"], errors="coerce")
df["UNITS"]             = pd.to_numeric(df["UNITS"], errors="coerce")
df["LOT_FRT"]           = pd.to_numeric(df["LOT_FRT"], errors="coerce")
df["LOT_DEP"]           = pd.to_numeric(df["LOT_DEP"], errors="coerce")

# Building age at time of sale
df["BUILDING_AGE"] = (df["SALE_YEAR"] - df["YRBUILT"]).clip(lower=0, upper=200)

# Property size — use sales sqft first, fall back to assessment sqft
df["SQFT"] = df["GROSS SQUARE FEET"].fillna(df["GROSS_SQFT"])

# Log transforms for skewed size variables
df["LOG_GROSS_SQFT"] = np.log1p(df["SQFT"])
df["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"])

# Encode categoricals
df["BLDG_CLASS_CODE"]    = LabelEncoder().fit_transform(df["BLDG_CLASS"].fillna("Unknown"))
df["TAX_CLASS_CODE"]     = LabelEncoder().fit_transform(df["TAX CLASS AT TIME OF SALE"].fillna("Unknown").astype(str))
df["ZONING_CODE"]        = LabelEncoder().fit_transform(df["ZONING"].fillna("Unknown"))
df["NEIGHBORHOOD_CODE"]  = LabelEncoder().fit_transform(df["NEIGHBORHOOD"].fillna("Unknown"))

# ── Features — NO sale price, NO assessed value, NO derived ratios ────────────
# These are pure property characteristics that should be known
# BEFORE the sale happens — no leakage
FEATURES = [
    "BOROUGH",           # location
    "SALE_YEAR",         # market year
    "BUILDING_AGE",      # age of building
    "LOG_GROSS_SQFT",    # building size
    "LOG_LAND_AREA",     # lot size
    "NUM_BLDGS",         # number of buildings on lot
    "BLD_STORY",         # number of floors
    "UNITS",             # number of units
    "LOT_FRT",           # lot frontage
    "LOT_DEP",           # lot depth
    "BLDG_CLASS_CODE",   # building classification
    "TAX_CLASS_CODE",    # tax class
    "ZONING_CODE",       # zoning district
    "NEIGHBORHOOD_CODE", # neighborhood
]

print(f"\nFeatures used (no leakage): {FEATURES}")

# ── Prepare X and y ───────────────────────────────────────────────────────────
df_model = df[FEATURES + ["label"]].copy()
for col in FEATURES:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

# Impute missing values with median
null_counts = df_model[FEATURES].isnull().sum()
print(f"\nNull counts before imputation:")
print(null_counts[null_counts > 0])

for col in FEATURES:
    if df_model[col].isnull().any():
        median_val = df_model[col].median()
        df_model[col] = df_model[col].fillna(median_val)
        print(f"  Imputed {col} with median: {median_val:.2f}")

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
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=5),
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

# ── Feature importance (Random Forest) ───────────────────────────────────────
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

# Baseline accuracy (always predict most common class)
baseline = y.value_counts(normalize=True).max()
print(f"\nBaseline accuracy (majority class): {baseline:.4f}")
print("Note: A good model should beat this baseline significantly.")

# ── Save ──────────────────────────────────────────────────────────────────────
results_df.to_csv(os.path.join(data_path, "model_results.csv"), index=False)
feat_imp.to_csv(os.path.join(data_path, "feature_importance.csv"), index=False)
print(f"\nResults saved to: {data_path}")