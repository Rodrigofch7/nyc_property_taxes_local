import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"
df = pd.read_parquet(os.path.join(data_path, "merged_2020_2024.parquet"))
print(f"Loaded shape: {df.shape}")

# ── Convert key columns to numeric ───────────────────────────────────────────
for col in ["BOROUGH", "GROSS_SQFT", "GROSS SQUARE FEET", "LAND_AREA",
            "YRBUILT", "NUM_BLDGS", "BLD_STORY", "UNITS", "LOT_FRT",
            "LOT_DEP", "FINACTTOT", "PYACTTOT", "PYACTLAND", "FINACTLAND"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Labels ────────────────────────────────────────────────────────────────────
df["target_ratio"] = df["TAX CLASS AT TIME OF SALE"].astype(str).apply(
    lambda x: 0.06 if x.strip().startswith("1") else 0.45
)
df["ratio_vs_target"] = df["assessment_ratio"] / df["target_ratio"]
df["label"] = pd.cut(
    df["ratio_vs_target"],
    bins=[0, 0.90, 1.10, float("inf")],
    labels=["undervalued", "fairly_valued", "overvalued"]
)
df = df.dropna(subset=["label", "assessment_ratio"])
print(f"\nLabel distribution:\n{df['label'].value_counts()}")

# ── Feature engineering ───────────────────────────────────────────────────────
df["BUILDING_AGE"] = (df["SALE_YEAR"] - df["YRBUILT"]).clip(lower=0, upper=200)
df["BUILDING_ERA"] = pd.cut(
    df["YRBUILT"],
    bins=[0, 1900, 1940, 1960, 1980, 2000, 2010, 2030],
    labels=["pre1900", "prewar", "postwar", "1960s", "1980s", "2000s", "modern"]
)
df["SQFT"] = df["GROSS SQUARE FEET"].fillna(df["GROSS_SQFT"])
df["LOG_GROSS_SQFT"] = np.log1p(df["SQFT"])
df["LOG_LAND_AREA"]  = np.log1p(df["LAND_AREA"])
df["LOT_AREA"]       = df["LOT_FRT"] * df["LOT_DEP"]
df["COVERAGE_RATIO"] = df["SQFT"] / df["LAND_AREA"].clip(lower=1)
df["SQFT_PER_UNIT"]  = df["SQFT"] / df["UNITS"].clip(lower=1)
df["PYACTTOT"]       = pd.to_numeric(df["PYACTTOT"], errors="coerce")
df["LOG_PYACTTOT"]   = np.log1p(df["PYACTTOT"])
df["ASSESS_CHANGE"]  = df["FINACTTOT"] - df["PYACTTOT"]
df["ASSESS_CHANGE_PCT"] = (df["ASSESS_CHANGE"] / df["PYACTTOT"].clip(lower=1)).clip(-1, 5)
df["ASSESS_GREW"]    = (df["FINACTTOT"] > df["PYACTTOT"]).astype(int)
df["ASSESS_AT_CAP"]  = df["ASSESS_CHANGE_PCT"].between(0.04, 0.065).astype(int)
df["LAND_TO_TOTAL"]  = df["FINACTLAND"] / df["FINACTTOT"].clip(lower=1)

neighborhood_stats = df.groupby("NEIGHBORHOOD")["SALE PRICE"].agg(
    NEIGHBORHOOD_MEDIAN_PRICE="median", NEIGHBORHOOD_SALE_COUNT="count"
).reset_index()
df = df.merge(neighborhood_stats, on="NEIGHBORHOOD", how="left")
df["PRICE_VS_NEIGHBORHOOD"] = df["SALE PRICE"] / df["NEIGHBORHOOD_MEDIAN_PRICE"].clip(lower=1)
df["LOG_NEIGHBORHOOD_MEDIAN"] = np.log1p(df["NEIGHBORHOOD_MEDIAN_PRICE"])

zip_stats = df.groupby("ZIP CODE")["SALE PRICE"].agg(
    ZIP_MEDIAN_PRICE="median", ZIP_SALE_COUNT="count"
).reset_index()
df = df.merge(zip_stats, on="ZIP CODE", how="left")
df["PRICE_VS_ZIP"] = df["SALE PRICE"] / df["ZIP_MEDIAN_PRICE"].clip(lower=1)

if "SALE DATE" in df.columns:
    df["SALE DATE"] = pd.to_datetime(df["SALE DATE"], errors="coerce")
    df["SALE_MONTH"]   = df["SALE DATE"].dt.month
    df["SALE_QUARTER"] = df["SALE DATE"].dt.quarter

df["BLDG_CLASS_CODE"]   = LabelEncoder().fit_transform(df["BLDG_CLASS"].fillna("Unknown"))
df["TAX_CLASS_CODE"]    = LabelEncoder().fit_transform(df["TAX CLASS AT TIME OF SALE"].fillna("Unknown").astype(str))
df["ZONING_CODE"]       = LabelEncoder().fit_transform(df["ZONING"].fillna("Unknown"))
df["NEIGHBORHOOD_CODE"] = LabelEncoder().fit_transform(df["NEIGHBORHOOD"].fillna("Unknown"))
df["ZIP_CODE_CODE"]     = LabelEncoder().fit_transform(df["ZIP CODE"].fillna("Unknown").astype(str))
df["BUILDING_ERA_CODE"] = LabelEncoder().fit_transform(df["BUILDING_ERA"].astype(str))

FEATURES = [
    "BOROUGH", "NEIGHBORHOOD_CODE", "ZIP_CODE_CODE",
    "SALE_YEAR", "SALE_MONTH", "SALE_QUARTER",
    "BUILDING_AGE", "BUILDING_ERA_CODE", "LOG_GROSS_SQFT", "LOG_LAND_AREA",
    "LOT_AREA", "COVERAGE_RATIO", "SQFT_PER_UNIT",
    "NUM_BLDGS", "BLD_STORY", "UNITS", "LOT_FRT", "LOT_DEP",
    "BLDG_CLASS_CODE", "TAX_CLASS_CODE", "ZONING_CODE",
    "LOG_PYACTTOT", "ASSESS_CHANGE_PCT", "ASSESS_GREW", "ASSESS_AT_CAP",
    "LAND_TO_TOTAL", "LOG_NEIGHBORHOOD_MEDIAN", "NEIGHBORHOOD_SALE_COUNT",
    "PRICE_VS_NEIGHBORHOOD", "PRICE_VS_ZIP", "ZIP_SALE_COUNT",
]
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"\nUsing {len(FEATURES)} features")

# ── Prepare X and y ───────────────────────────────────────────────────────────
df_model = df[FEATURES + ["label"]].copy()
for col in FEATURES:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
    if df_model[col].isnull().any():
        df_model[col] = df_model[col].fillna(df_model[col].median())

df_model = df_model.dropna(subset=["label"])
X = df_model[FEATURES]
y = df_model["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Helper ────────────────────────────────────────────────────────────────────
def evaluate(name, model, X_tr, X_te):
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test F1       : {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    post_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
    # n_jobs=1 here — model itself may already use threading internally
    cv_scores = cross_val_score(model, X_tr, y_train,
                                cv=post_cv, scoring="f1_weighted", n_jobs=1)
    print(f"Post-tuning CV F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return acc, f1, cv_scores.mean(), cv_scores.std()

results = []
best_estimators = {}

# ── 1. Logistic Regression ────────────────────────────────────────────────────
print(f"\n{'='*60}\nTuning: Logistic Regression")
search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid={"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["saga"]},
    cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1,
)
search.fit(X_train_scaled, y_train)
print(f"Best params : {search.best_params_}")
print(f"Best CV F1  : {search.best_score_:.4f}")
best_estimators["Logistic Regression"] = search.best_estimator_
acc, f1, cv_mean, cv_std = evaluate("LR", search.best_estimator_, X_train_scaled, X_test_scaled)
results.append({"Model": "Logistic Regression", "Best CV F1": round(search.best_score_, 4),
                "Test Accuracy": round(acc, 4), "Test F1": round(f1, 4),
                "CV F1 Mean": round(cv_mean, 4), "CV F1 Std": round(cv_std, 4),
                "Best Params": str(search.best_params_)})

# ── 2. Decision Tree ──────────────────────────────────────────────────────────
print(f"\n{'='*60}\nTuning: Decision Tree")
search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": [5, 10, 15, 20, None], "min_samples_split": [2, 10, 20],
                "min_samples_leaf": [1, 5, 10], "criterion": ["gini", "entropy"]},
    cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)
print(f"Best params : {search.best_params_}")
print(f"Best CV F1  : {search.best_score_:.4f}")
best_estimators["Decision Tree"] = search.best_estimator_
acc, f1, cv_mean, cv_std = evaluate("DT", search.best_estimator_, X_train, X_test)
results.append({"Model": "Decision Tree", "Best CV F1": round(search.best_score_, 4),
                "Test Accuracy": round(acc, 4), "Test F1": round(f1, 4),
                "CV F1 Mean": round(cv_mean, 4), "CV F1 Std": round(cv_std, 4),
                "Best Params": str(search.best_params_)})

# ── 3. Random Forest  (memory-safe) ──────────────────────────────────────────
# Three changes from the version that was OOM-killed:
#   a) n_jobs=2 on the estimator  → caps worker threads, prevents RAM explosion
#   b) n_estimators max 200       → fewer trees per candidate
#   c) n_iter=10, outer n_jobs=1  → only one search candidate runs at a time
print(f"\n{'='*60}\nTuning: Random Forest  (memory-safe)")
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=2),   # ← was n_jobs=-1
    param_distributions={
        "n_estimators": [100, 200],                      # ← removed 400
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    },
    n_iter=10,                                           # ← was 30
    cv=cv,
    scoring="f1_weighted",
    n_jobs=1,                                            # ← outer loop sequential
    random_state=42,
    verbose=1,
    pre_dispatch="2*n_jobs",
)
search.fit(X_train, y_train)
print(f"Best params : {search.best_params_}")
print(f"Best CV F1  : {search.best_score_:.4f}")
best_estimators["Random Forest"] = search.best_estimator_
acc, f1, cv_mean, cv_std = evaluate("RF", search.best_estimator_, X_train, X_test)
results.append({"Model": "Random Forest", "Best CV F1": round(search.best_score_, 4),
                "Test Accuracy": round(acc, 4), "Test F1": round(f1, 4),
                "CV F1 Mean": round(cv_mean, 4), "CV F1 Std": round(cv_std, 4),
                "Best Params": str(search.best_params_)})

# ── 4. HistGradientBoosting  (replaces GradientBoostingClassifier) ────────────
# Bins continuous features into histograms → 10-100x less RAM and faster
# than GradientBoostingClassifier on datasets of this size.
print(f"\n{'='*60}\nTuning: HistGradientBoosting  (fast, low-memory boosting)")
search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_distributions={
        "max_iter": [100, 200, 300],
        "max_depth": [3, 5, 7, None],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf": [5, 10, 20],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    n_iter=15,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=1,
    random_state=42,
    verbose=1,
)
search.fit(X_train, y_train)
print(f"Best params : {search.best_params_}")
print(f"Best CV F1  : {search.best_score_:.4f}")
best_estimators["HistGradientBoosting"] = search.best_estimator_
acc, f1, cv_mean, cv_std = evaluate("HGB", search.best_estimator_, X_train, X_test)
results.append({"Model": "HistGradientBoosting", "Best CV F1": round(search.best_score_, 4),
                "Test Accuracy": round(acc, 4), "Test F1": round(f1, 4),
                "CV F1 Mean": round(cv_mean, 4), "CV F1 Std": round(cv_std, 4),
                "Best Params": str(search.best_params_)})

# ── 5. K-Nearest Neighbors ────────────────────────────────────────────────────
print(f"\n{'='*60}\nTuning: K-Nearest Neighbors")
search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={"n_neighbors": [5, 10, 20, 40], "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]},
    cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1,
)
search.fit(X_train_scaled, y_train)
print(f"Best params : {search.best_params_}")
print(f"Best CV F1  : {search.best_score_:.4f}")
best_estimators["KNN"] = search.best_estimator_
acc, f1, cv_mean, cv_std = evaluate("KNN", search.best_estimator_, X_train_scaled, X_test_scaled)
results.append({"Model": "K-Nearest Neighbors", "Best CV F1": round(search.best_score_, 4),
                "Test Accuracy": round(acc, 4), "Test F1": round(f1, 4),
                "CV F1 Mean": round(cv_mean, 4), "CV F1 Std": round(cv_std, 4),
                "Best Params": str(search.best_params_)})

# ── Feature importance from best Random Forest ────────────────────────────────
rf_best = best_estimators["Random Forest"]
feat_imp = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": rf_best.feature_importances_
}).sort_values("Importance", ascending=False)

print(f"\n{'='*60}")
print("RANDOM FOREST (TUNED) — FEATURE IMPORTANCE")
print(feat_imp.to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("MODEL COMPARISON SUMMARY")
results_df = pd.DataFrame(results).sort_values("Test F1", ascending=False)
print(results_df[["Model", "Best CV F1", "Test Accuracy", "Test F1",
                   "CV F1 Mean", "CV F1 Std"]].to_string(index=False))

baseline = y.value_counts(normalize=True).max()
print(f"\nBaseline (majority class): {baseline:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
results_df.to_csv(os.path.join(data_path, "model_results_tuned.csv"), index=False)
feat_imp.to_csv(os.path.join(data_path, "feature_importance_tuned.csv"), index=False)
print(f"\nSaved to: {data_path}")