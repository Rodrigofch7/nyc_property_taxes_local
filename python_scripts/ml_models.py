import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/merged_fairness_2020_2024.parquet"
ASSESS_BASE = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"
OUTPUT_DIR  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load merged data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"Loaded: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Re-attach property characteristics from assessment
# ─────────────────────────────────────────────────────────────────────────────
prop_chars = pd.read_parquet(
    f"{ASSESS_BASE}/assessment_FY2024.parquet",
    columns=["BBL", "YRBUILT", "GROSS_SQFT", "LAND_AREA",
             "NUM_BLDGS", "BLD_STORY", "LOT_FRT", "LOT_DEP"]
).drop_duplicates(subset="BBL")

df = df.merge(prop_chars, on="BBL", how="left")
df["YRBUILT"]    = df["YRBUILT"].fillna(df["YEAR BUILT"])
df["GROSS_SQFT"] = df["GROSS_SQFT"].fillna(df["GROSS SQUARE FEET"])
df["LAND_AREA"]  = df["LAND_AREA"].fillna(df["LAND SQUARE FEET"])

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bring in PYACTTOT (prior year AV) — clean + prefix
# ─────────────────────────────────────────────────────────────────────────────
py_av = pd.read_parquet(
    f"{ASSESS_BASE}/assessment_FY2024.parquet",
    columns=["BBL", "PYACTTOT"]
).drop_duplicates(subset="BBL")

py_av["PYACTTOT_clean"] = (
    py_av["PYACTTOT"].astype(str)
    .str.replace(r"[+\s]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)
df = df.merge(py_av[["BBL", "PYACTTOT_clean"]], on="BBL", how="left")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Label definition (IAAO ±20% standard)
#    sales_ratio = FINMKTTOT / SALE PRICE
#    < 0.80 → undertaxed | 0.80–1.20 → fairly_assessed | > 1.20 → overtaxed
# ─────────────────────────────────────────────────────────────────────────────
df["label"] = pd.cut(
    df["sales_ratio"],
    bins=[0, 0.80, 1.20, np.inf],
    labels=["undertaxed", "fairly_assessed", "overtaxed"]
)
df = df.dropna(subset=["label", "sales_ratio"])

print("\nClass distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Numeric cleaning
# ─────────────────────────────────────────────────────────────────────────────
for col in [
    "GROSS_SQFT", "LAND_AREA", "YRBUILT", "NUM_BLDGS",
    "BLD_STORY", "UNITS", "COOP_APTS", "LOT_FRT", "LOT_DEP",
    "CURRENT_FINACTTOT", "CURRENT_FINACTLAND",
    "PYACTTOT_clean", "BOROUGH", "SALE PRICE",
    "RESIDENTIAL UNITS", "COMMERCIAL UNITS"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Base feature engineering (no sale price, no FINMKTTOT)
# ─────────────────────────────────────────────────────────────────────────────

# Building age
df["BUILDING_AGE"] = (df["SALE_YEAR"] - df["YRBUILT"]).clip(0, 200)

# Square footage
df["SQFT"]          = df["GROSS_SQFT"].fillna(df["GROSS SQUARE FEET"])
df["LOG_SQFT"]      = np.log1p(df["SQFT"])
df["LOG_LAND"]      = np.log1p(df["LAND_AREA"])

# Physical ratios
df["COVERAGE_RATIO"]   = df["SQFT"] / df["LAND_AREA"].clip(lower=1)
df["SQFT_PER_UNIT"]    = df["SQFT"] / df["UNITS"].clip(lower=1)
df["FLOOR_AREA_RATIO"] = (df["SQFT"] * df["BLD_STORY"].fillna(1)) / df["LAND_AREA"].clip(lower=1)

# Assessment signals — FINACTTOT only (taxable AV, set before sale)
df["LOG_FINACTTOT"]   = np.log1p(df["CURRENT_FINACTTOT"])
df["ASSESS_PER_SQFT"] = df["CURRENT_FINACTTOT"] / df["SQFT"].clip(lower=1)
df["ASSESS_PER_UNIT"] = df["CURRENT_FINACTTOT"] / df["UNITS"].clip(lower=1)
df["LAND_SHARE"]      = df["CURRENT_FINACTLAND"] / df["CURRENT_FINACTTOT"].clip(lower=1)

# AV cap effect signals
df["AV_YOY_CHANGE"] = (
    (df["CURRENT_FINACTTOT"] - df["PYACTTOT_clean"])
    / df["PYACTTOT_clean"].clip(lower=1)
).clip(-1, 5)
df["AV_FROZEN_FLAG"] = (df["AV_YOY_CHANGE"].abs() < 0.02).astype(int)

df["FINACTTOT_FY2020"] = pd.to_numeric(df["FINACTTOT_FY2020"], errors="coerce")
df["AV_5YR_CHANGE"] = (
    (df["CURRENT_FINACTTOT"] - df["FINACTTOT_FY2020"])
    / df["FINACTTOT_FY2020"].clip(lower=1)
).clip(-1, 10)

# Flags
df["TAX_CLASS_INT"]   = df["TAX CLASS AT TIME OF SALE"].astype(float).fillna(0).astype(int)
df["IS_COOP"]         = (df["COOP_APTS"].fillna(0) > 0).astype(int)
df["IS_MIXED"]        = (df["UNITS"].fillna(0) > df["RESIDENTIAL UNITS"].fillna(0)).astype(int)
df["FISCAL_YEAR_NUM"] = df["FISCAL_YEAR"].astype(int)

# Price per sqft (needed for lag computations below)
df["PRICE_PER_SQFT"] = df["SALE PRICE"] / df["SQFT"].clip(lower=1)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TEMPORAL LAG FEATURES — the key addition
#
#    Rule: for a sale in year Y, we may only use information from year Y-1
#    or earlier. This includes sale prices, sales ratios, and market stats
#    from prior transactions. Never from the same transaction.
# ─────────────────────────────────────────────────────────────────────────────

# Sort chronologically — required for all shift() operations
df = df.sort_values(["BBL", "SALE DATE"]).reset_index(drop=True)

# ── 7a. Lag features: prior sale of the SAME property (repeat sales) ─────────
df["PRIOR_SALE_PRICE"]  = df.groupby("BBL")["SALE PRICE"].shift(1)
df["PRIOR_SALES_RATIO"] = df.groupby("BBL")["sales_ratio"].shift(1)
df["PRIOR_SALE_YEAR"]   = df.groupby("BBL")["SALE_YEAR"].shift(1)

# Years elapsed since last sale (captures market drift)
df["YEARS_SINCE_PRIOR_SALE"] = (
    df["SALE_YEAR"] - df["PRIOR_SALE_YEAR"]
).clip(0, 30)

# Price appreciation since last sale (only meaningful for repeat sales)
df["PRICE_APPRECIATION"] = (
    (df["SALE PRICE"] - df["PRIOR_SALE_PRICE"])
    / df["PRIOR_SALE_PRICE"].clip(lower=1)
).clip(-1, 10)

# Log of prior sale price
df["LOG_PRIOR_PRICE"] = np.log1p(df["PRIOR_SALE_PRICE"].fillna(0))

# Was this property assessed fairly last time it sold?
df["PRIOR_LABEL_UNDERTAXED"]  = (df["PRIOR_SALES_RATIO"] < 0.80).astype(float)
df["PRIOR_LABEL_OVERTAXED"]   = (df["PRIOR_SALES_RATIO"] > 1.20).astype(float)

# Repeat sale flag
df["IS_REPEAT_SALE"] = df["PRIOR_SALE_PRICE"].notna().astype(int)

print(f"\nRepeat sales in dataset: {df['IS_REPEAT_SALE'].mean():.1%}")

# ── 7b. Lagged neighborhood market stats (prior year median price/sqft) ───────
# For each sale in year Y, compute the median price/sqft in that neighborhood
# from all sales in year Y-1. Completely safe — no current sale info.

nbhd_yearly_price = (
    df.groupby(["NEIGHBORHOOD", "SALE_YEAR"])
    .agg(
        NBHD_MEDIAN_PRICE_PSQFT = ("PRICE_PER_SQFT", "median"),
        NBHD_PRICE_P25          = ("PRICE_PER_SQFT", lambda x: x.quantile(0.25)),
        NBHD_PRICE_P75          = ("PRICE_PER_SQFT", lambda x: x.quantile(0.75)),
        NBHD_SALE_COUNT         = ("PRICE_PER_SQFT", "count"),
    )
    .reset_index()
)
# Shift year forward: year Y stats are features for year Y+1 sales
nbhd_yearly_price["SALE_YEAR"] = nbhd_yearly_price["SALE_YEAR"] + 1

df = df.merge(nbhd_yearly_price, on=["NEIGHBORHOOD", "SALE_YEAR"], how="left")

df["LOG_NBHD_MEDIAN_PRICE"] = np.log1p(df["NBHD_MEDIAN_PRICE_PSQFT"].fillna(0))
df["NBHD_PRICE_IQR"]        = df["NBHD_PRICE_P75"] - df["NBHD_PRICE_P25"]

# How does current assessment compare to last year's market price?
# Assessment is known before sale; prior-year price is historical — clean
df["ASSESS_VS_PRIOR_MARKET"] = (
    df["ASSESS_PER_SQFT"] / df["NBHD_MEDIAN_PRICE_PSQFT"].clip(lower=1)
)

# Same at zip code level (more granular)
zip_yearly_price = (
    df.groupby(["ZIP CODE", "SALE_YEAR"])["PRICE_PER_SQFT"]
    .median().reset_index()
    .rename(columns={"PRICE_PER_SQFT": "ZIP_MEDIAN_PRICE_PSQFT"})
)
zip_yearly_price["SALE_YEAR"] = zip_yearly_price["SALE_YEAR"] + 1

df = df.merge(zip_yearly_price, on=["ZIP CODE", "SALE_YEAR"], how="left")
df["LOG_ZIP_MEDIAN_PRICE"] = np.log1p(df["ZIP_MEDIAN_PRICE_PSQFT"].fillna(0))

# ── 7c. Lagged neighborhood assessment fairness (prior year median ratio) ─────
# "How unfair were assessments in this neighborhood last year?"
# Uses sales_ratio from prior sales — clean

nbhd_yearly_ratio = (
    df.groupby(["NEIGHBORHOOD", "SALE_YEAR"])
    .agg(
        NBHD_MEDIAN_RATIO    = ("sales_ratio", "median"),
        NBHD_RATIO_STD       = ("sales_ratio", "std"),
        NBHD_UNDERTAXED_RATE = ("sales_ratio", lambda x: (x < 0.80).mean()),
        NBHD_OVERTAXED_RATE  = ("sales_ratio", lambda x: (x > 1.20).mean()),
    )
    .reset_index()
)
nbhd_yearly_ratio["SALE_YEAR"] = nbhd_yearly_ratio["SALE_YEAR"] + 1

df = df.merge(nbhd_yearly_ratio, on=["NEIGHBORHOOD", "SALE_YEAR"], how="left")

# Borough + tax class lagged ratio
borough_class_yearly_ratio = (
    df.groupby(["BOROUGH", "TAX_CLASS_INT", "SALE_YEAR"])["sales_ratio"]
    .median().reset_index()
    .rename(columns={"sales_ratio": "BOROUGH_CLASS_RATIO_PRIOR"})
)
borough_class_yearly_ratio["SALE_YEAR"] = borough_class_yearly_ratio["SALE_YEAR"] + 1

df = df.merge(
    borough_class_yearly_ratio,
    on=["BOROUGH", "TAX_CLASS_INT", "SALE_YEAR"],
    how="left"
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Static spatial aggregates (from FINACTTOT only — no sale price)
# ─────────────────────────────────────────────────────────────────────────────
nbhd_av = (
    df.groupby("NEIGHBORHOOD")["ASSESS_PER_SQFT"]
    .median().reset_index()
    .rename(columns={"ASSESS_PER_SQFT": "NBHD_MEDIAN_AV_SQFT"})
)
df = df.merge(nbhd_av, on="NEIGHBORHOOD", how="left")
df["VS_NBHD_AV"]     = df["ASSESS_PER_SQFT"] / df["NBHD_MEDIAN_AV_SQFT"].clip(lower=1)
df["NBHD_AV_PCTILE"] = df.groupby("NEIGHBORHOOD")["ASSESS_PER_SQFT"].rank(pct=True)

borough_class_av = (
    df.groupby(["BOROUGH", "TAX_CLASS_INT"])["ASSESS_PER_SQFT"]
    .median().reset_index()
    .rename(columns={"ASSESS_PER_SQFT": "BOROUGH_CLASS_MEDIAN_AV"})
)
df = df.merge(borough_class_av, on=["BOROUGH", "TAX_CLASS_INT"], how="left")
df["VS_BOROUGH_CLASS_AV"] = df["ASSESS_PER_SQFT"] / df["BOROUGH_CLASS_MEDIAN_AV"].clip(lower=1)

nbhd_frozen = (
    df.groupby("NEIGHBORHOOD")["AV_FROZEN_FLAG"]
    .mean().reset_index()
    .rename(columns={"AV_FROZEN_FLAG": "NBHD_FROZEN_RATE"})
)
df = df.merge(nbhd_frozen, on="NEIGHBORHOOD", how="left")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Encoding
# ─────────────────────────────────────────────────────────────────────────────
le_nbhd = LabelEncoder()
le_zip  = LabelEncoder()
le_bldg = LabelEncoder()

df["NEIGHBORHOOD_CODE"] = le_nbhd.fit_transform(df["NEIGHBORHOOD"].astype(str))
df["ZIP_CODE_CODE"]     = le_zip.fit_transform(df["ZIP CODE"].astype(str))
df["BLDG_CLASS_CODE"]   = le_bldg.fit_transform(df["BLDG_CLASS"].astype(str))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Feature sets
#
#     FEATURES_DEPLOY — zero data leak, production-ready
#                       includes lagged price/ratio features (historically safe)
#
#     SALE_PRICE_FEATURES — current transaction price signals
#                           used only during training to improve label learning
#                           never used at test/deploy time
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_DEPLOY = [
    # Location
    "BOROUGH",
    "NEIGHBORHOOD_CODE",
    "ZIP_CODE_CODE",

    # Property type
    "TAX_CLASS_INT",
    "BLDG_CLASS_CODE",
    "IS_COOP",
    "IS_MIXED",

    # Physical characteristics
    "BUILDING_AGE",
    "LOG_SQFT",
    "LOG_LAND",
    "COVERAGE_RATIO",
    "SQFT_PER_UNIT",
    "FLOOR_AREA_RATIO",
    "BLD_STORY",
    "NUM_BLDGS",
    "LOT_FRT",
    "LOT_DEP",

    # Assessment (FINACTTOT only — set before sale, never FINMKTTOT)
    "LOG_FINACTTOT",
    "ASSESS_PER_SQFT",
    "ASSESS_PER_UNIT",
    "LAND_SHARE",

    # AV cap signals (all from assessment records, not sales)
    "AV_YOY_CHANGE",
    "AV_FROZEN_FLAG",
    "AV_5YR_CHANGE",

    # Static spatial context (assessment-based)
    "VS_NBHD_AV",
    "NBHD_AV_PCTILE",
    "VS_BOROUGH_CLASS_AV",
    "NBHD_FROZEN_RATE",

    # ── Lagged features (historically safe) ───────────────────────────────
    # Prior sale of same property
    "IS_REPEAT_SALE",
    "LOG_PRIOR_PRICE",
    "PRIOR_SALES_RATIO",
    "YEARS_SINCE_PRIOR_SALE",
    "PRICE_APPRECIATION",
    "PRIOR_LABEL_UNDERTAXED",
    "PRIOR_LABEL_OVERTAXED",

    # Prior-year neighborhood market prices
    "LOG_NBHD_MEDIAN_PRICE",
    "LOG_ZIP_MEDIAN_PRICE",
    "NBHD_PRICE_IQR",
    "NBHD_SALE_COUNT",
    "ASSESS_VS_PRIOR_MARKET",

    # Prior-year neighborhood assessment fairness
    "NBHD_MEDIAN_RATIO",
    "NBHD_RATIO_STD",
    "NBHD_UNDERTAXED_RATE",
    "NBHD_OVERTAXED_RATE",
    "BOROUGH_CLASS_RATIO_PRIOR",

    # Time
    "FISCAL_YEAR_NUM",
    "SALE_YEAR",
]

# Current sale price — used only during training
SALE_PRICE_FEATURES = [
    "LOG_PRICE",
    "PRICE_PER_SQFT_CURRENT",
    "PRICE_PER_UNIT",
    "TAXABLE_AV_RATIO",
]

# Build current-sale price features (training-only)
df["LOG_PRICE"]            = np.log1p(df["SALE PRICE"])
df["PRICE_PER_SQFT_CURRENT"] = df["SALE PRICE"] / df["SQFT"].clip(lower=1)
df["PRICE_PER_UNIT"]       = df["SALE PRICE"] / df["UNITS"].clip(lower=1)
df["TAXABLE_AV_RATIO"]     = df["CURRENT_FINACTTOT"] / df["SALE PRICE"].clip(lower=1)

FEATURES_TRAIN = FEATURES_DEPLOY + SALE_PRICE_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# 11. Prepare model dataset
# ─────────────────────────────────────────────────────────────────────────────
df_model = df[FEATURES_TRAIN + ["label"]].copy()

for c in FEATURES_TRAIN:
    df_model[c] = pd.to_numeric(df_model[c], errors="coerce")
    df_model[c] = df_model[c].fillna(df_model[c].median())

df_model = df_model.dropna(subset=["label"])

X = df_model[FEATURES_TRAIN]
y = df_model["label"].astype(str)

print(f"\nFull dataset:        {X.shape}")
print(f"Training features:   {len(FEATURES_TRAIN)}")
print(f"Deployment features: {len(FEATURES_DEPLOY)}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Train / test split — stratified
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_full  = X_train[FEATURES_TRAIN]   # training: all features
X_test_deploy = X_test[FEATURES_DEPLOY]   # testing: no current sale price

print(f"\nTrain (with price): {X_train_full.shape}")
print(f"Test  (no price):   {X_test_deploy.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. Hyperparameter tuning on 60k sample
# ─────────────────────────────────────────────────────────────────────────────
sample_size  = min(60_000, len(X_train_full))
X_tune       = X_train_full.sample(sample_size, random_state=42)
y_tune       = y_train.loc[X_tune.index]
tune_weights = compute_sample_weight("balanced", y_tune)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    "max_iter":          [300, 500, 700],
    "max_depth":         [5, 7, 10, None],
    "learning_rate":     [0.02, 0.05, 0.08, 0.1],
    "min_samples_leaf":  [20, 40, 60, 100],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
    "max_features":      [0.7, 0.8, 1.0],
}

search = RandomizedSearchCV(
    HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    ),
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_tune, y_tune, sample_weight=tune_weights)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV F1 (with price): {search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. Train two final models
#     A) Full model  — trained with current sale price (upper bound)
#     B) Deploy model — trained without current sale price (production)
# ─────────────────────────────────────────────────────────────────────────────
best_params = dict(search.best_params_)
best_params.pop("early_stopping", None)
best_params.pop("n_iter_no_change", None)

train_weights = compute_sample_weight("balanced", y_train)

model_full = HistGradientBoostingClassifier(
    **best_params, early_stopping=False, random_state=42
)
model_full.fit(X_train_full, y_train, sample_weight=train_weights)

model_deploy = HistGradientBoostingClassifier(
    **best_params, early_stopping=False, random_state=42
)
model_deploy.fit(X_train[FEATURES_DEPLOY], y_train, sample_weight=train_weights)

# ─────────────────────────────────────────────────────────────────────────────
# 15. Evaluation — both tested on deploy features (honest simulation)
# ─────────────────────────────────────────────────────────────────────────────
X_test_full  = X_test[FEATURES_TRAIN]
y_pred_full  = model_full.predict(X_test_full)
y_pred_deploy = model_deploy.predict(X_test_deploy)

print("\n" + "="*60)
print("EVALUATION — test set has NO current sale price")
print("="*60)

for name, y_pred in [
    ("Full model  (trained WITH current price)", y_pred_full),
    ("Deploy model (trained WITHOUT current price)", y_pred_deploy),
]:
    print(f"\n── {name} ──")
    print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 macro:    {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────────────────────────────────────
# 16. Confusion matrix — deploy model
# ─────────────────────────────────────────────────────────────────────────────
labels_ordered = ["undertaxed", "fairly_assessed", "overtaxed"]
cm = confusion_matrix(y_test, y_pred_deploy, labels=labels_ordered)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_ordered).plot(
    ax=ax, colorbar=True, cmap="Blues"
)
ax.set_title("Confusion matrix — deploy model (lagged features, no current price)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 17. Permutation importance — deploy model
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing permutation importance (~1–2 min)...")

X_imp = X_test_deploy.sample(min(10_000, len(X_test_deploy)), random_state=42)
y_imp = y_test.loc[X_imp.index]

perm = permutation_importance(
    model_deploy, X_imp, y_imp,
    n_repeats=10,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature":    FEATURES_DEPLOY,
    "importance": perm.importances_mean,
    "std":        perm.importances_std
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 features (deploy model):")
print(importance_df.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 9))
top20 = importance_df.head(20).sort_values("importance")
ax.barh(top20["feature"], top20["importance"],
        xerr=top20["std"], color="#378ADD", alpha=0.85, capsize=3)
ax.set_xlabel("Mean decrease in F1 macro (permutation importance)")
ax.set_title("Top 20 features — deploy model (lagged, no current price)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 18. Save deploy model + encoders + feature list
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(model_deploy, f"{OUTPUT_DIR}/tax_fairness_classifier.joblib")
joblib.dump(
    {"neighborhood": le_nbhd, "zip": le_zip, "bldg_class": le_bldg},
    f"{OUTPUT_DIR}/label_encoders.joblib"
)
joblib.dump(FEATURES_DEPLOY, f"{OUTPUT_DIR}/feature_list.joblib")
print(f"\nDeploy model saved to {OUTPUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# 19. Per-borough accuracy breakdown
# ─────────────────────────────────────────────────────────────────────────────
BOROUGH_MAP = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

test_results = X_test_deploy.copy()
test_results["y_true"]       = y_test.values
test_results["y_pred"]       = y_pred_deploy
test_results["correct"]      = (test_results["y_true"] == test_results["y_pred"]).astype(int)
test_results["BOROUGH_NAME"] = test_results["BOROUGH"].map(BOROUGH_MAP)

borough_acc = (
    test_results.groupby("BOROUGH_NAME")["correct"]
    .agg(accuracy="mean", count="count")
    .round(3).reset_index()
)
print("\nAccuracy by borough (deploy model):")
print(borough_acc.to_string(index=False))

class_borough_acc = (
    test_results.groupby(["BOROUGH_NAME", "y_true"])["correct"]
    .mean().unstack().round(3)
)
print("\nPer-class accuracy by borough:")
print(class_borough_acc.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 20. Repeat vs non-repeat sale breakdown
#     Repeat sales have PRIOR_SALE_PRICE so lag features are populated
#     Non-repeat sales have median-imputed lag features
#     This tells you how much the lag features actually contribute
# ─────────────────────────────────────────────────────────────────────────────
test_results["is_repeat"] = test_results["IS_REPEAT_SALE"].astype(int)

repeat_acc = (
    test_results.groupby("is_repeat")["correct"]
    .agg(accuracy="mean", count="count")
    .round(3).reset_index()
)
repeat_acc["is_repeat"] = repeat_acc["is_repeat"].map({0: "First sale", 1: "Repeat sale"})
print("\nAccuracy by repeat sale status:")
print(repeat_acc.to_string(index=False))