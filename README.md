# NYC Property Tax Assessment ML Project

**Course:** CAPP 30254 — Machine Learning for Public Policy, Spring 2026  
**Team:** Ahmed Lodhi, Faizan Imran, Rodrigo Chaves  
**Goal:** Classify NYC properties as *undervalued*, *fairly valued*, or *overvalued* by comparing each property's assessed value per square foot to the median of a peer group of structurally and geographically similar properties.

---

## Data Sources

| Dataset | Provider | Format | Coverage |
|---|---|---|---|
| Property Assessment Roll | NYC DOF | Tab-delimited `.txt` in `.zip` | FY2015–FY2026 |
| Annualized Sales | NYC DOF | `.xls` / `.xlsx` | 2015–2024 |

- Assessment roll is split into **TC1** (Class 1: 1–3 family homes, small condos) and **TC234** (Classes 2–4: rentals, co-ops, commercial, industrial). Both are stacked per year.
- Sales files have 5 borough files per year; non-arm's-length sales (price < $10,000) are excluded.
- Sales data is joined to assessment records via BBL key (~31% match rate) and used as supplementary features — not as the labeling mechanism.

---

## Pipeline Overview

### 1. Data Collection
- Assessment `.zip` files downloaded via `wget` directly into WSL; unzipped with a `for` loop.
- Sales `.xlsx`/`.xls` files downloaded per borough-year from the NYC DOF annualized sales page.

### 2. Data Processing (`processing_assessment_data.py`)
- Reads each fiscal year's TC1 and TC234 files in **100k-row chunks** (memory-safe).
- Files are Latin-1 encoded with no header row — column names mapped manually from the DOF record layout.
- Filters to `RECTYPE == "1"` (real property records only).
- Constructs a **BBL key** (`BORO` + zero-padded `BLOCK` + zero-padded `LOT`).
- Selects ~25 key columns (assessments, sqft, year built, zoning, etc.) and coerces numerics.
- Outputs one `.parquet` per fiscal year to `assessment_interim/`.

### 3. Data Merging (`merging_assessment_data.py`)
- Uses **FY2026 as the base** (most recent final roll; ~1.1M properties).
- Left-joins historical assessment values (`FINACTTOT`, `FINACTLAND`, `FINMKTTOT`) for FY2020–2025 onto the base by BBL.
- Outputs a single wide parquet: `assessment_wide.parquet`.

### 4. Sales Processing (`sales_data.py`)
- Stacks all borough-year Excel files; auto-detects header row (searches for `BOROUGH`/`BLOCK`).
- Standardizes column names, parses sale dates, drops duplicates.
- Filters: `SALE PRICE > $10,000`; constructs BBL key.
- Outputs `sales_clean.parquet`.

### 5. Labeling / Classification (`classifying_data.py`)
- Computes **assessed value per square foot** (`FINACTTOT / GROSS_SQFT`) for each property and fiscal year.
- Groups properties into peer groups using a **5-level fallback hierarchy** (finest → coarsest), falling back if group size < 10:
  1. Borough + Building Class + Tax Class + Decade Built + Size Bin + Market Value Bin
  2. Drop size bin
  3. Drop decade bin
  4. Drop market value bin
  5. Borough + Building Class only (coarsest fallback)
- **Market value bin:** `FINMKTTOT_FY2025` split into **20 ventile buckets** across all 1.1M properties. Effective price range within a peer group is much narrower because the other five dimensions filter the population first (e.g. Queens B2 1940s mid-size: $48k spread vs $475k with 5 bins).
- Labels: `overvalued` (>15% above peer median), `undervalued` (>15% below), `fairly_valued` (within ±15%).
- Generates binary status flags for FY2020–2025 (used as historical features) and a `target_2026` column.
- Outputs `processed_labeled_data.parquet`.

---

## Feature Engineering (`feature_engineering.py`)

138 features across several groups (4 removed due to redundancy or zero importance):

| Group | Examples |
|---|---|
| Property structure | `LOG_GROSS_SQFT`, `SQFT_PER_UNIT`, `COVERAGE_RATIO`, `BUILDING_AGE`, `BUILDING_ERA` |
| Assessment snapshot (FY2025) | `LOG_ASSESS_PER_SQFT`, `MKT_TO_ASSESS`, `LOG_MKT_TO_ASSESS` |
| ZIP neighborhood aggregates | `ZIP_MEAN_ASSESS`, `ZIP_ASSESS_STD`, `ASSESS_VS_ZIP_MEDIAN` |
| Building class aggregates | `ASSESS_VS_BLDG_CLASS_MEDIAN` |
| YoY % change | `ASSESS_YOY_FY{yr}`, `MKT_YOY_FY{yr}`, `LAND_YOY_FY{yr}` |
| Assessment acceleration | `ASSESS_ACCEL_FY{yr}` (2nd derivative of YoY) |
| Trend & volatility | `ASSESS_TREND`, `ASSESS_VOLATILITY`, `MKT_VS_ASSESS_TREND_SPREAD` |
| OLS projections to FY2026 | `PROJ_FINACTTOT_FY2026`, `PROJ_RATIO_*`, `PROJ_RESID_*` |
| Historical status flags | `overvalued_{yr}`, `undervalued_{yr}`, `fairly_valued_{yr}` |
| Consistency scores | `CONSISTENT_OVERVALUED`, `DOMINANT_CLASS`, `ASSESS_AT_CAP` |
| Sales features (BBL join, ~31% coverage) | `LAST_SALE_PRICE`, `SALE_TO_ASSESS_RATIO`, `SALE_PRICE_PER_SQFT`, `LOG_SALE_TO_ASSESS` |
| Interactions | `AGE_X_ASSESS_PER_SQFT` |
| Categoricals (label-encoded) | `BORO_CODE`, `BLDG_CLASS_CODE`, `ZIP_CODE_CODE`, `ZONING_CODE` |

**Removed features:** `ASSESS_PER_SQFT` (duplicate of `ASSESS_PER_SQFT_FY2025`), `LAND_TO_TOTAL` (duplicate of `LAND_RATIO_FY2025`), `LOG_PYACTTOT` (corr=0.9996 with `LOG_FINACTTOT_FY2025`), `ZIP_MEDIAN_SALE_PRICE` / `ZIP_SALE_VOLUME` / `SALE_VS_ZIP_MEDIAN` (zero LightGBM importance).

---

## Models

### Hyperparameter Tuning (`hyperparameter_tuning.py`)
- Manual tuning loop with `tqdm` progress bars.
- **Checkpoint-safe**: saves best params after every CV iteration → safe to `Ctrl+C` and resume.
- Candidate list fixed in checkpoint on first run (same combos guaranteed on resume).
- Best params persisted to `models/best_params.json` by model key.

### Linear Models (`linear_ml_models.py`)
Pipeline: `StandardScaler → PCA (60 components, ~99.2% variance) → SGDClassifier`  
Subsampled to 100k rows for tuning; trained on full train set.

| Model | Test Accuracy | Test F1 Macro | CV F1 Macro |
|---|---|---|---|
| SGD L2 | 0.8149 | 0.8027 | 0.8010 ± 0.0042 |
| SGD ElasticNet | 0.8143 | 0.8024 | 0.8006 ± 0.0037 |
| SGD L1 | 0.7909 | 0.7733 | 0.7456 ± 0.0321 |

### Non-Linear Model (`non_linear_ml_models.py`)
LightGBM classifier; subsampled to 300k rows for tuning (20 iterations × 5-fold CV).

| Model | Test Accuracy | Test F1 Macro |
|---|---|---|
| LightGBM | **0.8761** | **0.8672** |

**Top features (LightGBM):** `SQFT_PER_UNIT`, `AGE_X_ASSESS_PER_SQFT`, `ASSESS_VS_BLDG_CLASS_MEDIAN`, `ASSESS_VOLATILITY`, `ASSESS_VS_ZIP_MEDIAN`, `COVERAGE_RATIO`, `ASSESS_ACCEL_FY2022`.

**Baseline** (majority class): 0.570

---

## Target Distribution

| Class | Count | Share |
|---|---|---|
| fairly_valued | 625,030 | 56.3% |
| overvalued | 245,256 | 22.1% |
| undervalued | 227,273 | 20.5% |

Total: **1,110,445** properties (unknowns excluded from model training).

---

## Repository Structure

```
project-nyc_property_taxes/
├── python_scripts/
│   ├── processing_assessment_data.py   # Raw TXT → per-year parquets
│   ├── merging_assessment_data.py      # Wide panel merge (FY2020–2026)
│   ├── sales_data.py                   # Sales Excel files → clean parquet
│   ├── classifying_data.py             # Peer-group labeling (target + history)
│   ├── feature_engineering.py          # Shared feature pipeline (138 features)
│   ├── hyperparameter_tuning.py        # Checkpointed CV search utilities
│   ├── linear_ml_models.py             # SGD L2 / L1 / ElasticNet
│   ├── non_linear_ml_models.py         # LightGBM
│   └── diagnostics.py                  # Pipeline sanity checks (no training)
├── app_deploy/
│   ├── app.py                          # Streamlit app
│   ├── borough_summary.json            # Pre-aggregated borough stats
│   ├── sample_properties.json          # 3 demo properties (one per class)
│   └── nyc.png                         # App icon
├── data/
│   ├── assessment_wide.parquet         # Wide panel FY2020–2026
│   ├── processed_labeled_data.parquet  # Labeled dataset with features
│   └── sales_clean.parquet             # Cleaned sales data
├── models/
│   ├── lgbm_model.pkl                  # Best LightGBM model (Git LFS)
│   ├── sgd_l2.pkl / sgd_l1.pkl / sgd_elasticnet.pkl
│   ├── features.pkl                    # Feature list used at training time
│   ├── label_encoders.pkl              # Fitted label encoders
│   ├── scaler.pkl                      # Fitted StandardScaler
│   ├── linear_features.pkl             # PCA-transformed feature list
│   ├── best_params.json                # Best hyperparameters per model
│   └── tuning_checkpoint_*.json        # CV search checkpoints
├── outputs/
│   ├── lgbm_confusion_matrix.png
│   ├── lgbm_feature_importance.png
│   ├── lgbm_feature_importance.csv
│   ├── linear_model_results.csv
│   └── sgd_*_confusion_matrix.png / coefficients.csv
├── README.md
└── .gitignore
```

---

## Key Configuration

| Parameter | Value | Location |
|---|---|---|
| Valuation threshold | ±15% from peer median | `classifying_data.py` |
| Min peer group size | 10 | `classifying_data.py` |
| Market value bins | 20 ventiles | `classifying_data.py` |
| Sales price filter | > $10,000 | `sales_data.py` |
| PCA components | 60 | `linear_ml_models.py` |
| Linear subsample | 100,000 rows | `linear_ml_models.py` |
| LGBM subsample | 300,000 rows | `non_linear_ml_models.py` |
| `FORCE_RETUNE` | `False` (use cached params) | Both model scripts |

---

## Streamlit App

Live at: **https://nycpropertytaxes.streamlit.app**

Three tabs:
1. **Methodology** — model explanation, feature importance, confusion matrix
2. **Borough Analysis** — classification distributions across all 5 boroughs
3. **BBL Lookup** — explore 3 real example properties (one per class)

No full dataset is loaded in the app — all data served from pre-aggregated JSONs to stay within Streamlit Cloud memory limits.

---

## Reproducing the Data

The two large parquet files (`assessment_wide.parquet`, `processed_labeled_data.parquet`) are not tracked in git due to size (75MB each). To regenerate from scratch:

```bash
python python_scripts/processing_assessment_data.py   # raw TXT → per-year parquets
python python_scripts/merging_assessment_data.py       # → assessment_wide.parquet
python python_scripts/sales_data.py                    # → sales_clean.parquet
python python_scripts/classifying_data.py              # → processed_labeled_data.parquet
python python_scripts/feature_engineering.py           # adds engineered features
```

Raw assessment rolls and sales files must be downloaded from the NYC DOF website first (links in the Data Sources section above).
