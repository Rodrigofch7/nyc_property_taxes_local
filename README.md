# NYC Property Tax Assessment ML Project

**Course:** CAPP 30254 — Machine Learning for Public Policy, Spring 2026  
**Team:** Ahmed Lodhi, Faizan Imran, Rodrigo Chaves  
**Goal:** Classify NYC properties as *undervalued*, *fairly valued*, or *overvalued* by comparing assessed values against peer-group medians derived from actual sale prices.

---

## Data Sources

| Dataset | Provider | Format | Coverage |
|---|---|---|---|
| Property Assessment Roll | NYC DOF | Tab-delimited `.txt` in `.zip` | FY2015–FY2027 |
| Annualized Sales | NYC DOF | `.xls` / `.xlsx` | 2015–2024 |

- Assessment roll is split into **TC1** (Class 1: 1–3 family homes, small condos) and **TC234** (Classes 2–4: rentals, co-ops, commercial, industrial). Both must be stacked per year.
- FY2027 is the **tentative** roll (Jan 2026); all others are final (May of assessment year).
- Sales files have 5 borough files per year; non-arms-length sales (price < $10,000) are excluded.

---

## Pipeline Overview

### 1. Data Collection
- Assessment `.zip` files downloaded via `wget` directly into WSL; unzipped with a `for` loop.
- Sales `.xlsx`/`.xls` files downloaded per borough-year from the NYC DOF annualized sales page.

### 2. Data Processing (`processing_assessment_data.py`)
- Reads each fiscal year's TC1 and TC234 files in **100k-row chunks** (memory-safe).
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
- Computes **assessed value per square foot** for each property and fiscal year.
- Groups properties into peer groups using a **6-level fallback hierarchy** (finest → coarsest), falling back if group size < 10:
  1. Borough + Building Class + Tax Class + Decade Built + Size Bin + Market Value Bin
  2. Drop size bin
  3. Drop decade bin
  4. Drop market value bin
  5. Borough + Building Class
- Labels: `overvalued` (>15% above peer median), `undervalued` (>15% below), `fairly_valued` (within ±15%).
- Generates binary status flags for FY2020–2025 (historical features) and a `target_2026` column.
- Outputs `processed_labeled_data.parquet`.

---

## Feature Engineering (`feature_engineering.py`)

Features across several groups (4 removed vs prior version due to redundancy or zero importance):

| Group | Examples |
|---|---|
| Property structure | `LOG_GROSS_SQFT`, `SQFT_PER_UNIT`, `COVERAGE_RATIO`, `BUILDING_AGE`, `BUILDING_ERA` |
| Assessment snapshot (FY2025) | `LOG_ASSESS_PER_SQFT`, `MKT_TO_ASSESS`, `LOG_MKT_TO_ASSESS` |
| ZIP neighborhood aggregates | `ZIP_MEAN_ASSESS`, `ZIP_ASSESS_STD`, `ASSESS_VS_ZIP_MEDIAN` |
| Building class aggregates | `BLDG_CLASS_MEDIAN_ASSESS`, `ASSESS_VS_BLDG_CLASS_MEDIAN` |
| YoY % change | `ASSESS_YOY_FY{yr}`, `MKT_YOY_FY{yr}`, `LAND_YOY_FY{yr}` |
| Assessment acceleration | `ASSESS_ACCEL_FY{yr}` (2nd derivative of YoY) |
| Trend & volatility | `ASSESS_TREND`, `ASSESS_VOLATILITY`, `MKT_VS_ASSESS_TREND_SPREAD` |
| OLS projections to FY2026 | `PROJ_FINACTTOT_FY2026`, `PROJ_RATIO_*`, `PROJ_RESID_*` |
| Historical status flags | `overvalued_{yr}`, `undervalued_{yr}`, `fairly_valued_{yr}` |
| Consistency scores | `CONSISTENT_OVERVALUED`, `DOMINANT_CLASS`, `ASSESS_AT_CAP` |
| Sales features (BBL join) | `LAST_SALE_PRICE`, `SALE_TO_ASSESS_RATIO`, `SALE_PRICE_PER_SQFT`, `LOG_SALE_TO_ASSESS` |
| Interactions | `AGE_X_ASSESS_PER_SQFT` |
| Categoricals (label-encoded) | `BORO_CODE`, `BLDG_CLASS_CODE`, `ZIP_CODE_CODE`, `ZONING_CODE` |

**Removed features:** `ASSESS_PER_SQFT` (duplicate of `ASSESS_PER_SQFT_FY2025`), `LAND_TO_TOTAL` (duplicate of `LAND_RATIO_FY2025`), `LOG_PYACTTOT` (corr=0.9996 with `LOG_FINACTTOT_FY2025`), `ZIP_MEDIAN_SALE_PRICE` / `ZIP_SALE_VOLUME` / `SALE_VS_ZIP_MEDIAN` (zero LightGBM importance).

BBL match rate for sales features: ~31% (338k of 1.1M properties have a qualifying sale).

---

## Models

### Hyperparameter Tuning (`hyperparameter_tuning.py`)
- Manual tuning loop with `tqdm` progress bars.
- **Checkpoint-safe**: saves best params after every CV iteration → safe to `Ctrl+C` and resume.
- Candidate list fixed in checkpoint on first run (same combos guaranteed on resume).
- Best params persisted to `models/best_params.json` by model key.

### Linear Models (`linear_ml_models.py`)
Pipeline: `StandardScaler → PCA (60 components, ~99.2% variance) → SGDClassifier`  
Subsampled to 100k rows for tuning; trained on full 877k train set.

| Model | Test Accuracy | Test F1 Macro | CV F1 Macro |
|---|---|---|---|
| SGD L2 | 0.8188 | 0.8149 | 0.8135 ± 0.0024 |
| SGD ElasticNet | 0.8182 | 0.8147 | 0.8136 ± 0.0028 |
| SGD L1 | 0.7584 | 0.7520 | 0.7581 ± 0.0279 |

### Non-Linear Model (`non_linear_ml_models.py`)
LightGBM classifier; subsampled to 300k rows for tuning (20 iterations × 5-fold CV).

| Model | Test Accuracy | Test F1 Macro |
|---|---|---|
| LightGBM | **0.8822** | **0.8790** |

**Top features (LightGBM):** `SQFT_PER_UNIT`, `AGE_X_ASSESS_PER_SQFT`, `ASSESS_VOLATILITY`, `ASSESS_VS_BLDG_CLASS_MEDIAN`, `ASSESS_VS_ZIP_MEDIAN`, `COVERAGE_RATIO`, `ZIP_CODE_CODE`.

**Baseline** (majority class): 0.527

---

## Target Distribution

| Class | Count | Share |
|---|---|---|
| fairly_valued | 578,400 | 52.7% |
| overvalued | 269,377 | 24.6% |
| undervalued | 249,157 | 22.7% |

Total after dropping unknowns: **1,096,934** properties.

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
│   └── non_linear_ml_models.py         # LightGBM
├── data/
│   ├── assessment_wide.parquet
│   ├── processed_labeled_data.parquet
│   └── sales_clean.parquet
├── models/                             # Saved .pkl models + best_params.json
├── outputs/                            # Plots, CSVs, confusion matrices
└── README.md
```

---

## Key Configuration

| Parameter | Value | Location |
|---|---|---|
| Valuation threshold | ±15% from peer median | `classifying_data.py` |
| Min peer group size | 10 | `classifying_data.py` |
| PCA components | 60 | `linear_ml_models.py` |
| Linear subsample | 100,000 rows | `linear_ml_models.py` |
| LGBM subsample | 300,000 rows | `non_linear_ml_models.py` |
| Sales cutoff | ≤ FY2025, price ≥ $50k | `feature_engineering.py` |
| `FORCE_RETUNE` | `False` (use cached params) | Both model scripts |
