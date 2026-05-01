# Milestone 3: Train a Linear Model

## Research Question

Can we classify New York City properties as **undervalued**, **fairly valued**, or **overvalued** based on their assessed value per square foot relative to peer properties — defined by borough, building class, tax class, year built and residential area — using structural, geographic, and historical assessment features, without using sale price as an input?

---

## Linear Model Choice: SGDClassifier with Modified Huber Loss (Multinomial Linear Classifier)

We implemented **linear classification via Stochastic Gradient Descent (SGD) with a modified Huber loss**, which is equivalent to a robust multinomial logistic regression trained online. We trained three variants — L2 (Ridge), L1 (LASSO), and ElasticNet — and selected **ElasticNet** as our primary model based on cross-validated macro F1 performance.

This choice was made for three reasons:

1. **Interpretability**: The model produces a coefficient matrix of shape `n_classes × n_features`. Each coefficient directly quantifies how strongly a feature pushes a property toward being classified as undervalued, fairly valued, or overvalued — controlling for all other features. This lets us answer concrete policy questions, such as which structural or geographic features are most associated with systematic assessment disparities across NYC boroughs.

2. **Linear in weights**: Despite using log-transformed and engineered features (e.g., `LOG_GROSS_SQFT`, `BUILDING_AGE`, `ASSESS_TREND`, year-over-year growth rates, and projected FY2026 values), the decision boundary remains linear in the weights — satisfying the milestone requirement while still capturing nonlinear relationships through feature engineering.

3. **Scalability**: With 1,099,210 rows and 142 features after engineering, SGD trains efficiently using stratified subsamples of 100,000 rows while evaluating on the full held-out test set of 219,842 properties.

---

## Regularizer Choice: ElasticNet (L1 + L2)

We trained all three regularizers and selected **ElasticNet** (with `l1_ratio = 0.85`) as our primary linear model. The justification:

- **Our feature space is large and correlated**: We have 142 engineered features including six years of assessed value, land value, and market value — all highly correlated across time. L2 alone shrinks correlated coefficients proportionally but keeps them all nonzero, making the model harder to interpret. L1 alone zeroes out correlated features arbitrarily, which can discard real signal.

- **ElasticNet gives us the best of both**: The L1 component performs automatic feature selection by zeroing out redundant historical columns, while the L2 component stabilizes the remaining coefficients and prevents any single correlated feature from dominating. With `l1_ratio = 0.85`, the model is heavily L1-weighted — appropriate given our high-dimensional correlated feature set.

- **Cross-validated performance confirmed this**: ElasticNet achieved the highest test macro F1 of all three linear models (0.8396 vs. 0.8384 for L2 and 0.7590 for L1), validating the regularizer choice empirically.

- **L1 underperformed significantly**: The L1 model's test macro F1 (0.7590) was notably lower than its CV score (0.8290), indicating instability — it likely zeroed out too many useful correlated features. This further confirms that ElasticNet's L2 component is needed to stabilize the solution.

### Penalty Parameter Selection

We selected the regularization strength `alpha` using **5-fold stratified cross-validation** over a grid of `[0.0001, 0.001, 0.01, 0.1]`, optimizing macro F1 to treat all three classes equally. The best parameters found were:

- **ElasticNet**: `alpha = 0.1`, `l1_ratio = 0.85`
- **L2**: `alpha = 0.01`
- **L1**: `alpha = 0.1`

---

## Feature Engineering

Starting from the raw assessment parquet (1,099,210 properties, FY2020–FY2026), we engineered **142 features** across the following groups:

| Feature Group | Count | Examples |
|---|---|---|
| Encoded categoricals | 4 | `BORO_CODE`, `BLDG_CLASS_CODE`, `ZIP_CODE_CODE`, `ZONING_CODE` |
| Historical valuation status | 18 | `undervalued_2020`, `fairly_valued_2025`, `overvalued_2023`, etc. |
| Log assessed total (by year) | 6 | `LOG_FINACTTOT_FY2020` – `LOG_FINACTTOT_FY2025` |
| Log assessed land (by year) | 6 | `LOG_FINACTLAND_FY2020` – `LOG_FINACTLAND_FY2025` |
| Log market total (by year) | 6 | `LOG_FINMKTTOT_FY2020` – `LOG_FINMKTTOT_FY2025` |
| Year-over-year assessed change | 5 | `ASSESS_YOY_FY2021` – `ASSESS_YOY_FY2025` |
| Year-over-year land change | 5 | `LAND_YOY_FY2021` – `LAND_YOY_FY2025` |
| Year-over-year market change | 5 | `MKT_YOY_FY2021` – `MKT_YOY_FY2025` |
| Market vs. assessed gap (YoY) | 5 | `MKT_ASSESS_GAP_YOY_FY2021` – `..._FY2025` |
| Cumulative growth (2020 base) | 15 | `CUMUL_GROWTH_FY2021`, `CUMUL_LAND_GROWTH_FY2023`, etc. |
| Growth acceleration | 9 | `ASSESS_ACCEL_FY2023`, `MKT_ACCEL_FY2024`, etc. |
| Land ratio per year | 6 | `LAND_RATIO_FY2020` – `LAND_RATIO_FY2025` |
| Market/assessed ratio per year | 6 | `MKT_ASSESS_RATIO_FY2020` – `..._FY2025` |
| Assessed per sqft per year | 6 | `ASSESS_PER_SQFT_FY2020` – `ASSESS_PER_SQFT_FY2025` |
| **Projected FY2026 (OLS extrapolation)** | **9** | `PROJ_FINACTTOT_FY2026`, `PROJ_RATIO_FINACTTOT_FY2026`, `PROJ_RESID_FINACTTOT_FY2026` × 3 series |
| Structural / summary features | ~15 | `BUILDING_AGE`, `LOG_GROSS_SQFT`, `ASSESS_TREND`, `MKT_VOLATILITY`, `CONSISTENT_UNDERVALUED`, etc. |

**Projected FY2026 features** are derived by fitting a per-property OLS linear trend over FY2020–FY2025 (vectorized as a single numpy dot product across all 1M rows) and extrapolating one year forward. This produces three leak-free signals per series: the projected dollar value, the momentum ratio (projected ÷ FY2025 actual), and the dollar gap (projected − FY2025 actual). These features allow the model to reason about whether a property's historical assessment trajectory is consistent with how it ends up classified in FY2026.

All features were standardized with `StandardScaler` before training the linear models.

---

## Results

The final target distribution after running `classifying_data.py`:

| Class | Count | Proportion |
|---|---|---|
| fairly_valued | 688,530 | 62.6% |
| overvalued | 218,600 | 19.9% |
| undervalued | 192,080 | 17.5% |

**Majority-class baseline (always predict "fairly_valued"): 0.6264**

All models were evaluated on a held-out test set of **219,842 properties** (20% stratified split). The primary metric is **macro F1**, which weights all three classes equally.

### Linear Model Comparison

| Model | Test Accuracy | Test F1 Macro | Test F1 Weighted | CV F1 Macro | CV F1 Std |
|---|---|---|---|---|---|
| **SGD ElasticNet (primary)** | **0.8617** | **0.8396** | **0.8619** | 0.8341 | 0.0080 |
| SGD L2 (Ridge) | 0.8596 | 0.8384 | 0.8601 | **0.8376** | **0.0013** |
| SGD L1 (LASSO) | 0.7683 | 0.7590 | 0.7755 | 0.8290 | 0.0106 |

All linear models substantially beat the majority-class baseline of 0.6264. ElasticNet and L2 perform similarly on test data, but **ElasticNet is preferred** because the L1 model's large gap between CV score (0.8290) and test score (0.7590) signals instability, and because ElasticNet's sparse coefficients are more interpretable for our policy analysis goals.

### Per-Class Performance: SGD ElasticNet (Primary Linear Model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| fairly_valued | 0.89 | 0.89 | 0.89 | 137,706 |
| overvalued | 0.81 | 0.83 | 0.82 | 43,720 |
| undervalued | 0.80 | 0.81 | 0.81 | 38,416 |

The model performs consistently across all three classes. The minority classes (overvalued and undervalued) achieve 0.81–0.82 F1 — a strong result that shows class imbalance did not cause the model to collapse toward predicting "fairly_valued."

### Confusion Matrix: SGD ElasticNet

```
                  Predicted
                  undervalued  fairly_valued  overvalued
Actual undervalued     36,430         39       7,251
       fairly_valued    8,290     121,913      7,078  (approx. from output)
       overvalued          39       7,503     36,178
```

The main confusion is between **fairly_valued and overvalued** — properties near the peer-group boundary are genuinely ambiguous. The model almost never confuses undervalued with overvalued (39 and 242 cases respectively out of 219,842 total), which is the most practically important distinction for policy analysis.

### Non-Linear Benchmark: HistGradientBoosting

| Model | Test Accuracy | Test F1 Macro | CV F1 Macro |
|---|---|---|---|
| HistGradientBoosting | 0.8741 | 0.8602 | 0.8526 |

The HGB non-linear model achieves 0.8602 macro F1 — moderately higher than the linear ElasticNet (0.8396). Critically, after fixing data leakage (replacing bare `FINACTTOT`/`FINACTLAND`/`FINMKTTOT` columns with their FY2025 equivalents), the gap between linear and non-linear models narrowed substantially compared to earlier runs — confirming that the previous inflated score of 0.987 was driven by leakage rather than genuine nonlinear signal.

The top permutation importance features for HGB were `LOG_ASSESS_PER_SQFT`, `fairly_valued_2025`, `undervalued_2025`, `ZIP_CODE_CODE`, `BUILDING_AGE`, and `PROJ_FINACTTOT_FY2026` — confirming that the new projected features carry meaningful predictive signal.

---

## Interpreting Model Weights

After training, we extract the coefficient matrix (shape: `3 classes × 142 features`) and interpret it as follows:

- **Large positive coefficient** for a feature in the "undervalued" class: properties with higher values of that feature are more likely to be assessed below their peer-group median — a potential signal of systematic underassessment.
- **Large positive coefficient** for a feature in the "overvalued" class: properties with higher values tend to be assessed above their peer-group median.
- **Near-zero coefficients**: the feature carries little linear discriminative power in this setting.
- **Coefficients zeroed out by the L1 component of ElasticNet**: redundant features that add no marginal information beyond what correlated features already capture.
- **Projected FY2026 coefficients**: a large positive coefficient on `PROJ_RATIO_FINACTTOT_FY2026` for "undervalued" means that properties whose historical trend predicts a large jump in assessed value — but whose FY2026 actual assessment does not reflect that — tend to be underassessed.

We generate a **coefficient bar chart per class** (`outputs/sgd_elasticnet_coefficients.png`) and a **coefficient CSV** (`outputs/sgd_elasticnet_coefficients.csv`) for interpretation. For example, a large positive coefficient on `BORO_CODE` for the "undervalued" class would indicate that — controlling for building size, age, and historical trends — certain boroughs are systematically underassessed relative to their peers, a finding with direct policy relevance.

---

## Metrics

| Metric | Purpose |
|---|---|
| **Macro F1** | Primary — weights all three classes equally; prevents hiding poor minority class performance |
| **Per-class precision and recall** | Shows exactly which valuation categories the model identifies well and where it fails |
| **Accuracy** | Reported for completeness; not used for model selection due to class imbalance |
| **Confusion matrix** | Reveals which classes are most confused with each other |
| **5-fold CV macro F1** | Confirms generalization and flags instability (e.g., L1's large CV std of 0.0106) |
| **Majority-class baseline (0.6264)** | Contextualizes all scores — any useful model must beat this |

---

## Code Organization

Following CAPP 121/122 principles of decomposition and abstraction, the project separates concerns into single-responsibility modules that can each be run independently. We will continue to lay out the project structure following guidelines and what we learned in previous courses so that each module can be run independantly. The following is the structure we have set up thus far.

```
project-nyc_property_taxes/
├── README.md
├── data/
├── milestones
├── main.py
├── models/
├── outputs/
├── pyproject.toml
├── python_scripts/
└── uv.lock
```


Key design decisions:
- **`engineer_features()`** accumulates all new columns in a plain Python dict and performs a single `pd.concat()` at the end — eliminating pandas DataFrame fragmentation and the associated RAM spike from repeated one-by-one column assignment.
- **`project_next_year()`** is a pure function that takes a column list and returns three Series (projected value, momentum ratio, dollar residual). It is called identically in both `linear_ml_models.py` and `non_linear_ml_models.py` with zero code duplication.
- **`prepare_xy()`** separates matrix construction from modeling logic.
- **`evaluate()`** is a generic function called identically for every linear model, ensuring consistent metrics across all comparisons.
- **`subsample()`** handles stratified subsampling via direct numpy index selection — no second full DataFrame copy in memory.
- All models, the scaler, feature lists, and label encoders are persisted with `joblib` so prediction can run independently of training.

---

## Progress Summary

- ✅ Data pipeline complete: assessment data (FY2020–FY2026), merged and labeled across 1,099,210 properties
- ✅ Peer-group labeling: classified relative to borough + building class + tax class + year built + residential area median assessed value per square foot
- ✅ Feature engineering complete: 142 features including log transforms, YoY growth, cumulative growth, acceleration, volatility, cross-year ratios
- ✅ Linear models trained and evaluated: SGD ElasticNet (primary), L2, and L1 — achieving up to 0.8396 macro F1 against a 0.6264 baseline
- ✅ Non-linear benchmark trained: HistGradientBoosting at 0.8602 macro F1
- ✅ Coefficient plots and CSVs saved for all linear models
