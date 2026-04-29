# Milestone 3: Train a Linear Model

## Research Question (Updated)

Can we classify New York City properties as **undervalued**, **fairly valued**, or **overvalued** based on their assessed value per square foot relative to peer properties — defined by borough and building class — using structural, geographic, and historical assessment features, without using sale price as an input?

---

## Linear Model Choice: SGDClassifier with Modified Huber Loss (Multinomial Linear Classifier)

We implemented **linear classification via Stochastic Gradient Descent (SGD) with a modified Huber loss**, which is equivalent to a robust multinomial logistic regression trained online. We trained three variants — L2 (Ridge), L1 (LASSO), and ElasticNet — and selected **ElasticNet** as our primary model based on cross-validated macro F1 performance.

This choice was made for three reasons:

1. **Interpretability**: The model produces a coefficient matrix of shape `n_classes × n_features`. Each coefficient directly quantifies how strongly a feature pushes a property toward being classified as undervalued, fairly valued, or overvalued — controlling for all other features. This lets us answer concrete policy questions, such as which structural or geographic features are most associated with systematic assessment disparities.

2. **Linear in weights**: Despite using log-transformed and engineered features (e.g., `LOG_GROSS_SQFT`, `BUILDING_AGE`, `ASSESS_TREND`, year-over-year growth rates), the decision boundary remains linear in the weights. The model satisfies the milestone requirement while still capturing nonlinear relationships in the feature space through feature engineering.

3. **Scalability**: With 1,099,210 rows and 133 features after engineering, SGD trains efficiently using stratified subsamples of 100,000 rows while evaluating on the full held-out test set of 219,842 properties.

---

## Regularizer Choice: ElasticNet (L1 + L2)

We trained all three regularizers and selected **ElasticNet** (with `l1_ratio = 0.85`) as our primary model. Here is the justification:

- **Our feature space is large and redundant**: We have 133 engineered features, including six years of assessed value, land value, and market value — all highly correlated across time. L2 alone shrinks all correlated coefficients proportionally but keeps them all nonzero, making interpretation noisier. L1 alone zeroes out correlated features arbitrarily, which can discard real signal.

- **ElasticNet gives us the best of both**: The L1 component performs automatic feature selection by zeroing out redundant historical columns, while the L2 component stabilizes the remaining coefficients and prevents any single correlated feature from dominating. With `l1_ratio = 0.85`, the model is heavily L1-weighted, which suits our high-dimensional correlated feature set.

- **Cross-validated performance confirmed this**: ElasticNet achieved the highest macro F1 of all three linear models (see results below), validating the regularizer choice empirically, not just theoretically.

### Penalty Parameter Selection

We selected the regularization strength `alpha` (where `alpha = 1/C` in logistic regression terms) using **5-fold stratified cross-validation** over a grid of `[0.0001, 0.001, 0.01]` for alpha and `[0.15, 0.5, 0.85]` for `l1_ratio`. Cross-validation optimized macro F1 to treat all three classes equally.

The best parameters found were `alpha = 0.001` and `l1_ratio = 0.85` — a relatively small penalty with strong L1 weighting, consistent with our expectation that moderate regularization is needed given the large number of correlated historical features.

---

## Feature Engineering

Starting from the raw assessment parquet (1,099,210 properties, FY2020–FY2026), we engineered 133 features across the following groups:

| Feature Group | Count | Examples |
|---|---|---|
| Encoded categoricals | 4 | `BORO_CODE`, `BLDG_CLASS_CODE`, `ZIP_CODE_CODE` |
| Historical valuation status | 18 | `undervalued_2020`, `fairly_valued_2023`, etc. |
| Log assessed total (by year) | 6 | `LOG_FINACTTOT_FY2020` – `LOG_FINACTTOT_FY2025` |
| Log assessed land (by year) | 6 | `LOG_FINACTLAND_FY2020` – `LOG_FINACTLAND_FY2025` |
| Log market total (by year) | 6 | `LOG_FINMKTTOT_FY2020` – `LOG_FINMKTTOT_FY2025` |
| Year-over-year assessed change | 5 | `ASSESS_YOY_FY2021` – `ASSESS_YOY_FY2025` |
| Year-over-year land change | 5 | `LAND_YOY_FY2021` – `LAND_YOY_FY2025` |
| Year-over-year market change | 5 | `MKT_YOY_FY2021` – `MKT_YOY_FY2025` |
| Market vs. assessed gap (YoY) | 5 | `MKT_ASSESS_GAP_YOY_FY2021` – `..._FY2025` |
| Cumulative growth (2020 base) | 5+5+5 | `CUMUL_GROWTH_FY2021`, `CUMUL_LAND_GROWTH_FY2021`, etc. |
| Growth acceleration | 3+3+3 | `ASSESS_ACCEL_FY2023`, `MKT_ACCEL_FY2024`, etc. |
| Land ratio per year | 6 | `LAND_RATIO_FY2020` – `LAND_RATIO_FY2025` |
| Market/assessed ratio per year | 6 | `MKT_ASSESS_RATIO_FY2020` – `..._FY2025` |
| Assessed per sqft per year | 6 | `ASSESS_PER_SQFT_FY2020` – `ASSESS_PER_SQFT_FY2025` |
| Structural / summary features | ~15 | `BUILDING_AGE`, `LOG_GROSS_SQFT`, `ASSESS_TREND`, `MKT_VOLATILITY`, `CONSISTENT_UNDERVALUED`, etc. |

All features were standardized with `StandardScaler` before training linear models. The HGB non-linear model used raw (unscaled) features.

---

## Results

The target distribution was approximately balanced across three classes:

| Class | Count | Proportion |
|---|---|---|
| fairly_valued | 469,332 | 42.7% |
| overvalued | 326,600 | 29.7% |
| undervalued | 303,278 | 27.6% |

**Majority-class baseline (always predict "fairly_valued"): 0.4270**

All four models were evaluated on a held-out test set of **219,842 properties** (20% stratified split). The primary metric is **macro F1**, which weights all three classes equally regardless of their frequency.

### Model Comparison

| Model | Test Accuracy | Test F1 Macro | Test F1 Weighted | CV F1 Macro | CV F1 Std |
|---|---|---|---|---|---|
| **ElasticNet (primary)** | **0.8748** | **0.8780** | **0.8748** | **0.8752** | 0.0013 |
| SGD L1 (LASSO) | 0.8736 | 0.8769 | 0.8736 | 0.8742 | 0.0021 |
| SGD L2 (Ridge) | 0.8606 | 0.8642 | 0.8604 | 0.8600 | 0.0033 |
| HistGradientBoosting | 0.9868 | 0.9872 | 0.9868 | 0.9861 | — |

All linear models far exceed the majority-class baseline of 0.4270. The **ElasticNet model achieves 0.8780 macro F1**, meaning it correctly identifies undervalued, fairly valued, and overvalued properties at roughly 88% precision and recall on average — in a real-world dataset of over one million NYC parcels.

### Per-Class Performance: ElasticNet (Primary Linear Model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| fairly_valued | 0.86 | 0.85 | 0.85 | 93,866 |
| overvalued | 0.89 | 0.90 | 0.90 | 65,320 |
| undervalued | 0.88 | 0.89 | 0.88 | 60,656 |

The model performs consistently across all three classes, with no class severely lagging the others — a meaningful result given that the class imbalance could have biased a simpler model toward "fairly_valued."

### Confusion Matrix: ElasticNet

```
                  Predicted
                  fairly_valued  overvalued  undervalued
Actual fairly_valued   53,732      6,814         110
       overvalued       7,048     79,963       6,855
       undervalued        142      6,546      58,632
```

The main source of error is **fairly_valued vs. overvalued confusion**, which is expected: properties on the boundary of the peer-group median are genuinely ambiguous. The model almost never confuses undervalued with overvalued (110 and 142 cases respectively out of 219,842 total), which is the most practically important distinction.

### Non-Linear Benchmark: HistGradientBoosting

The HGB non-linear model achieves 0.9872 macro F1 — substantially higher than any linear model. This gap reveals the **nonlinear structure** present in assessment data that linear models cannot capture, and motivates further work with tree-based models. However, the linear model's 0.878 macro F1 is strong in its own right and yields directly interpretable coefficients that the HGB model does not.

---

## Interpreting Model Weights

After training, we extract the coefficient matrix (shape: `3 classes × 133 features`) and interpret it as follows:

- **Large positive coefficient** for a feature in the "undervalued" class: properties with higher values of that feature are more likely to be assessed below their peer-group median — a potential signal of systematic underassessment.
- **Large positive coefficient** for a feature in the "overvalued" class: properties with higher values tend to be assessed above their peer-group median.
- **Near-zero coefficients**: the feature carries little linear discriminative power (it may still matter in nonlinear interactions captured by HGB).
- **Coefficients zeroed out by L1**: redundant features that add no marginal information beyond what correlated features already capture.

We generate a **coefficient bar chart per class** (saved as `outputs/sgd_elasticnet_coefficients.png`) and a **coefficient CSV** (`outputs/sgd_elasticnet_coefficients.csv`) to support interpretation. For example, a large positive coefficient on `BORO_CODE` for "undervalued" would indicate that — controlling for building size, age, and historical trends — certain boroughs are systematically underassessed relative to peers, a finding with direct policy relevance.

---

## Metrics

| Metric | Purpose |
|---|---|
| **Macro F1** | Primary metric — weights all three classes equally; prevents the model from hiding poor performance on minority classes |
| **Per-class precision and recall** | Shows exactly which valuation categories the model identifies well and where it fails |
| **Accuracy** | Reported for completeness; not used for model selection due to class imbalance |
| **Confusion matrix** | Reveals which classes are most confused with each other |
| **5-fold CV macro F1** | Confirms generalization; tests that test-set performance is not a lucky split |
| **Majority-class baseline** | Contextualizes all scores — any useful model must beat 0.4270 |

---

## Code Organization

Following CAPP 121/122 principles of decomposition and abstraction, the project separates concerns into single-responsibility modules that can each be run independently:

```
project-nyc_property_taxes/
├── README.md
├── data/
│   ├── assessment_interim/
│   ├── assessment_wide.parquet
│   └── processed_labeled_data.parquet
├── main.py
├── models/
│   ├── features.pkl
│   ├── hgb.pkl
│   ├── label_encoders.pkl
│   ├── passive_aggressive.pkl
│   ├── scaler.pkl
│   ├── sgd_elasticnet.pkl
│   ├── sgd_l1.pkl
│   └── sgd_l2.pkl
├── outputs/
│   ├── all_model_results.csv
│   ├── hgb_feature_importance.csv
│   ├── histgradientboosting_confusion_matrix.png
│   ├── passive_aggressive_coefficients.csv
│   ├── passive_aggressive_coefficients.png
│   ├── passive_aggressive_confusion_matrix.png
│   ├── sgd_elasticnet_coefficients.csv
│   ├── sgd_elasticnet_coefficients.png
│   ├── sgd_elasticnet_confusion_matrix.png
│   ├── sgd_l1_coefficients.csv
│   ├── sgd_l1_coefficients.png
│   ├── sgd_l1_confusion_matrix.png
│   ├── sgd_l2_coefficients.csv
│   ├── sgd_l2_coefficients.png
│   └── sgd_l2_confusion_matrix.png
├── pyproject.toml
├── python_scripts/
│   ├── classifying_data.py
│   ├── merging_assessment_data.py
│   ├── ml_models.py
│   └── processing_assessment_data.py
└── uv.lock
```


Key design decisions:
- **`engineer_features()`** is a pure function: takes a DataFrame, returns a DataFrame and feature list with no side effects.
- **`prepare_xy()`** separates matrix construction from modeling.
- **`evaluate()`** is a generic function called identically for every model, ensuring consistent metrics across all comparisons.
- **`subsample()`** handles stratified subsampling in one place, keeping the main loop clean.
- All models, the scaler, feature lists, and label encoders are persisted with `joblib` so prediction can run independently of training.

---

## Progress Summary

- ✅ Data pipeline complete: assessment data (FY2020–FY2026), sales data (2020–2024), merged and labeled across 1,099,210 properties
- ✅ Peer-group labeling implemented: properties classified relative to borough + building class median assessed value per square foot
- ✅ Feature engineering complete: 133 features including log transforms, YoY growth, cumulative growth, acceleration, volatility, and cross-year ratios
- ✅ Linear models trained and evaluated: SGD L2, L1, and ElasticNet — all achieving ~0.86–0.88 macro F1
- ✅ Non-linear benchmark trained: HistGradientBoosting at 0.987 macro F1
- ✅ Coefficient plots and CSVs saved for all linear models
- ✅ All models persisted to disk for downstream use
- 🔄 Coefficient interpretation: planned as a dedicated analysis step to identify which features most strongly predict assessment disparities by borough and building class
