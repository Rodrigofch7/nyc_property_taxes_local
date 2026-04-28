# Milestone 3: Train a Linear Model

## Research Question (Updated)

Can we classify New York City properties as **undervalued**, **fairly valued**, or **overvalued** based on their assessed value per square foot relative to peer properties — defined by borough and building class — using structural, geographic, and historical assessment features, without using sale price as an input?

## Linear Model Choice: Multinomial Logistic Regression

We chose **multinomial logistic regression** as our linear model for three reasons:

1. **Interpretability**: The model produces a coefficient for each feature per class. This lets us directly answer policy questions — for example, a large positive coefficient on `BORO_CODE` for the "undervalued" class means that controlling for all other features, properties in certain boroughs are systematically more likely to be underassessed relative to their peers.

2. **Linear in weights**: Despite using log-transformed and engineered features (e.g., `LOG_GROSS_SQFT`, `BUILDING_AGE`, `ASSESS_TREND`), the model remains linear in its weights — satisfying the milestone requirement while still capturing non-linear relationships in the feature space.

3. **Appropriate baseline**: Before deploying more complex non-linear models (Random Forest, HistGradientBoosting), logistic regression gives us a performance baseline and helps identify which features carry the most signal in a linear setting.

## Regularizer Choice: L2 (Ridge)

We chose **L2 regularization (Ridge)** for the following reasons:

- **Our features are correlated**: Historical assessment columns across years (e.g., `LOG_FINACTTOT_FY2020` through `LOG_FINACTTOT_FY2025`) are highly correlated with each other. L1 (LASSO) would arbitrarily zero out some of these, discarding potentially useful signal. L2 shrinks all coefficients proportionally, keeping all features in the model while preventing any single correlated feature from dominating.

- **We want stable coefficients for interpretation**: Since a key goal is to interpret what drives over/under-assessment, we want stable, non-zero coefficients for all features. Ridge achieves this while still controlling overfitting.

- **L1 is better suited for feature selection, not our goal here**: We already performed feature selection manually. The regularizer's job is to prevent overfitting, not to further reduce the feature set.

### Penalty Parameter (Lambda / C)

We will select the regularization strength using **5-fold stratified cross-validation** over a log-scale grid of `C` values (where `C = 1/lambda`):

```python
from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    penalty="l2",
    multi_class="multinomial",
    solver="lbfgs",
    class_weight="balanced",   # handles class imbalance
    scoring="f1_macro",        # optimize for minority class performance
    max_iter=1000,
    random_state=42
)
```

We expect a small `C` (strong regularization) to perform best given the large number of correlated historical features.

## Interpreting Model Weights

After training, we will extract the coefficient matrix (shape: `n_classes × n_features`) and interpret it as follows:

- **Large positive coefficient** for a feature in the "undervalued" class: properties with higher values of that feature tend to be assessed below their peer group median — a potential inequity signal.
- **Large negative coefficient** for a feature in the "overvalued" class: properties with higher values are assessed below peers — consistent with the undervalued interpretation.
- **Near-zero coefficients**: the feature has little discriminative power in a linear setting (may still be useful in non-linear models).

We will produce a **coefficient plot** for each class, ranked by absolute magnitude, to visually communicate which features drive assessment disparities most strongly.

## Metrics

Given our highly imbalanced target (most properties are "undervalued"), we will use:

| Metric | Reason |
|--------|--------|
| **Macro F1** | Primary metric — weights all three classes equally, does not hide poor minority class performance |
| **Per-class precision/recall** | Shows exactly how well the model identifies each valuation category |
| **Accuracy** | Reported for completeness but not used for model selection |
| **Confusion matrix** | Shows which classes are most confused with each other |

We will also report the **majority-class baseline** (always predicting "undervalued") to contextualize model performance.

## Code Organization

Following CAPP 121/122 principles of decomposition and abstraction, the project is organized as follows:

```
project-nyc_property_taxes/
├── python_scripts/
│   ├── assessment_data_v2.py   # reads/cleans raw DOF assessment roll files
│   ├── sales_data.py           # reads/cleans DOF annualized sales files
│   ├── merging.py              # merges sales + assessment on BBL
│   ├── assessment_wide.py      # builds wide assessment table (one col per year)
│   ├── classifying_data.py     # creates peer-group labels (target variable)
│   └── ml_models.py            # trains, tunes, and evaluates all models
├── milestones/
│   ├── milestone2.md
│   └── milestone3.md
├── models/
│   ├── hgb_classifier.pkl
│   ├── rf_classifier.pkl
│   ├── features.pkl
│   └── label_encoders.pkl
└── data/                       # gitignored — too large to commit
    ├── sales_clean.parquet
    ├── assessment_wide.parquet
    ├── processed_labeled_data.parquet
    └── merged_2020_2024.parquet
```

Each script has a single responsibility and can be run independently. The pipeline runs in order:

```
assessment_data_v2.py → sales_data.py → merging.py →
assessment_wide.py → classifying_data.py → ml_models.py
```

For Milestone 3, we will add a `linear_model.py` script that:
1. Loads `processed_labeled_data.parquet`
2. Trains `LogisticRegressionCV` with L2 regularization
3. Selects the best `C` via cross-validation
4. Evaluates using macro F1 and per-class metrics
5. Outputs a coefficient interpretation table and plot

## Progress So Far

- ✅ Data pipeline complete: assessment data (FY2020–FY2026), sales data (2020–2024), merged and labeled
- ✅ Peer-group labeling implemented: properties classified relative to borough + building class median assessed value per sqft
- ✅ Non-linear models trained: HistGradientBoosting and Random Forest with hyperparameter tuning
- 🔄 Linear model (logistic regression with L2): in progress — `linear_model.py` being finalized
- 🔄 Coefficient interpretation: planned alongside linear model output
