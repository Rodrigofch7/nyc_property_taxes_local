import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Processed Data
DATA_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
df = pd.read_parquet(DATA_PATH)

# 2. Define Features and Target
target_col = 'target_2024'
historical_status_cols = [c for c in df.columns if any(yr in c for yr in ['2020', '2021', '2022', '2023'])]
static_features = ['BORO', 'BLOCK', 'LOT', 'BLDG_CLASS', 'GROSS_SQFT', 'LAND_AREA', 'NUM_BLDGS', 'YRBUILT', 'UNITS']
features = static_features + historical_status_cols

X = df[features].copy()
y = df[target_col]

# Handle Categoricals [cite: 10-15, 205]
categorical_cols = ['BORO', 'BLDG_CLASS']
for col in categorical_cols:
    X[col] = X[col].astype('category')

# 3. Train/Test Split for Final Validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Define Hyperparameter Search Space
param_distributions = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'l2_regularization': [0.0, 0.1, 1.0],
    'min_samples_leaf': [20, 50, 100]
}

# 5. Initialize Model and Cross-Validation
base_clf = HistGradientBoostingClassifier(
    categorical_features=[col in categorical_cols for col in X.columns],
    random_state=42
)

# Use StratifiedKFold to keep class proportions stable [cite: 179-191]
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search is faster than GridSearch for large datasets
tuned_search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_distributions,
    n_iter=10, # Adjust based on your hardware; 10-20 is a good start
    cv=cv_strategy,
    scoring='f1_macro', 
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("Starting Hyperparameter Tuning and Cross-Validation...")
tuned_search.fit(X_train, y_train)

# 6. Evaluate the Best Model
best_model = tuned_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Optimized Model Performance (2024) ---")
print(f"Best Params: {tuned_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# 7. Save the Optimized Model
model_dir = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_output = os.path.join(model_dir, "valuation_classifier_optimized.pkl")
joblib.dump(best_model, model_output)
print(f"Optimized model saved to: {model_output}")