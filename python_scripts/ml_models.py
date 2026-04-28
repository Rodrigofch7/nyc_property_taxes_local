import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the Processed Data
DATA_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
df = pd.read_parquet(DATA_PATH)

# 2. Define Features (X) and Target (y)
target_col = 'target_2024' # Derived from 2024 FINACTTOT 

# Explicitly exclude 2024 data from features to prevent leakage [cite: 44-48]
historical_status_cols = [c for c in df.columns if any(yr in c for yr in ['2020', '2021', '2022', '2023'])]

static_features = [
    'BORO', 'BLOCK', 'LOT', 'BLDG_CLASS', 
    'GROSS_SQFT', 'LAND_AREA', 'NUM_BLDGS', 
    'YRBUILT', 'UNITS'
]

features = static_features + historical_status_cols

X = df[features].copy()
y = df[target_col]

# 3. Categorical Handling for HistGradientBoosting [cite: 10-15, 205]
categorical_cols = ['BORO', 'BLDG_CLASS']
for col in categorical_cols:
    X[col] = X[col].astype('category')

# 4. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Model Initialization and Training
clf = HistGradientBoostingClassifier(
    categorical_features=[col in categorical_cols for col in X.columns],
    random_state=42
)

print(f"Training model on historical features...")
clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)
print("\n--- Model Performance for Year 2024 ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# 7. SAVE THE MODEL (With Directory Fix)
model_dir = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
model_output = os.path.join(model_dir, "valuation_classifier_v1.pkl")

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

joblib.dump(clf, model_output)
print(f"Model successfully saved to: {model_output}")