import pandas as pd
import numpy as np

# 1. Define Paths
INPUT_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/merged_2020_2024.parquet"
OUTPUT_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"

# 2. Load the Merged Dataset
df = pd.read_parquet(INPUT_PATH)

# 3. Initial Cleaning
# Remove properties with 0 square footage to prevent division errors 
df = df[df['GROSS_SQFT'] > 0].copy()

# Drop the columns previously identified as unnecessary for the model features
cols_to_drop = [
    "FINACTLAND_FY2020", "FINMKTTOT_FY2020", 
    "FINACTLAND_FY2021", "FINMKTTOT_FY2021", 
    "FINACTLAND_FY2022", "FINMKTTOT_FY2022", 
    "FINACTLAND_FY2023", "FINMKTTOT_FY2023"
]
df = df.drop(columns=cols_to_drop, errors='ignore')

# 4. Define Classification Logic
def classify_valuation(dataframe, value_col, area_col):
    """
    Classifies properties based on the ratio of Assessed Value to Square Footage[cite: 88, 276].
    Compares each property to the median of its Borough and Building Class cohort [cite: 10-15, 205].
    """
    # Calculate Assessed Value per Square Foot
    val_per_sqft = dataframe[value_col] / dataframe[area_col]
    
    # Determine the median for each Borough/Building Class group [cite: 10-15, 205]
    group_medians = val_per_sqft.groupby([dataframe['BORO'], dataframe['BLDG_CLASS']]).transform('median')
    
    # Define thresholds (15% deviation from group median)
    upper = group_medians * 1.15
    lower = group_medians * 0.85
    
    # Categorize properties
    conditions = [
        (val_per_sqft > upper),
        (val_per_sqft < lower)
    ]
    choices = ['overvalued', 'undervalued']
    
    return np.select(conditions, choices, default='fairly_valued')

# 5. Generate Labels and Features (2020 - 2024)
years = [2020, 2021, 2022, 2023, 2024]

for yr in years:
    # 2024 uses 'FINACTTOT', previous years use historical columns [cite: 88, 35-37]
    col_to_check = "FINACTTOT" if yr == 2024 else f"FINACTTOT_FY{yr}"
    
    # Calculate the status for the year
    status_label = classify_valuation(df, col_to_check, "GROSS_SQFT")
    
    if yr < 2024:
        # Create separate columns for each status as requested: overvalued_2020, etc.
        df[f'overvalued_{yr}'] = (status_label == 'overvalued').astype(int)
        df[f'undervalued_{yr}'] = (status_label == 'undervalued').astype(int)
        df[f'fairly_valued_{yr}'] = (status_label == 'fairly_valued').astype(int)
    else:
        # For 2024, keep the status as the target variable 
        df['target_2024'] = status_label

# 6. Final Feature Set Selection
# Identify columns that will be used for training (Static traits + Historical statuses)
base_features = ['BORO', 'BLOCK', 'LOT', 'GROSS_SQFT', 'LAND_AREA', 'NUM_BLDGS', 'YRBUILT', 'UNITS']
historical_features = [c for c in df.columns if any(str(y) in c for y in [2020, 2021, 2022, 2023]) if 'val' in c]

# 7. Save the Processed Dataset
df.to_parquet(OUTPUT_PATH, index=False)

print(f"Dataset preparation complete.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Target variable defined: 'target_2024'")