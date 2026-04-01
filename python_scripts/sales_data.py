import pandas as pd
import os
import glob

data_path = "/mnt/c/Users/rodri/Desktop/UChicago/3.ThirdQuarter/Machine Learning for Public Policy/Data For Project"


# ── 1. Stack all sales files ──────────────────────────────────────────────────
sales_files = glob.glob(os.path.join(data_path, "20*.xlsx"))
print(f"Found {len(sales_files)} sales files")

dfs = []
for f in sales_files:
    temp = pd.read_excel(f, skiprows=4)
    # add year and borough from filename
    basename = os.path.basename(f)
    temp['source_file'] = basename
    dfs.append(temp)
    print(f"Read {basename}: {temp.shape}")

df_sales = pd.concat(dfs, ignore_index=True)

# Clean column names - remove newlines and extra spaces
df_sales.columns = (df_sales.columns
                    .str.replace('\n', ' ', regex=False)
                    .str.strip()
                    .str.upper()
                    .str.replace(r'\s+', ' ', regex=True))

print("Cleaned columns:", df_sales.columns.tolist())
print("Shape:", df_sales.shape)


# Keep only the real columns
cols_to_keep = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 
                'TAX CLASS AS OF FINAL ROLL 18/19', 'BLOCK', 'LOT', 
                'EASE-MENT', 'BUILDING CLASS AS OF FINAL ROLL 18/19', 
                'ADDRESS', 'APARTMENT NUMBER', 'ZIP CODE', 
                'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 
                'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT',
                'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE', 
                'SALE PRICE', 'SALE DATE', 'SOURCE_FILE']

df_sales = df_sales[cols_to_keep].copy()

# Drop rows where BOROUGH is null (leftover metadata rows)
df_sales = df_sales.dropna(subset=['BOROUGH', 'BLOCK', 'LOT', 'SALE PRICE'])

# Ensure correct types for BBL creation
df_sales['BOROUGH'] = df_sales['BOROUGH'].astype(int)
df_sales['BLOCK'] = df_sales['BLOCK'].astype(int)
df_sales['LOT'] = df_sales['LOT'].astype(int)

# Create BBL key
df_sales['BBL'] = (df_sales['BOROUGH'].astype(str) + 
                   df_sales['BLOCK'].astype(str).str.zfill(5) + 
                   df_sales['LOT'].astype(str).str.zfill(4))

# Create BBL for assessment data
df_assessment['BBL'] = (df_assessment['BORO'].astype(str) + 
                        df_assessment['BLOCK'].astype(str).str.zfill(5) + 
                        df_assessment['LOT'].astype(str).str.zfill(4))

print("Sales shape after cleaning:", df_sales.shape)
print("Sample BBL sales:", df_sales['BBL'].head())
print("\nSample BBL assessment:", df_assessment['BBL'].head())

# Check they look the same format
print("\nSales BBL length sample:", df_sales['BBL'].str.len().value_counts())
print("Assessment BBL length sample:", df_assessment['BBL'].str.len().value_counts())