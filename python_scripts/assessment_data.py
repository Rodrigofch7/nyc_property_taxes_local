import pandas as pd
import os
import glob

data_path = "/mnt/c/Users/rodri/Desktop/UChicago/3.ThirdQuarter/Machine Learning for Public Policy/Data For Project"
output_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"

os.makedirs(output_path, exist_ok=True)

cols_to_keep = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY',
                'BLOCK', 'LOT', 'ADDRESS', 'APARTMENT NUMBER', 'ZIP CODE',
                'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS',
                'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT',
                'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE',
                'SALE PRICE', 'SALE DATE', 'SOURCE_FILE']

sales_files = glob.glob(os.path.join(data_path, "20*.xlsx"))
print(f"Found {len(sales_files)} sales files")

dfs = []
for f in sales_files:
    basename = os.path.basename(f)
    year = int(basename[:4])
    
    # 2020+ files have different header structure
    skiprows = 4 if year <= 2019 else 7
    
    temp = pd.read_excel(f, skiprows=skiprows)
    temp['SOURCE_FILE'] = basename
    
    # Clean column names
    temp.columns = (temp.columns
                    .str.replace('\n', ' ', regex=False)
                    .str.strip()
                    .str.upper()
                    .str.replace(r'\s+', ' ', regex=True))
    
    # Keep only real columns that exist in this file
    existing_cols = [c for c in cols_to_keep if c in temp.columns]
    temp = temp[existing_cols]
    dfs.append(temp)
    print(f"Read {basename}: {temp.shape}")

df_sales = pd.concat(dfs, ignore_index=True)
print("\nStacked shape:", df_sales.shape)

# ── Clean ─────────────────────────────────────────────────────────────────────
# Drop metadata/empty rows
df_sales = df_sales.dropna(subset=['BOROUGH', 'BLOCK', 'LOT', 'SALE PRICE'])

# Remove non-arms-length sales
df_sales['SALE PRICE'] = pd.to_numeric(df_sales['SALE PRICE'], errors='coerce')
df_sales = df_sales[df_sales['SALE PRICE'] > 10000]

# Fix types for BBL
df_sales['BOROUGH'] = pd.to_numeric(df_sales['BOROUGH'], errors='coerce').astype('Int64')
df_sales['BLOCK'] = pd.to_numeric(df_sales['BLOCK'], errors='coerce').astype('Int64')
df_sales['LOT'] = pd.to_numeric(df_sales['LOT'], errors='coerce').astype('Int64')
df_sales = df_sales.dropna(subset=['BOROUGH', 'BLOCK', 'LOT'])

# Fix APARTMENT NUMBER - convert to string
df_sales['APARTMENT NUMBER'] = df_sales['APARTMENT NUMBER'].astype(str)

# Drop duplicates
df_sales = df_sales.drop_duplicates()
print("Shape after cleaning:", df_sales.shape)

# ── Create BBL key ────────────────────────────────────────────────────────────
df_sales['BBL'] = (df_sales['BOROUGH'].astype(str) +
                   df_sales['BLOCK'].astype(str).str.zfill(5) +
                   df_sales['LOT'].astype(str).str.zfill(4))

print("\nDuplicate rows:", df_sales.duplicated().sum())
print("BBL length check:\n", df_sales['BBL'].str.len().value_counts())
print("Null counts:\n", df_sales.isnull().sum())

# ── Save ──────────────────────────────────────────────────────────────────────
df_sales.to_parquet(os.path.join(output_path, "sales_clean.parquet"), index=False)
print("\nSales data saved!")