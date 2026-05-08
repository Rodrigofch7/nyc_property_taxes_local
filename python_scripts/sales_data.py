import pandas as pd
import os
import glob
import gc

# ── Paths ─────────────────────────────────────────────────────────────────────
data_path   = "/mnt/c/Users/rodri/Documents/NYC Datasets/01.sales_data"
output_path = "/home/rodrigofrancachaves/project-nyc_property_taxes/data"
os.makedirs(output_path, exist_ok=True)

# ── 1. Find all sales files (.xls and .xlsx) ──────────────────────────────────
sales_files = sorted(
    glob.glob(os.path.join(data_path, "20*.xlsx")) +
    glob.glob(os.path.join(data_path, "20*.xls"))
)
print(f"Found {len(sales_files)} sales files")


def find_header_row(filepath):
    """Find the row where BOROUGH and BLOCK appear as column headers."""
    preview = pd.read_excel(filepath, header=None, nrows=10)
    for i, row in preview.iterrows():
        row_str = ' '.join([str(v).upper() for v in row.values])
        if 'BOROUGH' in row_str and 'BLOCK' in row_str:
            return i
    return 4  # fallback


# ── 2. Stack all files ────────────────────────────────────────────────────────
dfs = []
for f in sales_files:
    basename = os.path.basename(f)
    year = int(basename[:4])

    try:
        header_row = find_header_row(f)
        temp = pd.read_excel(f, skiprows=header_row)
        temp['SOURCE_FILE'] = basename
        temp['SALE_YEAR']   = year

        # Clean column names
        temp.columns = (
            temp.columns
            .str.replace('\n', ' ', regex=False)
            .str.strip()
            .str.upper()
            .str.replace(r'\s+', ' ', regex=True)
        )

        dfs.append(temp)
        print(f"  Read {basename} (header row {header_row}): {temp.shape}")

    except Exception as e:
        print(f"  ERROR reading {basename}: {type(e).__name__}: {e}")

# ── 3. Concat ─────────────────────────────────────────────────────────────────
print(f"\nTotal files successfully read: {len(dfs)}")
if not dfs:
    raise ValueError("No files were successfully read!")

df_sales = pd.concat(dfs, ignore_index=True)
print(f"Stacked shape: {df_sales.shape}")
del dfs
gc.collect()

# ── 4. Keep only real columns ─────────────────────────────────────────────────
cols_to_keep = [
    'BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY',
    'BLOCK', 'LOT', 'ADDRESS', 'APARTMENT NUMBER', 'ZIP CODE',
    'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS',
    'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT',
    'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE',
    'SALE PRICE', 'SALE DATE', 'SOURCE_FILE', 'SALE_YEAR'
]

existing_cols = [c for c in cols_to_keep if c in df_sales.columns]
missing_cols  = [c for c in cols_to_keep if c not in df_sales.columns]
if missing_cols:
    print(f"\nWARNING: Missing columns: {missing_cols}")

df_sales = df_sales[existing_cols].copy()

# ── 5. Clean ──────────────────────────────────────────────────────────────────
df_sales = df_sales.dropna(subset=['BOROUGH', 'BLOCK', 'LOT', 'SALE PRICE'])

df_sales['SALE PRICE'] = pd.to_numeric(df_sales['SALE PRICE'], errors='coerce')
df_sales = df_sales[df_sales['SALE PRICE'] > 10000]

df_sales['BOROUGH'] = pd.to_numeric(df_sales['BOROUGH'], errors='coerce').astype('Int64')
df_sales['BLOCK']   = pd.to_numeric(df_sales['BLOCK'],   errors='coerce').astype('Int64')
df_sales['LOT']     = pd.to_numeric(df_sales['LOT'],     errors='coerce').astype('Int64')
df_sales = df_sales.dropna(subset=['BOROUGH', 'BLOCK', 'LOT'])

if 'APARTMENT NUMBER' in df_sales.columns:
    df_sales['APARTMENT NUMBER'] = (
        df_sales['APARTMENT NUMBER']
        .astype(str).str.strip().replace('nan', '')
    )

if 'SALE DATE' in df_sales.columns:
    df_sales['SALE DATE'] = pd.to_datetime(df_sales['SALE DATE'], errors='coerce')

for col in ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS',
            'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT']:
    if col in df_sales.columns:
        df_sales[col] = pd.to_numeric(df_sales[col], errors='coerce')

df_sales = df_sales.drop_duplicates()

# ── 6. Create BBL key ─────────────────────────────────────────────────────────
df_sales['BBL'] = (
    df_sales['BOROUGH'].astype(str) +
    df_sales['BLOCK'].astype(str).str.zfill(5) +
    df_sales['LOT'].astype(str).str.zfill(4)
)

# ── 7. Summary ────────────────────────────────────────────────────────────────
print(f"\nShape after cleaning: {df_sales.shape}")
print(f"\nSales by year:\n{df_sales['SALE_YEAR'].value_counts().sort_index()}")
print(f"\nBBL length check:\n{df_sales['BBL'].str.len().value_counts()}")
print(f"\nNull counts:\n{df_sales.isnull().sum()[df_sales.isnull().sum() > 0]}")

# ── 8. Save ───────────────────────────────────────────────────────────────────
out_file = os.path.join(output_path, "sales_clean.parquet")
df_sales.to_parquet(out_file, index=False)
print(f"\nSales data saved to: {out_file}")
print(f"Final shape: {df_sales.shape}")