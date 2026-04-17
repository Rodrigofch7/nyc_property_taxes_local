import pandas as pd
import os
import gc

# ── Paths ─────────────────────────────────────────────────────────────
DATA_PATH   = "/mnt/c/Users/rodri/Documents/NYC Datasets"
OUTPUT_PATH = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Output file ────────────────────────────────────────────────────────
out_file = os.path.join(OUTPUT_PATH, "assessment_FY2023.parquet")



def read_in_chunks(filepath, fy, tax_year, chunksize=100_000):
    """Read large assessment file in chunks to avoid RAM issues."""
    basename = os.path.basename(filepath)
    print(f"  Reading {basename} in chunks...")

    with open(filepath, "r", encoding="latin-1") as f:
        first_line = f.readline()
    n_cols = len(first_line.split("\t"))
    print(f"  Detected {n_cols} columns")

    names_buffered = COL_NAMES[:n_cols] + [f"EXTRA_{i}" for i in range(1, 21)]

    chunks = []
    chunk_num = 0

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=names_buffered,
        dtype=str,
        encoding="latin-1",
        on_bad_lines="skip",
        engine="python",
        chunksize=chunksize,
        quoting=3
    ):
        chunk_num += 1

        # Filter to ordinary real estate only
        chunk = chunk[chunk["RECTYPE"].str.strip() == "1"].copy()

        if chunk.empty:
            continue

        # Add metadata
        chunk["FISCAL_YEAR"] = fy
        chunk["TAX_YEAR"]    = tax_year

        # Create BBL
        chunk["BORO"]  = pd.to_numeric(chunk["BORO"],  errors="coerce")
        chunk["BLOCK"] = pd.to_numeric(chunk["BLOCK"], errors="coerce")
        chunk["LOT"]   = pd.to_numeric(chunk["LOT"],   errors="coerce")
        chunk = chunk.dropna(subset=["BORO", "BLOCK", "LOT"])
        chunk["BBL"] = (
            chunk["BORO"].astype(int).astype(str) +
            chunk["BLOCK"].astype(int).astype(str).str.zfill(5) +
            chunk["LOT"].astype(int).astype(str).str.zfill(4)
        )

        # Keep only needed columns
        existing = [c for c in COLS_NEEDED if c in chunk.columns]
        chunk = chunk[existing]

        # Convert numeric columns
        for col in NUM_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunks.append(chunk)
        print(f"  Chunk {chunk_num}: {chunk.shape[0]:,} rows kept")

    result = pd.concat(chunks, ignore_index=True).drop_duplicates()
    print(f"  Total: {result.shape[0]:,} rows, {result.shape[1]} cols")
    return result


if __name__ == "__main__":
    out_file = os.path.join(OUTPUT_PATH, "assessment_FY2024.parquet")

    if os.path.exists(out_file):
        print("FY2024 already processed — skipping")
    else:
        print("Processing FY2024 (tax year 2023/24)...")

        tc1 = read_in_chunks(
            f"{DATA_PATH}/fy24_tc1/fy24_tc1.txt", "FY2024", "2023/24"
        )
        tc234 = read_in_chunks(
            f"{DATA_PATH}/fy24_tc234/fy24_tc234.txt", "FY2024", "2023/24"
        )

        combined = pd.concat([tc1, tc234], ignore_index=True).drop_duplicates()
        print(f"\nFY2024 total: {combined.shape[0]:,} rows")
        print(f"\nKey stats:\n{combined[['FINACTTOT', 'GROSS_SQFT', 'YRBUILT']].describe()}")

        combined.to_parquet(out_file, index=False)
        print(f"\nSaved to {out_file}")

        del tc1, tc234, combined
        gc.collect()




# ── Run only if not already created ────────────────────────────────────
if os.path.exists(out_file):
    print("FY2023 already exists — skipping")
else:
    print("Processing FY2023 (tax year 2022/23)...")

    # ── Load chunks ────────────────────────────────────────────────────
    tc1 = read_in_chunks(
        f"{DATA_PATH}/fy23_tc1/fy23_tc1.txt",
        "FY2023",
        "2022/23"
    )

    tc234 = read_in_chunks(
        f"{DATA_PATH}/fy23_tc234/fy23_tc234.txt",
        "FY2023",
        "2022/23"
    )

    # ── Combine ────────────────────────────────────────────────────────
    combined = pd.concat([tc1, tc234], ignore_index=True).drop_duplicates()

    print(f"\nFY2023 total: {combined.shape[0]:,} rows")
    print(f"\nKey stats:\n{combined[['FINACTTOT', 'GROSS_SQFT', 'YRBUILT']].describe()}")

    # ── Save ───────────────────────────────────────────────────────────
    combined.to_parquet(out_file, index=False)
    print(f"\nSaved to {out_file}")

    # ── Cleanup ────────────────────────────────────────────────────────
    del tc1, tc234, combined
    gc.collect()