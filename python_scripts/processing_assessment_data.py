import pandas as pd
import os
import gc

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "/mnt/c/Users/rodri/Documents/NYC Datasets"
OUTPUT_PATH = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Column names (140) ────────────────────────────────────────────────────────
COL_NAMES = [
    "PARID", "BORO", "BLOCK", "LOT", "EASE", "SUBIDENT_REUC", "RECTYPE",
    "TAXYR", "IDENT", "SUBIDENT", "ROLL_SECTION", "SECVOL",
    "PYMKTLAND", "PYMKTTOT", "PYACTLAND", "PYACTTOT", "PYACTEXTOT",
    "PYTRNLAND", "PYTRNTOT", "PYTRNEXTOT", "PYTXBTOT", "PYTXBEXTOT", "PYTAXCLASS",
    "TENMKTLAND", "TENMKTTOT", "TENACTLAND", "TENACTTOT", "TENACTEXTOT",
    "TENTRNLAND", "TENTRNTOT", "TENTRNEXTOT", "TENTXBTOT", "TENTXBEXTOT", "TENTAXCLASS",
    "CBNMKTLAND", "CBNMKTTOT", "CBNACTLAND", "CBNACTTOT", "CBNACTEXTOT",
    "CBNTRNLAND", "CBNTRNTOT", "CBNTRNEXTOT", "CBNTXBTOT", "CBNTXBEXTOT", "CBNTAXCLASS",
    "FINMKTLAND", "FINMKTTOT", "FINACTLAND", "FINACTTOT", "FINACTEXTOT",
    "FINTRNLAND", "FINTRNTOT", "FINTRNEXTOT", "FINTXBTOT", "FINTXBEXTOT", "FINTAXCLASS",
    "CURMKTLAND", "CURMKTTOT", "CURACTLAND", "CURACTTOT", "CURACTEXTOT",
    "CURTRNLAND", "CURTRNTOT", "CURTRNEXTOT", "CURTXBTOT", "CURTXBEXTOT", "CURTAXCLASS",
    "PERIOD", "NEWDROP", "NOAV", "VALREF", "BLDG_CLASS", "OWNER",
    "ZONING", "HOUSENUM_LO", "HOUSENUM_HI", "STREET_NAME", "ZIP_CODE",
    "GEOSUPPORT_RC", "STCODE", "LOT_FRT", "LOT_DEP", "LOT_IRREG",
    "BLD_FRT", "BLD_DEP", "BLD_EXT", "BLD_STORY", "CORNER",
    "LAND_AREA", "NUM_BLDGS", "YRBUILT", "YRBUILT_RANGE", "YRBUILT_FLAG",
    "YRALT1", "YRALT1_RANGE", "YRALT2", "YRALT2_RANGE",
    "COOP_APTS", "UNITS", "REUC_REF", "APTNO", "COOP_NUM",
    "CPB_BORO", "CPB_DIST", "APPT_DATE", "APPT_BORO", "APPT_BLOCK",
    "APPT_LOT", "APPT_EASE", "CONDO_NUMBER", "CONDO_SFX1", "CONDO_SFX2",
    "CONDO_SFX3", "UAF_LAND", "UAF_BLDG", "PROTEST_1", "PROTEST_2",
    "PROTEST_OLD", "ATTORNEY_GROUP1", "ATTORNEY_GROUP2", "ATTORNEY_GROUP_OLD",
    "GROSS_SQFT", "HOTEL_AREA_GROSS", "OFFICE_AREA_GROSS", "RESIDENTIAL_AREA_GROSS",
    "RETAIL_AREA_GROSS", "LOFT_AREA_GROSS", "FACTORY_AREA_GROSS",
    "WAREHOUSE_AREA_GROSS", "STORAGE_AREA_GROSS", "GARAGE_AREA",
    "OTHER_AREA_GROSS", "REUC_DESCRIPTION", "EXTRACTDT",
    "PYTAXFLAG", "TENTAXFLAG", "CBNTAXFLAG", "FINTAXFLAG", "CURTAXFLAG", "EXTRA_140"
]

# ── Only columns we need ──────────────────────────────────────────────────────
COLS_NEEDED = [
    "BBL", "BORO", "BLOCK", "LOT", "BLDG_CLASS",
    "FINACTTOT", "FINACTLAND", "FINMKTTOT",
    "PYACTTOT", "PYACTLAND",
    "GROSS_SQFT", "LAND_AREA", "NUM_BLDGS",
    "YRBUILT", "BLD_STORY", "UNITS", "COOP_APTS",
    "LOT_FRT", "LOT_DEP", "LOT_IRREG",
    "STREET_NAME", "ZIP_CODE", "ZONING",
    "FISCAL_YEAR", "TAX_YEAR"
]

NUM_COLS = [
    "FINACTTOT", "FINACTLAND", "FINMKTTOT", "GROSS_SQFT",
    "PYACTTOT", "PYACTLAND",
    "LAND_AREA", "NUM_BLDGS", "YRBUILT", "BLD_STORY",
    "UNITS", "COOP_APTS", "LOT_FRT", "LOT_DEP",
    "BORO", "BLOCK", "LOT"
]

# ── File mapping ──────────────────────────────────────────────────────────────
FILE_MAP = {
    "FY2023": {
        "tc1":      f"{DATA_PATH}/final_tc1_2023/final_tc1_2023.txt",
        "tc234":    f"{DATA_PATH}/final_tc234_2023/final_tc234_2023.txt",
        "tax_year": "2022/23"
    },
    "FY2024": {
        "tc1":      f"{DATA_PATH}/fy24_tc1/fy24_tc1.txt",
        "tc234":    f"{DATA_PATH}/fy24_tc234/fy24_tc234.txt",
        "tax_year": "2023/24"
    },
    "FY2025": {
        "tc1":      f"{DATA_PATH}/fy25_tc1/PROPMAST_TC1_T2025_FIN.TXT",
        "tc234":    f"{DATA_PATH}/fy25_tc234/PROPMAST_TC234_T2025_FIN.TXT",
        "tax_year": "2024/25"
    },
    "FY2026": {
        "tc1":      f"{DATA_PATH}/fy26_tc1/PROPMAST_TC1_T2026_FIN.TXT",
        "tc234":    f"{DATA_PATH}/fy26_tc234/PROPMAST_TC234_T2026_FIN.TXT",
        "tax_year": "2025/26"
    },
}


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

        chunk = chunk[chunk["RECTYPE"].str.strip() == "1"].copy()
        if chunk.empty:
            continue

        chunk["FISCAL_YEAR"] = fy
        chunk["TAX_YEAR"]    = tax_year

        chunk["BORO"]  = pd.to_numeric(chunk["BORO"],  errors="coerce")
        chunk["BLOCK"] = pd.to_numeric(chunk["BLOCK"], errors="coerce")
        chunk["LOT"]   = pd.to_numeric(chunk["LOT"],   errors="coerce")
        chunk = chunk.dropna(subset=["BORO", "BLOCK", "LOT"])
        chunk["BBL"] = (
            chunk["BORO"].astype(int).astype(str) +
            chunk["BLOCK"].astype(int).astype(str).str.zfill(5) +
            chunk["LOT"].astype(int).astype(str).str.zfill(4)
        )

        existing = [c for c in COLS_NEEDED if c in chunk.columns]
        chunk = chunk[existing]

        for col in NUM_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunks.append(chunk)
        print(f"  Chunk {chunk_num}: {chunk.shape[0]:,} rows kept")

    result = pd.concat(chunks, ignore_index=True).drop_duplicates()
    print(f"  Total: {result.shape[0]:,} rows, {result.shape[1]} cols")
    return result


if __name__ == "__main__":
    for fy, info in FILE_MAP.items():
        out_file = os.path.join(OUTPUT_PATH, f"assessment_{fy}.parquet")

        if os.path.exists(out_file):
            print(f"\n{fy} already processed — skipping")
            continue

        print(f"\nProcessing {fy} (tax year {info['tax_year']})...")

        # Check files exist before attempting
        missing = [k for k in ["tc1", "tc234"] if not os.path.exists(info[k])]
        if missing:
            print(f"  WARNING: Missing files for {fy}: {[info[k] for k in missing]}")
            print(f"  Skipping {fy}")
            continue

        tc1   = read_in_chunks(info["tc1"],   fy, info["tax_year"])
        tc234 = read_in_chunks(info["tc234"], fy, info["tax_year"])

        combined = pd.concat([tc1, tc234], ignore_index=True).drop_duplicates()
        print(f"\n{fy} total: {combined.shape[0]:,} rows")

        combined.to_parquet(out_file, index=False)
        print(f"Saved to {out_file}")

        del tc1, tc234, combined
        gc.collect()

    print("\nAll done!")
    print("Files saved:")
    for f in sorted(os.listdir(OUTPUT_PATH)):
        if f.endswith(".parquet"):
            size_mb = os.path.getsize(os.path.join(OUTPUT_PATH, f)) / 1e6
            print(f"  {f} ({size_mb:.1f} MB)")