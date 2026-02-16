import pyreadstat
import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
SAS_LIBRARY_PATH = "/data/sas/mylib"   # folder backing the SAS libname
DATASETS = ["customers", "transactions", "products"]  # dataset names (no .sas7bdat)

# dictionary to hold all DataFrames
dfs = {}

# =========================
# LOAD DATASETS
# =========================
for name in DATASETS:
    file_path = Path(SAS_LIBRARY_PATH) / f"{name}.sas7bdat"

    if not file_path.exists():
        print(f"Dataset not found: {name}")
        continue

    print(f"Reading {name} ...")

    df, meta = pyreadstat.read_sas7bdat(file_path)

    # store dataframe
    dfs[name] = df

    print(f"{name} loaded -> rows: {df.shape[0]}, cols: {df.shape[1]}")

print("\nFinished loading all datasets.")

# =========================
# USE DATAFRAMES
# =========================

# Example usage
customers_df = dfs["customers"]
transactions_df = dfs["transactions"]

print(customers_df.head())
print(transactions_df.describe())
