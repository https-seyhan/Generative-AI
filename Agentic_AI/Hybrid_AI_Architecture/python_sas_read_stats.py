import pyreadstat
import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
SAS_LIBRARY_PATH = "/data/sas/mylib"   # folder behind SAS libname
DATASET_NAME = "customers"            # SAS dataset (mylib.customers)
OUTPUT_DIR = "./profiling_output"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# BUILD FILE PATH
# =========================
sas_file = Path(SAS_LIBRARY_PATH) / f"{DATASET_NAME}.sas7bdat"

if not sas_file.exists():
    raise FileNotFoundError(f"Dataset not found: {sas_file}")

# =========================
# READ DATASET
# =========================
print("Reading dataset...")

df, meta = pyreadstat.read_sas7bdat(sas_file)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# =========================
# 1. MISSING VALUE REPORT
# =========================
missing = df.isnull().sum().to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df)) * 100
missing.sort_values("missing_pct", ascending=False)\
       .to_csv(f"{OUTPUT_DIR}/{DATASET_NAME}_missing_report.csv")

# =========================
# 2. NUMERIC VARIABLE STATS
# =========================
numeric_cols = df.select_dtypes(include="number")

if len(numeric_cols.columns) > 0:
    numeric_summary = numeric_cols.describe(
        percentiles=[.01, .05, .25, .5, .75, .95, .99]
    ).T

    numeric_summary["variance"] = numeric_cols.var()
    numeric_summary["skewness"] = numeric_cols.skew()
    numeric_summary["kurtosis"] = numeric_cols.kurt()

    numeric_summary.to_csv(f"{OUTPUT_DIR}/{DATASET_NAME}_numeric_summary.csv")

# =========================
# 3. CATEGORICAL VARIABLE STATS
# =========================
cat_cols = df.select_dtypes(include=["object","category"])

for col in cat_cols.columns:
    freq = df[col].value_counts(dropna=False).to_frame("count")
    freq["pct"] = freq["count"] / len(df) * 100

    # only save top 50 values
    freq.head(50).to_csv(f"{OUTPUT_DIR}/{DATASET_NAME}_{col}_top_values.csv")

# =========================
# 4. BASIC CORRELATION (numeric only)
# =========================
if len(numeric_cols.columns) > 1:
    corr = numeric_cols.corr()
    corr.to_csv(f"{OUTPUT_DIR}/{DATASET_NAME}_correlation_matrix.csv")

print("✅ Profiling complete. Reports saved to:", OUTPUT_DIR)
