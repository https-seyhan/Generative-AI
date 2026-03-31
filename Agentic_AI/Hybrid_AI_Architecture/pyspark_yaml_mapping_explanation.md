# PySpark YAML Mapping Engine — Detailed Explanation

## Overview
This document explains a **metadata-driven transformation function** used in PySpark.  
It interprets a YAML configuration and applies transformations to a DataFrame.

---

## Source Code

```python
from pyspark.sql import functions as F

def apply_mapping(df, mapping):

    # 1. Rename columns
    for col_map in mapping["columns"]:
        df = df.withColumnRenamed(col_map["source"], col_map["target"])

    # 2. Type casting
    for col_map in mapping["columns"]:
        if "type" in col_map:
            df = df.withColumn(
                col_map["target"],
                F.col(col_map["target"]).cast(col_map["type"])
            )

    # 3. Transformations
    for t in mapping.get("transformations", []):
        if t["operation"] == "multiply":
            df = df.withColumn(
                t["column"],
                F.col(t["column"]) * t["value"]
            )

    # 4. Filters
    for f in mapping.get("filters", []):
        df = df.filter(f"{f['column']} {f['condition']}")

    return df
```

---

## Step-by-Step Explanation

### 1. Import Functions
```python
from pyspark.sql import functions as F
```
- Alias for PySpark built-in functions
- Used for column-level transformations (`col`, `cast`, arithmetic)

---

### 2. Function Purpose
```python
def apply_mapping(df, mapping):
```
- `df`: Input Spark DataFrame  
- `mapping`: YAML configuration (converted to dictionary)

This implements a **declarative ETL pattern**:
- Logic → YAML  
- Execution → Python  

---

### 3. Column Renaming
```python
df = df.withColumnRenamed(source, target)
```

- Renames columns based on YAML mapping  
- Spark DataFrames are immutable → each call returns a new DataFrame  

Example:
```
fname → first_name
```

---

### 4. Type Casting
```python
F.col("column").cast("type")
```

- Converts column data types dynamically  
- Ensures schema consistency  

Example:
```
"salary" → float
```

---

### 5. Transformations
```python
F.col(column) * value
```

- Applies business rules defined in YAML  
- Current implementation supports multiplication  

Example:
```
annual_income = annual_income * 1.1
```

---

### 6. Filters
```python
df.filter("column > value")
```

- Applies row-level filtering using SQL expressions  
- Flexible but not compile-time safe  

---

### 7. Return Value
```python
return df
```

- Returns transformed DataFrame  
- Spark uses **lazy execution** → no computation until an action is triggered  

---

## Execution Model (Critical)

Spark builds a **logical execution plan** instead of executing immediately.

Execution happens only when:
```python
df.show()
df.write.parquet("output/")
df.count()
```

---

## Architecture Pattern

This function represents a:

### Metadata-Driven ETL Engine

| Layer | Responsibility |
|------|---------------|
| YAML | Defines schema & rules |
| Python | Interprets rules |
| Spark | Executes at scale |

---

## Strengths

- Decouples logic from code  
- Scales across datasets  
- Easy to maintain and audit  
- Supports rapid schema changes  

---

## Limitations

### 1. Multiple DataFrame Passes
- Each loop creates a new transformation stage  
- Can impact performance  

### 2. String-Based Filters
- No validation  
- Risk of runtime errors  

### 3. Limited Transformation Support
- Only supports `multiply` operation  

---

## Recommended Improvements

### Optimised Column Mapping
```python
df = df.select([
    F.col(c["source"]).alias(c["target"])
    for c in mapping["columns"]
])
```

### Benefits
- Reduces execution stages  
- Improves performance  
- Simplifies lineage  

---

## Mental Model

> This function acts like a compiler that converts YAML rules into a Spark execution plan.

---

## Use Cases

- Data warehouse ingestion  
- Financial crime pipelines (AML/KYC)  
- Schema standardisation (ASIC / ABN datasets)  
- Migration frameworks  

---

## Author
Generated on 2026-03-31
