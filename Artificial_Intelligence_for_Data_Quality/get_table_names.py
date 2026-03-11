import os
import re
import csv

# Root directory containing SAS programs
ROOT_DIR = "path_to_sas_folder"

# Output CSV
OUTPUT_FILE = "sas_tables_inventory.csv"

# Regex pattern to capture SAS table references
# Format: libname.table
table_pattern = re.compile(r'\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b')

results = []

def extract_tables_from_file(file_path):
    tables = set()

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()

        matches = table_pattern.findall(content)

        for lib, table in matches:
            tables.add(f"{lib}.{table}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return tables


for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".sas"):

            full_path = os.path.join(root, file)

            tables = extract_tables_from_file(full_path)

            for table in tables:
                results.append({
                    "file_name": file,
                    "file_path": full_path,
                    "table_name": table
                })


# Write to CSV
with open(OUTPUT_FILE, "w", newline="") as csvfile:

    fieldnames = ["file_name", "file_path", "table_name"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in results:
        writer.writerow(row)


print(f"Extraction complete. Results saved to {OUTPUT_FILE}")
