import os
import csv
import spacy
from spacy.matcher import Matcher

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)

# Pattern for libname.table
table_pattern = [
    {"TEXT": {"REGEX": "^[A-Za-z0-9_]+$"}},
    {"TEXT": "."},
    {"TEXT": {"REGEX": "^[A-Za-z0-9_]+$"}}
]

matcher.add("SAS_TABLE", [table_pattern])

ROOT_DIR = "path_to_sas_folder"
OUTPUT_FILE = "sas_tables_inventory.csv"

results = []

def extract_tables(file_path):

    tables = set()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    doc = nlp(text)

    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        tables.add(span.text)

    return tables


for root, dirs, files in os.walk(ROOT_DIR):

    for file in files:

        if file.endswith(".sas"):

            file_path = os.path.join(root, file)

            try:

                tables = extract_tables(file_path)

                for table in tables:

                    results.append({
                        "file_name": file,
                        "file_path": file_path,
                        "table_name": table
                    })

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


with open(OUTPUT_FILE, "w", newline="") as csvfile:

    fieldnames = ["file_name", "file_path", "table_name"]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    writer.writerows(results)

print("Extraction complete. Results saved to:", OUTPUT_FILE)
