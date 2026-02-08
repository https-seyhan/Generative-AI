import pandas as pd
from tqdm import tqdm
from config import DATA_PATH, FEATURE_PATH
from llm_feature_extractor import extract_fincrime_features
from embedding_generator import get_embedding

df = pd.read_csv(DATA_PATH)

rows = []

for _, r in tqdm(df.iterrows(), total=len(df)):

    text = r["narrative"]

    llm_features = extract_fincrime_features(text)
    embedding = get_embedding(text)

    record = {
        "label": r["label"],
        "amount": r["amount"],
        "customer_age": r["customer_age"]
    }

    record.update(llm_features)

    for i, val in enumerate(embedding):
        record[f"emb_{i}"] = float(val)

    rows.append(record)

features_df = pd.DataFrame(rows)
features_df.to_parquet(FEATURE_PATH, index=False)

print("Feature dataset created:", FEATURE_PATH)
