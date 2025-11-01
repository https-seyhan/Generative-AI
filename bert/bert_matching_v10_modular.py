import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute BERT embeddings for a given sentence
def get_bert_embedding(text):
    if pd.isna(text):
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Modular function to perform BERT search
def bert_search(df1, col1, col2, df3, df3_col, similarity_threshold=0.9):
    # Extract the relevant columns from df1 and df3
    column1_texts = df1.iloc[:, col1]
    column2_texts = df1.iloc[:, col2]
    df3_texts = df3.iloc[:, df3_col]
    
    # Compute BERT embeddings for both columns
    column1_embeddings = np.array([get_bert_embedding(text) for text in column1_texts])
    column2_embeddings = np.array([get_bert_embedding(text) for text in column2_texts])
    df3_embeddings = np.array([get_bert_embedding(text) for text in df3_texts])
    
    results = []

    for i, df3_embedding in enumerate(df3_embeddings):
        if df3_embedding is None:
            continue  # Skip NaN values in df3
        for j, (emb1, emb2) in enumerate(zip(column1_embeddings, column2_embeddings)):
            if emb1 is not None and emb2 is not None:
                # Calculate cosine similarity for both column1 and column2
                sim1 = cosine_similarity([df3_embedding], [emb1])[0][0]
                sim2 = cosine_similarity([df3_embedding], [emb2])[0][0]
                # Check if both similarities are above the threshold
                if sim1 >= similarity_threshold and sim2 >= similarity_threshold:
                    results.append((i, j, sim1, sim2))
    
    return results

# Example usage
# Assuming df1 and df3 are the dataframes, col1 and col2 are column numbers of df1, and df3_col is column number in df3
df1 = pd.DataFrame({'A': ['text1 df1 A', 'text2 df1 A'], 'B': ['text1 df1 B', 'text2 df1 B']})
df3 = pd.DataFrame({'C': ['text1 df3 C', 'text2 df3 C']})

col1 = 0  # Column index for df1
col2 = 1  # Column index for df1
df3_col = 0  # Column index for df3

results = bert_search(df1, col1, col2, df3, df3_col)
print(results)

