import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    # Use the [CLS] token's embedding for similarity comparison
    return outputs.last_hidden_state[:, 0, :].detach()

# Example DataFrames
df1 = pd.DataFrame({'col1': ['apple', 'banana'], 'col2': ['fruit', 'yellow']})
df2 = pd.DataFrame({'col3': ['orange', 'grapefruit'], 'col4': ['citrus', 'red']})

# Select columns by column number (e.g., col1=1 and col2=2)
col1_num = 0  # First variable column in df1
col2_num = 1  # Second variable column in df2

# Get the text columns
df1_text = df1.iloc[:, col1_num].astype(str)
df2_text = df2.iloc[:, col2_num].astype(str)

# Get BERT embeddings for both DataFrames
df1_embeddings = torch.cat([get_bert_embedding(text) for text in df1_text])
df2_embeddings = torch.cat([get_bert_embedding(text) for text in df2_text])

# Calculate cosine similarity between embeddings
similarities = cosine_similarity(df1_embeddings, df2_embeddings)

# Set similarity threshold
threshold = 0.9

# Find matches where similarity is above threshold
matches = (similarities >= threshold)

# Display matching rows with similarity score
for i, row in enumerate(matches):
    for j, is_similar in enumerate(row):
        if is_similar:
            print(f"Match between df1 row {i} and df2 row {j} with similarity: {similarities[i, j]}")


