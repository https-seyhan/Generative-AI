import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Sample DataFrames
df1 = pd.DataFrame({
    'Variable_1': ['Task A', 'Task B', 'Task C'],
    'Variable_2': ['Piling', 'Excavation', 'Welding']
})

df2 = pd.DataFrame({
    'Text': ['Task A Piling works', 'Excavation for Task B', 'Welding operations for Task C']
})

# Generate embeddings for df1 and df2
df1['Embeddings'] = df1.apply(lambda row: get_bert_embedding(f"{row['Variable_1']} {row['Variable_2']}"), axis=1)
df2['Embeddings'] = df2['Text'].apply(lambda text: get_bert_embedding(text))

# Define a function to compute cosine similarity between two embeddings
def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Compare embeddings from df1 to those in df2 and store the highest similarity
results = []
for i, row1 in df1.iterrows():
    best_match = None
    best_score = -1
    for j, row2 in df2.iterrows():
        similarity = compute_similarity(row1['Embeddings'], row2['Embeddings'])
        if similarity > best_score:
            best_score = similarity
            best_match = row2['Text']
    results.append((row1['Variable_1'], row1['Variable_2'], best_match, best_score))

# Convert results into a DataFrame
results_df = pd.DataFrame(results, columns=['Variable_1', 'Variable_2', 'Best_Match_in_df2', 'Similarity_Score'])

print(results_df)

