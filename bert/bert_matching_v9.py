import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Function to get BERT embeddings
def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example DataFrames
df1 = pd.DataFrame({
    'Column1': ['construction', 'engineering', 'building'],
    'Column2': ['planning', 'design', 'development']
})

df3 = pd.DataFrame({
    'Text': ['construction management', 'architectural design', 'building planning']
})

# Choose columns from the first DataFrame
col1 = df1['Column1']  # First column to compare (from df1)
col2 = df1['Column2']  # Second column to compare (from df1)

# Loop through rows in df1
results = []
for i, (text1, text2) in enumerate(zip(col1, col2)):
    combined_text = text1 + " " + text2  # Combine both columns text for comparison
    df1_embedding = get_bert_embedding(combined_text, model, tokenizer)
    
    # Loop through rows in df3 and calculate similarity
    for j, row in df3.iterrows():
        df3_embedding = get_bert_embedding(row['Text'], model, tokenizer)
        
        # Compute cosine similarity
        similarity = cosine_similarity(df1_embedding, df3_embedding)
        
        # Check if similarity exceeds the threshold (0.9)
        if similarity >= 0.9:
            results.append({
                'df1_row': i,
                'df3_row': j,
                'similarity': similarity[0][0],
                'df1_text': combined_text,
                'df3_text': row['Text']
            })

# Convert results to a DataFrame
similarity_df = pd.DataFrame(results)

# Display the results
print(similarity_df)

