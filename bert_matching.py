import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute BERT embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the output embeddings
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Example data
reference_phrases = pd.DataFrame({'phrase': ['phrase1', 'phrase2', 'phrase3']})  # Reference table
target_df = pd.DataFrame({'variable': ['text to search in', 'another piece of text']})  # Target DataFrame

# Precompute embeddings for reference phrases
reference_phrases['embedding'] = reference_phrases['phrase'].apply(get_embedding)

# Search and append phrases based on similarity
def search_and_append_phrases(row, reference_phrases):
    target_embedding = get_embedding(row['variable'])
    similar_phrases = []

    for _, ref_row in reference_phrases.iterrows():
        similarity = cosine_similarity([target_embedding], [ref_row['embedding']])[0][0]
        if similarity > 0.8:
            similar_phrases.append(ref_row['phrase'])
    
    return ', '.join(similar_phrases) if similar_phrases else None

# Apply the function to append similar phrases
target_df['matched_phrases'] = target_df.apply(search_and_append_phrases, axis=1, reference_phrases=reference_phrases)

# Output the updated DataFrame
print(target_df)

