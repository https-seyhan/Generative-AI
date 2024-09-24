import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for a phrase
def get_bert_embedding(phrase):
    tokens = tokenizer(phrase, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    # Average of last hidden states (BERT outputs embeddings for each token, averaging gives the phrase embedding)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Example data
reference_phrases = ["phrase1", "phrase2", "phrase3"]  # Reference table
df = pd.DataFrame({'text': ["example phrase", "another phrase"]})  # DataFrame to search in

# Get embeddings for the reference phrases
ref_embeddings = [get_bert_embedding(phrase) for phrase in reference_phrases]

# Initialize columns for matches and similarity scores
df['matched_phrases'] = ''
df['similarity_scores'] = ''
df['matched_count'] = 0

# Loop through each row in the DataFrame
for idx, row in df.iterrows():
    text = row['text']
    text_embedding = get_bert_embedding(text)
    
    matched_phrases = []
    similarity_scores = []
    
    # Loop through reference phrases and compute similarity
    for i, ref_embedding in enumerate(ref_embeddings):
        similarity = cosine_similarity([text_embedding], [ref_embedding])[0][0]
        
        if similarity >= 0.8:
            matched_phrases.append(reference_phrases[i])
            similarity_scores.append(similarity)
    
    # Assign matched phrases and similarity scores to the DataFrame
    if matched_phrases:
        df.at[idx, 'matched_phrases'] = ', '.join(matched_phrases)
        df.at[idx, 'similarity_scores'] = ', '.join(map(str, similarity_scores))
        df.at[idx, 'matched_count'] = len(matched_phrases)

print(df)

