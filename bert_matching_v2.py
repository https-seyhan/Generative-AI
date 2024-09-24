import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample DataFrames
reference_phrases = ['build foundation', 'install piping', 'pour concrete', 'concretes']
target_phrases = ['foundation building process', 'piping installation', 'concrete pouring work']

# Convert into DataFrame
reference_df = pd.DataFrame(reference_phrases, columns=['reference_phrase'])
target_df = pd.DataFrame(target_phrases, columns=['target_phrase'])

# Function to get the BERT embeddings
def get_bert_embedding(phrase):
    inputs = tokenizer(phrase, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Get embeddings for reference and target phrases
reference_df['embedding'] = reference_df['reference_phrase'].apply(lambda x: get_bert_embedding(x))
target_df['embedding'] = target_df['target_phrase'].apply(lambda x: get_bert_embedding(x))

# Function to calculate similarity and append matching phrases
def find_similar_phrases(target_embedding):
    similarities = []
    matching_phrases = []
    for i, ref_embedding in enumerate(reference_df['embedding']):
        similarity = cosine_similarity(target_embedding, ref_embedding)[0][0]
        if similarity >= 0.8:
            matching_phrases.append(reference_df['reference_phrase'][i])
            similarities.append(similarity)
    return ', '.join(matching_phrases), ', '.join([str(s) for s in similarities])

# Apply the function to the target_df
target_df[['matched_phrases', 'similarity_scores']] = target_df['embedding'].apply(
    lambda x: pd.Series(find_similar_phrases(x)))

# Drop the embedding column to clean up the DataFrame
target_df = target_df.drop('embedding', axis=1)

# Display the final DataFrame
print(target_df)

