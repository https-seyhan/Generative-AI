import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(phrase):
    inputs = tokenizer(phrase, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Sample dataframes
df_ref = pd.DataFrame({
    'variable_1': ["construction task", "pile installation", "excavation work"],
    'variable_2': ["task planning", "pile drilling", "earthwork"]
})

df_search = pd.DataFrame({
    'search_variable': ["pile installation task", "earth excavation", "building construction"]
})

# Threshold for similarity
similarity_threshold = 0.8

# List to store results
df_search['matched_phrases'] = ""
df_search['similarity_score'] = ""
df_search['matched_count'] = 0

# Iterate over each row in the search dataframe
for i, search_phrase in enumerate(df_search['search_variable']):
    search_embedding = get_bert_embedding(search_phrase)
    matched_phrases = []
    similarity_scores = []
    
    # Compare with each row in the reference dataframe
    for j, ref_phrase_1 in enumerate(df_ref['variable_1']):
        ref_embedding_1 = get_bert_embedding(ref_phrase_1)
        
        # Calculate similarity
        similarity_1 = cosine_similarity(search_embedding, ref_embedding_1)[0][0]
        
        # If similarity exceeds threshold
        if similarity_1 >= similarity_threshold:
            ref_phrase_2 = df_ref.loc[j, 'variable_2']
            matched_phrases.append(ref_phrase_1)
            similarity_scores.append(similarity_1)
    
    # Store matched phrases and similarity score in the search dataframe
    df_search.at[i, 'matched_phrases'] = ", ".join(matched_phrases)
    df_search.at[i, 'similarity_score'] = ", ".join(map(str, similarity_scores))
    df_search.at[i, 'matched_count'] = len(matched_phrases)

# Display results
print(df_search)

