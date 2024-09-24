import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load the BERT model (pre-trained)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample data
reference_phrases = pd.DataFrame({
    'phrases': ['drilling', 'foundation piling', 'bored piles', 'soil testing', 'site preparation']
})

search_df = pd.DataFrame({
    'activity': ['The foundation piling work is ongoing.', 'Drilling will begin next week.', 'Excavation work started yesterday.']
})

# Generate embeddings for the reference phrases
reference_embeddings = model.encode(reference_phrases['phrases'].tolist(), convert_to_tensor=True)

# Function to perform similarity search
def find_similar_phrases(row, ref_embs, threshold=0.8):
    search_text = row['activity']
    search_embedding = model.encode(search_text, convert_to_tensor=True)

    # Calculate cosine similarity with reference phrases
    cos_sim = util.pytorch_cos_sim(search_embedding, ref_embs)
    
    # Filter matches above the threshold
    matches = []
    sim_scores = []
    for idx, sim in enumerate(cos_sim[0]):
        if sim >= threshold:
            matches.append(reference_phrases['phrases'].iloc[idx])
            sim_scores.append(float(sim))

    # Append results
    return pd.Series({
        'matched_phrases': ', '.join(matches) if matches else None,
        'similarity_scores': ', '.join([f'{score:.2f}' for score in sim_scores]) if sim_scores else None,
        'matched_count': len(matches)
    })

# Apply the similarity search on the DataFrame
search_df[['matched_phrases', 'similarity_scores', 'matched_count']] = search_df.apply(find_similar_phrases, ref_embs=reference_embeddings, axis=1)

print(search_df)

