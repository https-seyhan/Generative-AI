import torch
from sklearn.metrics.pairwise import cosine_similarity

# Example embeddings (from BERT, for instance)
embedding1 = torch.rand(1, 768)  # Randomly generated example
embedding2 = torch.rand(1, 768)

# Convert to numpy arrays
embedding1_np = embedding1.detach().numpy()
embedding2_np = embedding2.detach().numpy()

# Calculate cosine similarity
cos_sim = cosine_similarity(embedding1_np, embedding2_np)

# Print cosine similarity before rounding
print("Cosine Similarity (before rounding):", cos_sim)

# Round the cosine similarity to 3 decimal places
rounded_cos_sim = round(float(cos_sim), 3)

print("Cosine Similarity (after rounding):", rounded_cos_sim)

