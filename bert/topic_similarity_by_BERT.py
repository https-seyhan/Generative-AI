from sentence_transformers import SentenceTransformer, util

# Load a pre-trained BERT model (e.g., 'bert-base-uncased')
model = SentenceTransformer('bert-base-uncased')

# List of words
word_list = ['translation', 'values',  'transformer', 'recurrent', 'input', 'output', 'positions', 
             'sequence', 'self', 'decoder', 'encoder', 'training', 'neural', 'keys', 'bleu', 
             'product', 'dot', 'arxiv', 'layer', 'table']

# Compute BERT embeddings for the words
word_embeddings = model.encode(word_list, convert_to_tensor=True)

# Calculate similarity scores between each pair of words
similarity_matrix = util.pytorch_cos_sim(word_embeddings, word_embeddings)

# Print similar word pairs based on similarity scores
for i in range(len(word_list)):
    for j in range(i+1, len(word_list)):
        if similarity_matrix[i][j] > 0.5:  # Adjust the similarity threshold as needed
            print(f"Similarity between '{word_list[i]}' and '{word_list[j]}': {similarity_matrix[i][j]}")
