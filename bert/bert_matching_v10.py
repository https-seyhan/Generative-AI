import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other BERT models based on requirements

# DataFrames df1 and df3
# df1: First DataFrame with two columns to compare (select by their column numbers)
# df3: Third DataFrame to check similarity with, having potential NaN values

# Example DataFrames
df1 = pd.DataFrame({
    'col1': ['text1', 'text2', 'text3'],
    'col2': ['text4', 'text5', 'text6']
})

df3 = pd.DataFrame({
    'col3': ['textA', 'textB', 'textC', None],  # None represents NaN
})

# Select the columns from df1 by their column numbers (e.g., 0 and 1 for first and second columns)
df1_col1 = df1.iloc[:, 0].fillna('')  # Replace NaN with empty strings to avoid errors
df1_col2 = df1.iloc[:, 1].fillna('')

# Handle NaN values in df3
df3_col = df3['col3'].fillna('')  # Replace NaN with empty strings for processing

# Encode the text into BERT embeddings
embeddings_df1_col1 = model.encode(df1_col1.to_list(), convert_to_tensor=True)
embeddings_df1_col2 = model.encode(df1_col2.to_list(), convert_to_tensor=True)
embeddings_df3_col = model.encode(df3_col.to_list(), convert_to_tensor=True)

# Calculate cosine similarity for the first variable
similarity_1 = cosine_similarity(embeddings_df1_col1, embeddings_df3_col)

# Calculate cosine similarity for the second variable
similarity_2 = cosine_similarity(embeddings_df1_col2, embeddings_df3_col)

# Set threshold for matching
threshold = 0.9

# Filter rows where either column similarity is above the threshold
matches = np.logical_or(similarity_1 >= threshold, similarity_2 >= threshold)

# Create a DataFrame to view the results
result = pd.DataFrame(matches, index=df1.index, columns=df3.index)

# Print results where matches were found
matched_rows = result.apply(lambda x: x[x].index.tolist(), axis=1)
print(matched_rows)

