from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample property data with descriptions
property_data = pd.DataFrame({
    'property_id': [101, 102, 103, 104],
    'description': [
        'Luxury apartment with pool in downtown',
        'Affordable house in suburbs near school',
        'Modern condo with gym and parking',
        'Spacious home with garden and garage'
    ],
    'price': [500000, 300000, 400000, 600000],
    'bedrooms': [3, 4, 2, 5],
})

# Step 1: Convert text descriptions into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(property_data['description'])

# Step 2: Compute similarity between properties using cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get similar properties based on content
def get_content_recommendations(property_id, cosine_sim=cosine_sim):
    idx = property_data.index[property_data['property_id'] == property_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 similar properties
    similar_properties = [i[0] for i in sim_scores]
    return property_data['property_id'].iloc[similar_properties]

# Example: Get content-based recommendations for property_id 101
recommended_properties = get_content_recommendations(101)
print(f"Recommended properties based on content for property 101: {recommended_properties.tolist()}")

