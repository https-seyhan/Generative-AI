import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate

# Load your user-property interaction data
# Let's assume you have a dataset with 'user_id', 'property_id', and 'interaction' (e.g., a rating or a click)
data = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3],
    'property_id': [101, 102, 101, 103, 104],
    'interaction': [5, 3, 4, 2, 5]  # This could be a rating or some form of interaction score
})

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[['user_id', 'property_id', 'interaction']], reader)

# Use Singular Value Decomposition (SVD) for collaborative filtering
algo = SVD()

# Cross-validate the model to check performance
cross_validate(algo, surprise_data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Train the model on the entire dataset
trainset = surprise_data.build_full_trainset()
algo.fit(trainset)

# Predict an interaction score for a specific user and property
user_id = 1
property_id = 103
predicted_interaction = algo.predict(user_id, property_id)
print(f"Predicted interaction score for user {user_id} and property {property_id}: {predicted_interaction.est}")

