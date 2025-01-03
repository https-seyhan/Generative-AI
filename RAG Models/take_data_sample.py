import pandas as pd

# Example DataFrame
data = {
    'task_id': [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
    'value': range(1, 57)
}

df = pd.DataFrame(data)

# Step 1: Sample 50 unique task IDs
unique_task_ids = df['task_id'].drop_duplicates().sample(n=50, random_state=42)

# Step 2: Filter the DataFrame based on these task IDs
sampled_df = df[df['task_id'].isin(unique_task_ids)]

# Display the sampled DataFrame
print(sampled_df)
