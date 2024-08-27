import pandas as pd

# Sample DataFrame
data = {
    'title': ['Fast search algorithms', 'Efficient data matching', 'Tokenization techniques', 'Row-based search optimization'],
    'description': ['This article discusses fast search algorithms.', 
                    'How to match data efficiently using modern techniques.', 
                    'Various techniques for effective tokenization.', 
                    'Optimizing search operations in row-based databases.'],
    'tags': ['search, algorithms', 'data, matching', 'tokenization', 'search, optimization']
}

df = pd.DataFrame(data)

# Define search term
search_term = 'search'

# Function to perform token search and match
def search_in_dataframe(df, search_term):
    # Convert search term to lowercase for case-insensitive matching
    search_term = search_term.lower()
    
    # Tokenize the search term
    search_tokens = set(search_term.split())
    
    # Function to check if any token in the column contains the search term
    def token_match(column_value):
        tokens = set(str(column_value).lower().split())
        return not search_tokens.isdisjoint(tokens)
    
    # Apply the function to each column and filter rows
    matching_rows = df[df.apply(lambda row: any(token_match(row[col]) for col in df.columns), axis=1)]
    
    return matching_rows

# Perform search
matching_rows = search_in_dataframe(df, search_term)

print("Matching Rows:")
print(matching_rows)
