from collections import defaultdict

# Build a token index
token_index = defaultdict(set)
for idx, tokens in enumerate(df['tokens']):
    for token in tokens:
        token_index[token].add(idx)

def search_tokens(search_term):
    # Retrieve row indices for the search term
    indices = token_index.get(search_term.lower(), set())
    return df.loc[list(indices)]

# Perform search
search_term = 'token'
results = search_tokens(search_term)
print(results)

