import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "The children are running quickly. They ran all day yesterday."

# Process the text
doc = nlp(text)

# Display the lemmas of all tokens
print("Lemmas of all tokens:")
print([(token.text, token.lemma_) for token in doc])

# Function to search for lemmas in text
def search_lemma(doc, query_lemma):
    return [token.text for token in doc if token.lemma_ == query_lemma]

# Search for the lemma 'run'
lemmas_found = search_lemma(doc, 'run')

# Display the search results
print("\nWords matching the lemma 'run':", lemmas_found)
