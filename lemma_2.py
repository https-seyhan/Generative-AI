import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "The boys are running and ran faster than their friends who run daily."

# Process the text
doc = nlp(text)

# Display lemmatized words
lemmas = [token.lemma_ for token in doc]
print("Lemmatized words:", lemmas)

# Example of a lemmatization-based search
def search_lemma(doc, query):
    return [token.text for token in doc if token.lemma_ == query]

# Searching for the lemma 'run'
matches = search_lemma(doc, 'run')
print("Matches for 'run':", matches)
