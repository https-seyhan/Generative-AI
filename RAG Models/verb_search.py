import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "The boy is running and his friends ran quickly to catch up."

# Process the text
doc = nlp(text)

# Search for verbs
verbs = [token.text for token in doc if token.pos_ == 'VERB']

# Display the verbs
print("Verbs found:", verbs)
