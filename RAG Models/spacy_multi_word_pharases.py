import spacy
from spacy.matcher import PhraseMatcher

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

# Initialize PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)

# Define the phrases to search
phrases = ["construction scheduling", "project management"]

# Create Doc objects from the phrases
patterns = [nlp(text) for text in phrases]

# Add the phrases to the PhraseMatcher
phrase_matcher.add("CONSTRUCTION_TERMS", patterns)

# Example text
doc = nlp("Construction scheduling is essential for project management. Proper construction scheduling helps avoid delays.")

# Apply the PhraseMatcher to the doc
matches = phrase_matcher(doc)

# Iterate over the matches and print the matching spans
for match_id, start, end in matches:
    span = doc[start:end]
    print("Matched span:", span.text)
