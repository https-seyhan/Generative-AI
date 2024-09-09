import spacy
from spacy.matcher import Matcher

# Load SpaCy's medium or large model (they include word vectors for synonyms)
nlp = spacy.load("en_core_web_md")  # 'md' or 'lg' models include word vectors

# Initialize the Matcher
matcher = Matcher(nlp.vocab)

# Define word variations and synonyms for construction-related words
# You can add more synonyms or variations based on your needs
pattern1 = [{"LOWER": "construction"}, {"LOWER": {"IN": ["schedule", "scheduling"]}}]
pattern2 = [{"LEMMA": "build"}, {"LOWER": {"IN": ["project", "site"]}}]
pattern3 = [{"LEMMA": "foundation"}, {"IS_PUNCT": False, "OP": "*"}, {"LOWER": "work"}]

# Add patterns to matcher
matcher.add("CONSTRUCTION_PATTERN", [pattern1, pattern2, pattern3])

# Example text
text = """
Construction scheduling is essential for managing a site.
Building projects can face delays if construction schedules are not followed.
Foundation-related work is critical for any construction project.
"""

# Process the text
doc = nlp(text)

# Apply the matcher to the doc
matches = matcher(doc)

# Iterate over the matches and print the matching spans
for match_id, start, end in matches:
    span = doc[start:end]
    print("Matched span:", span.text)

# Match word similarity (for synonyms using word vectors)
query_word = nlp("scheduling")
for token in doc:
    if token.has_vector and query_word.similarity(token) > 0.7:  # Adjust similarity threshold
        print(f"Word '{token.text}' is similar to '{query_word.text}' with similarity: {query_word.similarity(token)}")
