import spacy

# Load the GloVe model (e.g., 'en_core_web_md' includes GloVe vectors)
nlp = spacy.load('en_core_web_md')

# List of words
word_list = ['translation', 'values', 'transformer', 'recurrent', 'input', 'output', 'positions', 'sequence',  'self', 'decoder', 'encoder', 'training', 'neural', 'keys', 'bleu', 'product', 'dot', 'arxiv', 'layer', 'table']

property_descriptions = [
    "This charming 3-bedroom, 2-bathroom home features hardwood floors, a spacious backyard, and a newly renovated kitchen.",
    "Stunning 2-bedroom apartment with panoramic city views, modern amenities, and rooftop access.",
    "Beautiful townhouse in a prime location, with 4 bedrooms, 3 bathrooms, and a private garage."
]

# Calculate similarity between pairs of words
similar_word_pairs = []
for word1 in word_list:
    for word2 in word_list:
        if word1 != word2:
            similarity = nlp(word1).similarity(nlp(word2))
            similar_word_pairs.append((word1, word2, similarity))

# Sort the pairs by similarity (highest first)
similar_word_pairs.sort(key=lambda x: x[2], reverse=True)

# Print the top similar word pairs
for word1, word2, similarity in similar_word_pairs[:10]:
    print(f"{word1} - {word2}: {similarity}")
