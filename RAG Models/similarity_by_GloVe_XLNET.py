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


def query_xlnet_advanced(query, property_descriptions, model, tokenizer):
    # Combine conversation history with the current query
    full_query = " <SEP> ".join(property_descriptions + [query])
    #full_query = " <EOD> ".join(property_descriptions + [query])
    #full_query = " <CLS> ".join(property_descriptions + [query])
    
    # Tokenize and encode the sequence
    #inputs = tokenizer.encode_plus(full_query, add_special_tokens=True, return_tensors='pt')
    inputs = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors='pt')
    #inputs = tokenizer.encode_plus(query, add_special_tokens=False, return_tensors='pt')

    # Generate a sequence of tokens to predict
    #output_sequences = model.generate(input_ids=inputs['input_ids'], 
    #                                 max_length=50, 
    #                                  num_return_sequences=1)
                                      
    output_sequences = model.generate(input_ids=inputs['input_ids'], 
                                      max_length=50, num_beams=50,
                                      num_return_sequences=2)

    # Decode the output sequence
    rewritten_query = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return rewritten_query

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
