import spacy
from spacy.matcher import Matcher
import re
from transformers import (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)
from spacy.tokenizer import Tokenizer

#tokenizer = XLNetTokenizer.from_pretrained(model_name)
# Load the GloVe model (e.g., 'en_core_web_md' includes GloVe vectors)
nlp = spacy.load('en_core_web_md')

# List of words
word_list = ['translation', 'values', 'transformer', 'recurrent', 'input', 'output', 'positions', 'sequence',  'self', 'decoder', 'encoder', 'training', 'neural', 'keys', 'bleu', 'product', 'dot', 'arxiv', 'layer', 'table']

property_descriptions = [
    "This charming 3-bedroom, 2-bathroom home features hardwood floors, a spacious backyard, and a newly renovated kitchen.",
    "Stunning 2-bedroom apartment with panoramic city views, modern amenities, and rooftop access.",
    "Beautiful townhouse in a prime location, with 4 bedrooms, 3 bathrooms, and a private garage."
]

current_query = "I'm looking for a family-friendly home with a backyard. Do you have any properties like that?"

property_descriptions = "This charming 3-bedroom, 2-bathroom home features hardwood floors, a spacious backyard, and a newly renovated kitchen., Stunning 2-bedroom apartment with panoramic city views, modern amenities, and rooftop access., Beautiful townhouse in a prime location, with 4 bedrooms, 3 bathrooms, and a private garage."

# Add customised stop words
customize_stop_words = [
    'a', 'the', ',', '.'
]

for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

matcher = Matcher(nlp.vocab)

query = nlp(current_query)
query_tokens = [token.text for token in query if not token.is_stop]

descriptions = nlp(property_descriptions)
description_tokens = [token.text for token in descriptions if not token.is_stop]

#print('Description Tokens ', description_tokens)

    
query_description = [sent for sent in query.sents]
#print(query_description)

def get_similars(query_tokens, description_tokens):
	# Calculate similarity between pairs of words
	similar_word_pairs = []
	for word1 in query_tokens:
		for word2 in description_tokens:
			#print('Word 2 ', word2)
			if word1 != word2:
				similarity = nlp(word1).similarity(nlp(word2))
				similar_word_pairs.append((word1, word2, similarity))

	# Sort the pairs by similarity (highest first)
	similar_word_pairs.sort(key=lambda x: x[2], reverse=True)

	# Print the top similar word pairs
	for word1, word2, similarity in similar_word_pairs[:10]:
		print(f"{word1} - {word2}: {similarity}")


#get_similars(query_tokens, description_tokens)

class GenAI_LLM():
	def __init__(self):
		print('This is using LLM for GEN_AI purposes', '\n')
		self.property_descriptions_list = [
    "This charming 3-bedroom, 2-bathroom home features hardwood floors, a spacious backyard, and a newly renovated kitchen.",
    "Stunning 2-bedroom apartment with panoramic city views, modern amenities, and rooftop access.",
    "Beautiful townhouse in a prime location, with 4 bedrooms, 3 bathrooms, and a private garage."
]

		self.similar_word_pairs = []
		self.property_descriptions = "This charming 3-bedroom, 2-bathroom home features hardwood floors, a spacious backyard, and a newly renovated kitchen., Stunning 2-bedroom apartment with panoramic city views, modern amenities, and rooftop access., Beautiful townhouse in a prime location, with 4 bedrooms, 3 bathrooms, and a private garage."
		self.matcher = Matcher(nlp.vocab) # get matcher

	def get_similars(self, query_tokens, description_tokens):
		
		# Calculate similarity between pairs of words
		similar_word_pairs = []
		for query_word in query_tokens:
			for description_word in description_tokens:
				#print('Word 2 ', word2)
				if query_word != description_word:
					similarity = nlp(query_word).similarity(nlp(description_word))
					similar_word_pairs.append((query_word, description_word, similarity)) # append words and similarity degrees to the list

		# Sort the pairs by similarity (highest first)
		similar_word_pairs.sort(key=lambda x: x[2], reverse=True)

		# Print the top similar word pairs
		for query_word, description_word, similarity in similar_word_pairs[:10]:
			print(f"{query_word} - {description_word}: {similarity}")
		
		#print(similar_word_pairs[:5][:5][0][0])
		
		for query_word in similar_word_pairs[:5]:
			#print(query_word[1]) # Get descriptions
			self.__generate_summary(query_word[1])
			
		
	def __generate_summary(self, description):
		print('Genetate Summary Called')
		doc = nlp(property_descriptions)
		#print(property_descriptions)
		pattern = [{"LOWER": description}]
		matcher.add("description", [pattern])
		descriptions = nlp(property_descriptions)
		#print('Description ', descriptions)
		matches = matcher(descriptions)
		#print('Matches ', matches)
		for match_id, start, end in matches:
			string_id = nlp.vocab.strings[match_id]  # Get string representation
			span = doc[start:end]  # The matched span
			#print(match_id, string_id, start, end, span.text)
		self.__get_text(description)

	def __get_text(self, word):
		print("Word ", word)
		re.findall(r"([^.]*? + str(word) + [^.]*\.)",self.property_descriptions_list)  

genAI = GenAI_LLM()
genAI.get_similars(query_tokens, description_tokens)
