from transformers import XLNetLMHeadModel, GPT2LMHeadModel
from transformers import (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load sentence-transformers model for retrieval
retriever_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def query_xlnet(query, property_descriptions, model, tokenizer):
    # Combine conversation history with the current query
    #full_query = " <SEP> ".join(property_descriptions + [query])
    full_query = " <EOD> ".join(property_descriptions + [query])
    
    # Tokenize and encode the sequence
    inputs = tokenizer.encode_plus(full_query, add_special_tokens=True, return_tensors='pt')
    #inputs = tokenizer.encode_plus(full_query, add_special_tokens=True, return_tensors='pt')

    # Generate a sequence of tokens to predict
    output_sequences = model.generate(input_ids=inputs['input_ids'], 
                                      max_length=50, 
                                      num_return_sequences=1)

    # Decode the output sequence
    rewritten_query = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return rewritten_query
    
def query_xlnet_advanced(query, property_descriptions, model, tokenizer):
    # Combine conversation history with the current query
    full_query = " <SEP> ".join(property_descriptions + [query])
    #full_query = " <EOD> ".join(property_descriptions + [query])
    #full_query = " <CLS> ".join(property_descriptions + [query])
    
    # Tokenize and encode the sequence
    inputs = tokenizer.encode_plus(full_query, add_special_tokens=True, return_tensors='pt')
    #inputs = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors='pt')

    # Generate a sequence of tokens to predict
    output_sequences = model.generate(input_ids=inputs['input_ids'], 
                                      max_length=50, 
                                      num_return_sequences=1)

    # Decode the output sequence
    rewritten_query = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return rewritten_query

# Load pre-trained model and tokenizer
model_name = 'xlnet-base-cased'
vocabulary = 'xlnet-base-cased-spiece.model'

os.chdir('/home/saul/Desktop/generative-AI/RE/')
# Create XLNET model
model = XLNetLMHeadModel.from_pretrained(model_name)
tokenizer = XLNetTokenizer.from_pretrained(model_name)
#tokenizer = XLNetTokenizer(vocab_file=model_name,do_lower_case=False)
#tokenizer = XLNetTokenizer(vocab_file=vocabulary,do_lower_case=False)

# Example conversation history and current query
property_descriptions = [
    "This charming 3-bedroom, 2-bathroom home features hardwood floors, a spacious backyard, and a newly renovated kitchen.",
    "Stunning 2-bedroom apartment with panoramic city views, modern amenities, and rooftop access.",
    "Beautiful townhouse in a prime location, with 4 bedrooms, 3 bathrooms, and a private garage."
]
current_query = "I'm looking for a family-friendly home with a backyard. Do you have any properties like that?"

# Use XLNet to rewrite the query with conversation history
#rewritten_query = query_xlnet(current_query, property_descriptions, model, tokenizer)
rewritten_query = query_xlnet_advanced(current_query, property_descriptions, model, tokenizer)
print(f"Rewritten Query: {rewritten_query}\n")

# Start RAG
# Encode property descriptions and client query
property_embeddings = retriever_model.encode(property_descriptions, convert_to_tensor=True)
query_embedding = retriever_model.encode(rewritten_query, convert_to_tensor=True)

#print(query_embedding)

# Find similar property descriptions using cosine similarity
similarities = util.pytorch_cos_sim(query_embedding, property_embeddings)
#print('Similarities: ', similarities)

# Find index of most similar property description
most_similar_index = similarities.argmax().item()

# Retrieve most relevant property description
relevant_property_description = property_descriptions[most_similar_index]

# End RAG
# Generate response based on the relevant property description
input_text = "summarize: " + relevant_property_description.strip()
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output response
#output = model.generate(input_ids=input_ids, max_length=100, num_beams=50, num_return_sequences=2, early_stopping=True)
output = model.generate(input_ids=input_ids, max_length=100,  num_return_sequences=1, early_stopping=True)

#print('Output :', output)

# Decode and print generated response
generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
#print(type(property_descriptions))
#print(type(generated_response))
#print("Generated Response:", generated_response)

def save_to_csv(generated_response):
	#print(generated_response)
	os.chdir('/home/saul/Desktop/generative-AI/RE/')
	df = pd.DataFrame(generated_response, columns=["generated_text"])
	df.to_csv('generated_text_dev.csv')
	
def compare_output_to_input(generated_response, property_descriptions):
	#print('Generated Response: ', generated_response, end='\n')
	generated_response = generated_response.split(',')
	print('Generated Response: ', generated_response, end='\n')
	#print('Property Descriptions: ', property_descriptions)
	save_to_csv(generated_response)

compare_output_to_input(generated_response, property_descriptions)
