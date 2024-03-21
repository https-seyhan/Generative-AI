from transformers import XLNetTokenizer, XLNetModel, XLNetLMHeadModel #XLNetForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# Load XLNet model and tokenizer for generation
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
#model = XLNetForConditionalGeneration.from_pretrained("xlnet-base-cased")
#model = XLNetModel.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')

# Load sentence-transformers model for retrieval
retriever_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Sample property descriptions
property_descriptions = [
    "This charming 3-bedroom, 2-bathroom home features hardwood floors, a spacious backyard, and a newly renovated kitchen.",
    "Stunning 2-bedroom apartment with panoramic city views, modern amenities, and rooftop access.",
    "Beautiful townhouse in a prime location, with 4 bedrooms, 3 bathrooms, and a private garage."
]

# Client query
client_query = "I'm looking for a family-friendly home with a backyard. Do you have any properties like that?"

# Encode property descriptions and client query
property_embeddings = retriever_model.encode(property_descriptions, convert_to_tensor=True)
query_embedding = retriever_model.encode(client_query, convert_to_tensor=True)

# Find similar property descriptions using cosine similarity
similarities = util.pytorch_cos_sim(query_embedding, property_embeddings)

# Find index of most similar property description
most_similar_index = similarities.argmax().item()

# Retrieve most relevant property description
relevant_property_description = property_descriptions[most_similar_index]

# Generate response based on the relevant property description
input_text = "summarize: " + relevant_property_description.strip()
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output response
output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print generated response
generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
print(type(property_descriptions))
print(type(generated_response))
print("Generated Response:", generated_response)

def compare_output_to_input(generated_response, property_descriptions):
	print(generated_response, property_descriptions)



compare_output_to_input(generated_response, property_descriptions)



