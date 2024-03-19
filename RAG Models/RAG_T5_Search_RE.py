from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Load T5 model and tokenizer for generation
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load sentence-transformers model for retrieval
retriever_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Sample property descriptions
property_descriptions = [
    "This beautiful two-bedroom apartment is located in a vibrant neighborhood with easy access to amenities and public transportation.",
    "A charming single-family home nestled in a quiet cul-de-sac, featuring spacious rooms and a large backyard perfect for outdoor gatherings.",
    "Luxurious penthouse condo with breathtaking views of the city skyline, boasting high-end finishes and state-of-the-art amenities.",
    # Add more property descriptions as needed
]

# Sample customer query
customer_query = "Looking for a cozy apartment with nearby parks and restaurants."

# Encode property descriptions and customer query
property_embeddings = retriever_model.encode(property_descriptions, convert_to_tensor=True)
query_embedding = retriever_model.encode(customer_query, convert_to_tensor=True)

# Find similar property descriptions using cosine similarity
similarities = util.pytorch_cos_sim(query_embedding, property_embeddings)

# Find index of most similar property description
most_similar_index = similarities.argmax().item()

# Retrieve most relevant property description
relevant_property_description = property_descriptions[most_similar_index]

# Generate property listing based on the relevant description
input_text = "generate property listing: " + relevant_property_description
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output property listing
output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print generated property listing
generated_property_listing = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Property Listing:", generated_property_listing)
