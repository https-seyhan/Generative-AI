from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Load T5 model and tokenizer for generation
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load sentence-transformers model for retrieval
retriever_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Sample database of property descriptions
property_database = [
    "This spacious 3-bedroom apartment features hardwood floors and a modern kitchen.",
    "Charming 2-bedroom house with a large backyard, perfect for family gatherings.",
    "Luxurious penthouse with panoramic city views and top-of-the-line amenities."
]

# User query
query = "What features does the penthouse have?"

# Encode query
query_embedding = retriever_model.encode(query, convert_to_tensor=True)

# Calculate cosine similarity with each property description
similarities = []
for description in property_database:
    description_embedding = retriever_model.encode(description, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embedding, description_embedding)
    similarities.append(similarity.item())

# Find index of most similar property description
most_similar_index = similarities.index(max(similarities))
relevant_description = property_database[most_similar_index]

# Generate text based on the relevant property description
input_text = "summarize: " + relevant_description.strip()
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output text
output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
