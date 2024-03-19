from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Load T5 model and tokenizer for generation
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load sentence-transformers model for retrieval
retriever_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Sample property features or query
property_features = "3 bedroom apartment with a balcony in downtown."

# Sample query for the type of property description to generate
query = "Generate property description for a 3-bedroom apartment in downtown with a balcony."

# Encode property features and query
property_embedding = retriever_model.encode(property_features, convert_to_tensor=True)
query_embedding = retriever_model.encode(query, convert_to_tensor=True)

# Find similar property descriptions using cosine similarity
similarities = util.pytorch_cos_sim(query_embedding, property_embedding)

# Find index of most similar property description
most_similar_index = similarities.argmax().item()

# Retrieve most relevant property description
relevant_description = ["Spacious 3-bedroom apartment located in downtown with a private balcony."]

# Generate text based on the relevant property description
input_text = "summarize: " + relevant_description[most_similar_index].strip()
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output text
output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print generated property description
generated_description = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Property Description:", generated_description)
