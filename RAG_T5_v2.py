from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Load T5 model and tokenizer for generation
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load sentence-transformers model for retrieval
retriever_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Define context and query
context = "Context text goes here."
query = "Query text goes here."

# Encode context and query
context_embedding = retriever_model.encode(context, convert_to_tensor=True)
query_embedding = retriever_model.encode(query, convert_to_tensor=True)

# Find similar sentences using cosine similarity
similarities = util.pytorch_cos_sim(query_embedding, context_embedding)

# Find index of most similar sentence
most_similar_index = similarities.argmax().item()

# Retrieve most relevant context sentence
relevant_sentence = context.split('.')[most_similar_index]

# Generate text based on the relevant context
input_text = "summarize: " + relevant_sentence.strip()
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output text
output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
