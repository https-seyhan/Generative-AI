from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Load the T5 model and tokenizer for generation
generator_model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the sentence-transformers model for retrieval
retriever_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample context for retrieval
context_corpus = [
    "Romeo and Juliet is a tragedy written by William Shakespeare.",
    "Hamlet is a tragedy written by William Shakespeare.",
    "Macbeth is a tragedy written by William Shakespeare."
]

# Sample query for retrieval
query = "What is Romeo and Juliet about?"

# Encode the query
query_embedding = retriever_model.encode(query, convert_to_tensor=True)

# Find similar contexts using cosine similarity
similarities = util.pytorch_cos_sim(query_embedding, retriever_model.encode(context_corpus, convert_to_tensor=True))

# Find the index of the most similar context
max_sim_index = similarities.argmax().item()

# Retrieve the most relevant context
relevant_context = context_corpus[max_sim_index]

# Generate text based on the relevant context
input_text = f"summarize: {relevant_context}"
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output text
output = generator_model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
