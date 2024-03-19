from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Load the T5 model and tokenizer for generation
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the sentence-transformers model for retrieval
retriever_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Sample context for retrieval
#context = "Romeo and Juliet is a play written by William Shakespeare."

context = "About 75 percent of the value that generative AI use cases could deliver falls across four areas: Customer operations, marketing and sales, software engineering, and R&D. Across 16 business functions, we examined 63 use cases in which the technology can address specific business challenges in ways that produce one or more measurable outcomes. Examples include generative AI’s ability to support interactions with customers, generate creative content for marketing and sales, and draft computer code based on natural-language prompts, among many other tasks."

# Sample query for retrieval
#query = "What is the play Romeo and Juliet about?"
query = "What are generative AI use cases?"

# Encode the context and query
context_embedding = retriever_model.encode(context, convert_to_tensor=True)
query_embedding = retriever_model.encode(query, convert_to_tensor=True)

# Find similar sentences using cosine similarity
similar_sentences = util.pytorch_cos_sim(query_embedding, context_embedding)

# Find the index of the most similar sentence
max_sim_index = similar_sentences.argmax().item()

# Retrieve the most relevant context sentence
relevant_sentence = context.split('.')[max_sim_index]

# Generate text based on the relevant context
input_text = "summarize: " + relevant_sentence.strip()
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output text
output = model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
