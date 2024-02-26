import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel

# Load pretrained XLNet model and tokenizer
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetLMHeadModel.from_pretrained(model_name)

# Define a function for content generation
def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate text using XLNet model
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example prompt for chatbot
prompt = "How's the weather today?"
print('Model ', model)
# Generate response
response = generate_response(prompt)
print("Generated Response:", response)
