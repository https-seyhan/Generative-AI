import torch
from transformers import XLNetLMHeadModel, XLNetTokenizer

def generate_text(model_name_or_path, prompt_text, num_sequences=1, max_length=50, temperature=1.0):
    # Load pre-trained model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained(model_name_or_path)
    model = XLNetLMHeadModel.from_pretrained(model_name_or_path)

    # Tokenize input text
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generate text
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_sequences,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_texts = []
    for output_sequence in output_sequences:
        generated_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts

# Example usage
model_name_or_path = "xlnet-base-cased"
#prompt_text = "The cat"
prompt_text = "The weather today is"
num_sequences = 1 #3
max_length = 50
temperature = 0.7

generated_texts = generate_text(model_name_or_path, prompt_text, num_sequences, max_length, temperature)
for i, text in enumerate(generated_texts):
    print(f"Generated Text {i+1}: {text}")
