import torch
from transformers import XLNetLMHeadModel, XLNetTokenizer

# Load XLNet model and tokenizer
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetLMHeadModel.from_pretrained(model_name)

# Set seed for reproducibility
torch.manual_seed(42)

# Generate text
def generate_text(prompt, max_length=50, num_return_sequences=1, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences= 1, #num_return_sequences,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    return generated_texts

# Example prompt
prompt = "Once upon a time,"

# Generate text
generated_text = generate_text(prompt, max_length=100, num_return_sequences=5, temperature=0.8)

# Print generated text
for i, text in enumerate(generated_text, start=1):
    print(f"Generated Text {i}: {text}\n")
