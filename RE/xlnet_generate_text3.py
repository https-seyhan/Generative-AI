import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel

# Load pre-trained XLNet tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')

# Set the model to generate mode
model.eval()

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate text
def generate_text(prompt, max_length=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids=input_ids, max_length=max_length, temperature=temperature)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt, max_length=100, temperature=0.7)
print("Generated text:")
print(generated_text)
