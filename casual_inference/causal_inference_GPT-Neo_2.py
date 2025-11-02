import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pre-trained GPT-Neo model and tokenizer
model_name = 'EleutherAI/gpt-neo-1.3B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=150):
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate text
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )

    # Decode the generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example prompt
#prompt = "The benefits of regular exercise include"
prompt = "I'm looking for a family-friendly home with a backyard. Do you have any properties like that?"
generated_text = generate_text(prompt)

print("Generated Text:\n", generated_text)
