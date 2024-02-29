import torch
from transformers import XLNetLMHeadModel, XLNetTokenizer

def generate_text(model, tokenizer, prompt_text="[CLS]", max_length=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        for step in range(max_length):
            logits = model(input_ids)[0][:, -1, :] / temperature
            next_token_logits = logits[0]
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return generated_text

def main():
    model_name = "xlnet-base-cased"
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetLMHeadModel.from_pretrained(model_name)
    model.eval()

    prompt_text = "The weather today is"
    generated_text = generate_text(model, tokenizer, prompt_text=prompt_text, max_length=100, temperature=0.7)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
