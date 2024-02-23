from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Ask users about their preferences (e.g., location, budget, property type)")
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
