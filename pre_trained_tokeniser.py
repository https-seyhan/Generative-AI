from transformers import BertTokenizer

#pre-trained bert-base-uncased tokenizer to tokenize a sample sentence
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Ask users about their preferences (e.g., location, budget, property type)")
print(tokens)

#convert tokens to ids 
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)


token_ids = tokenizer.encode("This is an example of the bert tokenizer")
print(token_ids)
