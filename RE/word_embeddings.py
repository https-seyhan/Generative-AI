import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")

# get the embedding vector for the words "Ask users about their preferences (e.g., location, budget, property type)"
example_token_id = tokenizer.convert_tokens_to_ids(["Ask users about their preferences (e.g., location, budget, property type)"])[0]
example_embedding = model.embeddings.word_embeddings(torch.tensor([example_token_id]))

print(example_embedding.shape)
