import boto3

# Initialize Bedrock client
client = boto3.client('bedrock')

# Example text for token searching
text = "Sample text for token searching"

# Call Bedrock model for token search
response = client.invoke_model(
    model_id='example-model-id',
    input={'text': text, 'operation': 'token_search'}
)

tokens = response['tokens']
print(tokens)


batch_texts = ["Text 1 for token search", "Text 2 for token search"]

response = client.invoke_model(
    model_id='example-model-id',
    input={'texts': batch_texts, 'operation': 'token_search'}
)

batch_tokens = response['tokens']
print(batch_tokens)