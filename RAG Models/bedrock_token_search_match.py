import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock', region_name='us-west-2')

# Example text to search within
text = "OpenSearch is a powerful search engine built on top of Elasticsearch."

# Example token to search for
token_to_search = "search engine"

# Request payload for the model
payload = {
    "text": text,
    "token": token_to_search,
    "task": "token-matching"
}

# Replace 'example-model-id' with the actual model ID, e.g., 'cohere-command-xlarge'
response = bedrock.invoke_model(
    modelId="example-model-id",  # replace with actual model ID
    body=payload
)

# Process the response to find matches
matches = response.get('matches', [])

for match in matches:
    print(f"Found match: {match['text']} at position {match['start']}")

