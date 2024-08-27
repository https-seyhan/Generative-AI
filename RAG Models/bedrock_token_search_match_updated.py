import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock')

# Example model ID (this will vary depending on availability and use case)
model_id = 'titan-ml-cml-001'  # Replace with the correct model ID for your task

# Example input text for token searching and matching
input_text = "Find matching tokens in this text."
# Example token to search for
token_to_search = "search engine"


# Prepare the request payload
payload = {
    "input_text": input_text,
    # Add other necessary parameters for token searching/matching
     "token": token_to_search,
    "task": "token-matching"
}

# Invoke the Bedrock model
response = bedrock.invoke_model(
    model_id=model_id,
    content_type='application/json',
    accept='application/json',
    body=payload
)

# Process the response
result = response['body']
print(result)
