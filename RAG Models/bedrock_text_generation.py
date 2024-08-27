import boto3

# Initialize the Bedrock client
bedrock = boto3.client('bedrock')

# Example input for text generation using Amazon Titan
response = bedrock.invoke_model(
    modelId='amazon.titan-tg1-large',
    contentType='application/json',
    accept='application/json',
    body='{"text": "Once upon a time,"}'
)

# Print the response
print(response['body'].read().decode('utf-8'))
