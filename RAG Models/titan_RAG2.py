import boto3
from aws_cdk import core

# Initialize AWS Bedrock client for Titan model
bedrock_client = boto3.client('bedrock')

# Initialize Amazon Kendra for document retrieval
kendra_client = boto3.client('kendra')

# Example query
query = "How to optimize task scheduling in foundation work?"

# Use Kendra to retrieve relevant documents
response = kendra_client.query(
    IndexId='your-kendra-index-id',
    QueryText=query
)

# Get the top retrieved documents
retrieved_docs = [doc['DocumentExcerpt']['Text'] for doc in response['ResultItems']]

# Use Titan (via Bedrock) to generate an answer based on the query and retrieved documents
generate_response = bedrock_client.invoke_model(
    modelId='titan-llm-1',
    body={
        'input_text': f"Query: {query}\nDocuments: {retrieved_docs}"
    }
)

# Output the generated response
print(generate_response['output_text'])
