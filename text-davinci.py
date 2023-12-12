import openai

# Replace 'your-api-key' with your OpenAI API key
openai.api_key = 'your-api-key'

# Prompt for the language model
prompt = "Generate synthetic data for the following variables: age, income, and gender."

# Request data generation from the language model
response = openai.Completion.create(
  engine="text-davinci-002",  # You can experiment with different engines
  prompt=prompt,
  max_tokens=100
)

# Print the generated text
print(response.choices[0].text)
