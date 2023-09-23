import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="My_model",
  prompt="Web3 has the potential to change the nature of the internet from corporate-owned networks to controlled by users while maintaining the Web2 functionalities people love today. It can also be described as read/write/own. Users can govern these blockchain-based networks through cryptocurrency tokens. As the network grows, value can accrue to the community through the rising price of tokens.",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response["choices"][0]["text"])
