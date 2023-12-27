import openai

# Set your OpenAI GPT-3.5 API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def generate_response(user_input):
    prompt = f"Customer: {user_input}\nAgent:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate engine
        prompt=prompt,
        max_tokens=150,  # Adjust as needed
        temperature=0.7,  # Adjust for more creative or focused responses
        stop=None  # You can add custom stop words to limit the response
    )
    
    return response.choices[0].text.strip()

# Example usage
user_input = "What properties do you have available in downtown?"
bot_response = generate_response(user_input)
print("Bot:", bot_response)
