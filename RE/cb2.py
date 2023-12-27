import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

def get_bot_response(user_input):
    # Use OpenAI GPT-3 to generate a response
    prompt = f"Customer: {user_input}\nAgent:"
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can experiment with different engines
        prompt=prompt,
        temperature=0.7,
        max_tokens=150,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        bot_response = get_bot_response(user_input)
        print(f"Bot: {bot_response}")
