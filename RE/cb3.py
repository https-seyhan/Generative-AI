import openai

# Set your OpenAI GPT-3 API key
openai.api_key = 'YOUR_API_KEY'

def get_bot_response(user_input):
    # You can customize the prompt based on your application's needs
    prompt = f"User: {user_input}\nBot:"

    # Make a request to GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",  # You may choose a different engine
        prompt=prompt,
        max_tokens=100  # Adjust max_tokens as needed
    )

    # Extract and return the bot's reply from GPT-3 response
    bot_reply = response['choices'][0]['text'].strip()
    return bot_reply

if __name__ == "__main__":
    print("Welcome to the Real Estate Chatbot!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break

        bot_response = get_bot_response(user_input)
        print("Bot:", bot_response)
