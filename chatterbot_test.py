from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a new chatbot instance
chatbot = ChatBot("RealEstateBot")

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot on the English language corpus data
trainer.train("chatterbot.corpus.english")

# Define a function to handle user input
def get_response(user_input):
    response = chatbot.get_response(user_input)
    return str(response)

# Main loop for interacting with the chatbot
if __name__ == "__main__":
    print("Real Estate Chatbot: Hello! How can I assist you today?")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "bye", "quit"]:
            print("Real Estate Chatbot: Goodbye!")
            break

        bot_response = get_response(user_input)
        print("Real Estate Chatbot:", bot_response)
