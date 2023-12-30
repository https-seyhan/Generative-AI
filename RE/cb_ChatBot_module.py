

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a chatbot instance
chatbot = ChatBot('PropertyBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot on English language data
trainer.train('chatterbot.corpus.english')

# Sample property data
properties = [
    {"type": "Apartment", "location": "City Center", "price": 2000},
    {"type": "House", "location": "Suburb", "price": 3000},
    # Add more property data as needed
]

# Define a function to search for properties based on user preferences
def search_properties(user_preferences):
    # Implement your property search logic here
    # For simplicity, this example returns all properties
    return properties

# Define a function to recommend properties based on user preferences
def recommend_properties(user_preferences):
    # Implement your property recommendation logic here
    # For simplicity, this example returns a random property
    import random
    return random.choice(properties)

# Main chat loop
while True:
    user_input = input("You: ")
    
    # Exit the loop if the user types 'exit'
    if user_input.lower() == 'exit':
        print("PropertyBot: Goodbye!")
        break
    
    # Get the chatbot's response
    response = chatbot.get_response(user_input)
    
    # If the user asks to search for properties
    if "search" in user_input.lower():
        # Implement property search logic
        user_preferences = {}  # Placeholder for user preferences
        search_results = search_properties(user_preferences)
        print("PropertyBot: Here are the search results:")
        for prop in search_results:
            print(f"{prop['type']} in {prop['location']} for ${prop['price']}/month")
    
    # If the user asks for property recommendations
    elif "recommend" in user_input.lower():
        # Implement property recommendation logic
        user_preferences = {}  # Placeholder for user preferences
        recommended_property = recommend_properties(user_preferences)
        print("PropertyBot: I recommend the following property:")
        print(f"{recommended_property['type']} in {recommended_property['location']} for ${recommended_property['price']}/month")
    
    else:
        # Default response from the chatbot
        print("PropertyBot:", response)
