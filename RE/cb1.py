# Import necessary libraries
from flask import Flask, request, jsonify
import spacy

# Initialize Flask app
app = Flask(__name__)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample data - replace this with your real estate information
property_listings = [
    {"id": 1, "type": "Apartment", "price": 100000, "location": "Downtown"},
    {"id": 2, "type": "House", "price": 250000, "location": "Suburb"},
    # Add more listings as needed
]

# Function to handle incoming messages
def process_message(message):
    doc = nlp(message)
    
    # Implement your chatbot logic here based on user input
    # For simplicity, let's assume the user is asking for property listings
    if any(token.text.lower() in ["property", "listings", "houses", "apartments"] for token in doc):
        return property_listings
    else:
        return "I'm sorry, I didn't understand. Please ask about property listings."

# Route for handling incoming messages
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data["message"]
    
    bot_response = process_message(user_message)
    
    return jsonify({"response": bot_response})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
