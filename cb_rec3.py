# Import necessary libraries
from flask import Flask, request, jsonify
from fuzzywuzzy import process  # Install using: pip install fuzzywuzzy

app = Flask(__name__)

# Sample property dataset (replace this with your actual dataset)
properties = [
    {"id": 1, "name": "Cozy Apartment", "type": "Apartment", "price": 1500, "location": "City A"},
    {"id": 2, "name": "Spacious House", "type": "House", "price": 3000, "location": "City B"},
    # Add more properties...
]

# Function to search for properties based on user input
def search_properties(query, limit=5):
    property_names = [property["name"] for property in properties]
    result, score = process.extractOne(query, property_names)
    if score >= 80:  # Adjust the similarity threshold as needed
        property_match = next(property for property in properties if property["name"] == result)
        return [property_match]
    else:
        return []

# API endpoint for property search
@app.route('/search', methods=['POST'])
def property_search():
    data = request.get_json()
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "Invalid input"}), 400

    results = search_properties(user_query)
    
    if results:
        return jsonify({"results": results})
    else:
        return jsonify({"message": "No matching properties found."})

# API endpoint for property recommendation
@app.route('/recommend', methods=['POST'])
def property_recommendation():
    # Implement your recommendation logic here based on user preferences, location, budget, etc.
    # This is a simplified example; you might want to use machine learning for better recommendations.
    return jsonify({"message": "Property recommendation feature is under development."})

if __name__ == '__main__':
    app.run(debug=True)
