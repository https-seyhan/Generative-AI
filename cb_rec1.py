# Import necessary libraries
from flask import Flask, request, jsonify

app = Flask(__name__)

# Dummy property data for demonstration purposes
property_data = [
    {"id": 1, "name": "Cozy Apartment", "location": "City Center", "price": 1000},
    {"id": 2, "name": "Spacious House", "location": "Suburb", "price": 1500},
    {"id": 3, "name": "Luxury Penthouse", "location": "Downtown", "price": 2000},
    # Add more properties as needed
]

# Function to recommend properties based on user preferences
def recommend_properties(budget, location):
    recommended_properties = []
    for property in property_data:
        if property["price"] <= budget and property["location"] == location:
            recommended_properties.append(property)
    return recommended_properties

# Endpoint for property search and recommendation
@app.route('/property/search', methods=['POST'])
def property_search():
    try:
        # Get user input from the request
        user_data = request.get_json()
        budget = user_data['budget']
        location = user_data['location']

        # Validate user input
        if not budget or not location:
            return jsonify({"error": "Budget and location are required"}), 400

        # Perform property recommendation
        recommended_properties = recommend_properties(budget, location)

        # Return the recommended properties as JSON
        return jsonify({"properties": recommended_properties})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
