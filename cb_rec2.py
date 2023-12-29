import random

# Property data (replace with your own data)
property_data = {
    'house1': {'type': 'house', 'location': 'City A', 'price': 500000},
    'apartment1': {'type': 'apartment', 'location': 'City B', 'price': 300000},
    'villa1': {'type': 'villa', 'location': 'City C', 'price': 800000},
    # Add more properties as needed
}

def search_properties(location=None, property_type=None, budget=None):
    results = []
    for property_id, details in property_data.items():
        if (location is None or details['location'].lower() == location.lower()) and \
           (property_type is None or details['type'].lower() == property_type.lower()) and \
           (budget is None or details['price'] <= budget):
            results.append((property_id, details))
    return results

def recommend_property(preferences):
    # For simplicity, the recommendation is random in this example
    matching_properties = search_properties(**preferences)
    if matching_properties:
        property_id, details = random.choice(matching_properties)
        return f"We recommend '{property_id}' - Type: {details['type']}, Location: {details['location']}, Price: ${details['price']}"
    else:
        return "Sorry, we couldn't find any properties matching your preferences."

def main():
    print("Welcome to the Property Search and Recommendation Chatbot!")

    while True:
        user_input = input("You: ").lower()

        if user_input == 'exit':
            print("Goodbye!")
            break

        elif user_input == 'help':
            print("Commands: 'search', 'recommend', 'exit'")

        elif user_input == 'search':
            location = input("Enter location (or press Enter to skip): ").strip()
            property_type = input("Enter property type (or press Enter to skip): ").strip()
            budget_str = input("Enter your budget (or press Enter to skip): ").strip()

            try:
                budget = float(budget_str)
            except ValueError:
                budget = None

            results = search_properties(location, property_type, budget)
            if results:
                for property_id, details in results:
                    print(f"{property_id} - Type: {details['type']}, Location: {details['location']}, Price: ${details['price']}")
            else:
                print("No properties found matching the criteria.")

        elif user_input == 'recommend':
            preferences = {
                'location': input("Enter preferred location: ").strip(),
                'property_type': input("Enter preferred property type: ").strip(),
                'budget': float(input("Enter your budget: ").strip())
            }
            recommendation = recommend_property(preferences)
            print(recommendation)

        else:
            print("Invalid command. Type 'help' for a list of commands.")

if __name__ == "__main__":
    main()
