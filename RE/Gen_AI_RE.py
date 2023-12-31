import random

class PropertyGenerator:
    def __init__(self):
        self.property_types = ["Apartment", "House", "Condo", "Townhouse"]
        self.locations = ["Downtown", "Suburb", "Countryside"]
        self.features = ["Swimming Pool", "Garden", "Garage", "Fireplace"]

    def generate_property(self):
        property_type = random.choice(self.property_types)
        location = random.choice(self.locations)
        bedrooms = random.randint(1, 5)
        bathrooms = random.randint(1, 4)
        price = random.randint(100000, 1000000)
        features = random.sample(self.features, k=random.randint(1, len(self.features)))

        property_data = {
            "Type": property_type,
            "Location": location,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Price": price,
            "Features": features
        }

        return property_data

    def generate_property_listings(self, num_listings):
        listings = [self.generate_property() for _ in range(num_listings)]
        return listings

# Example usage
generator = PropertyGenerator()
generated_listings = generator.generate_property_listings(5)

# Display generated listings
for i, listing in enumerate(generated_listings, 1):
    print(f"Property {i}:")
    for key, value in listing.items():
        print(f"{key}: {value}")
    print("\n")
