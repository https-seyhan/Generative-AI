class VectorDB:
    def __init__(self):
        self.database = {}

    def insert_vector(self, key, vector):
        self.database[key] = vector

    def get_vector(self, key):
        return self.database.get(key, None)

    def delete_vector(self, key):
        if key in self.database:
            del self.database[key]
        else:
            print(f"Key '{key}' not found in the database.")

# Example usage:
if __name__ == "__main__":
    db = VectorDB()

    # Insert vectors into the database
    db.insert_vector('vector1', [1, 2, 3, 4])
    db.insert_vector('vector2', [5, 6, 7, 8])
    db.insert_vector('vector3', [9, 10, 11, 12])

    # Retrieve vectors from the database
    print("Vector for 'vector1':", db.get_vector('vector1'))
    print("Vector for 'vector2':", db.get_vector('vector2'))

    # Try to retrieve a non-existing vector
    print("Vector for 'vector4':", db.get_vector('vector4'))

    # Delete a vector from the database
    db.delete_vector('vector2')

    # Try to retrieve the deleted vector
    print("Vector for 'vector2':", db.get_vector('vector2'))
