import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import spacy

# Load XLNet model and tokenizer
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetForSequenceClassification.from_pretrained(model_name)

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to extract entities using SpaCy
def extract_entities(text):
    doc = nlp(text)
    
    # Analyze topics based on spaCy's named entities
    topics = set()
    entities = [ent.text for ent in doc.ents]
    
    return entities

# Function to get XLNet response
def get_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    print('Logits : ', logits)
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class  # Replace this with an appropriate real estate response

# Main loop for interacting with the chatbot
if __name__ == "__main__":
    print("Real Estate Chatbot: Hello! How can I assist you today?")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "bye", "quit"]:
            print("Real Estate Chatbot: Goodbye!")
            break

        # Extract entities using SpaCy
        entities = extract_entities(user_input)
        print("Entities:", entities)

        # Get XLNet response
        bot_response = get_response(user_input)
        print("Real Estate Chatbot:", bot_response)
