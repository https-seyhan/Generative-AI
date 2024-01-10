import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import spacy

# Load XLNet model and tokenizer
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetForSequenceClassification.from_pretrained(model_name)

# Load SpaCy for NLP processing
nlp = spacy.load("en_core_web_sm")

# Fine-tune the model on real estate data (not provided in this example)

# Define a function to process user input using SpaCy
def process_input(user_input):
    doc = nlp(user_input)
    # Extract relevant information using SpaCy, e.g., entities, keywords
    entities = [ent.text for ent in doc.ents]
    keywords = [token.text for token in doc if token.is_alpha]
    return entities, keywords

# Define a function to get the model's response
def get_response(user_input):
    # Process input using SpaCy
    entities, keywords = process_input(user_input)

    # Use XLNet for generating a response (replace this with your fine-tuned model logic)
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Replace this with logic to map the predicted class to an appropriate real estate response
    response = f"Response based on predicted class: {predicted_class}"

    return response

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
