import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline

# Load XLNet model and tokenizer
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetForSequenceClassification.from_pretrained(model_name)

# Fine-tune the model on real estate data (not provided in this example)

# Define a function to get the model's response
def get_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
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

        bot_response = get_response(user_input)
        print("Real Estate Chatbot:", bot_response)
