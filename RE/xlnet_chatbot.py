import torch
from transformers import XLNetForSequenceClassification, XLNetTokenizer

class RealEstateChatbot:
    def __init__(self, model_name='xlnet-base-cased', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, user_input):
        # Tokenize user input and convert to tensor
        input_ids = self.tokenizer.encode(user_input, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        # Generate model output
        with torch.no_grad():
            output = self.model(input_ids)[0]

        # Get predicted label (classification head for response)
        predicted_label = torch.argmax(output[0]).item()

        # Define responses based on predicted label (customize based on your use case)
        if predicted_label == 0:
            response = "I'm sorry, I didn't understand your query."
        elif predicted_label == 1:
            response = "Thank you for your interest in real estate. How can I assist you today?"
        else:
            response = "I'm sorry, I couldn't generate a valid response at the moment."

        return response

if __name__ == "__main__":
    # Initialize the chatbot
    real_estate_chatbot = RealEstateChatbot()

    print("Real Estate Chatbot: Hello! How can I assist you today? Type 'exit' to end the conversation.")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "bye", "quit"]:
            print("Real Estate Chatbot: Goodbye!")
            break

        bot_response = real_estate_chatbot.generate_response(user_input)
        print("Real Estate Chatbot:", bot_response)
