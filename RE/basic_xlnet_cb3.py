from transformers import XLNetTokenizer, XLNetLMHeadModel

# Load pre-trained XLNet model and tokenizer
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetLMHeadModel.from_pretrained(model_name)

# Define a function for generating responses
def generate_response(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, 
                            pad_token_id=tokenizer.eos_token_id)
    responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    return responses

# Chat loop
print("Chatbot: Hi! I'm a simple XLNet-based chatbot. You can start talking to me.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = generate_response(user_input)[0]
    print("Chatbot:", response)
