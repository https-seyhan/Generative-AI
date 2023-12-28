import openai

openai.api_key = "YOUR_API_KEY"

def generate_response(user_input):
    prompt = f"User: {user_input}\nAI:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate engine
        prompt=prompt,
        max_tokens=150  # Adjust based on your desired response length
    )
    
    return response.choices[0].text.strip()

# Example Flask app
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    bot_response = generate_response(user_input)
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
