import openai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Set your OpenAI GPT-3 API key here
openai.api_key = 'YOUR_API_KEY'

def get_bot_response(user_input):
    # Use the OpenAI GPT-3 API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()

@app.route('/')
def index():
    return render_template('index_generative.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    bot_response = get_bot_response(user_input)
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
