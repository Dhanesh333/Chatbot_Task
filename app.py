
from flask import Flask, request, render_template, jsonify

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
import json
from keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)
import json

# Loading tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# Loading model
model = load_model('model.h5')

max_length = 20  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        
        if request.is_json:
            data = request.get_json()
            user_question = data.get('question')
            if user_question:
                tokenized_question = tokenizer.texts_to_sequences([user_question])
                padded_question = pad_sequences(tokenized_question, maxlen=max_length, padding='post')
                predicted_answer = model.predict(padded_question)
                predicted_answer = np.argmax(predicted_answer, axis=-1)
                answer_text = tokenizer.sequences_to_texts(predicted_answer)
                return jsonify({'question': user_question, 'answer': answer_text[0]})
            else:
                return jsonify({'error': 'Question not provided in JSON data'}), 400
        else:
            return jsonify({'error': 'Request must be in JSON format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)