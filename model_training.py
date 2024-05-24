
import tensorflow
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


# Load data
with open('qa_pairs.json', 'r') as f:
    qa_pairs = json.load(f)

questions = [pair['question'] for pair in qa_pairs]
answers = [pair['answer'] for pair in qa_pairs]

# Tokenize data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

max_length = 20  # Define max sequence length based on your data
X = tokenizer.texts_to_sequences(questions)
y = tokenizer.texts_to_sequences(answers)

X = pad_sequences(X, maxlen=max_length, padding='post')
y = pad_sequences(y, maxlen=max_length, padding='post')

# Ensure y is the right shape
y = np.expand_dims(y, -1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length))
model.add(LSTM(64, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save('model.h5')

# Save tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

print("Model and tokenizer saved")
