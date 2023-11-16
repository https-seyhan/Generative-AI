import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sample data
text = "hello world! how are you doing today?"

# Create a mapping of unique characters to integers
chars = sorted(list(set(text)))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

# Prepare the dataset
max_len = 10
data_X = []
data_y = []

for i in range(0, len(text) - max_len, 1):
    seq_in = text[i:i + max_len]
    seq_out = text[i + max_len]
    data_X.append([char_to_int[char] for char in seq_in])
    data_y.append(char_to_int[seq_out])

# Reshape X to be [samples, time steps, features]
X = np.reshape(data_X, (len(data_X), max_len, 1))
# Normalize input values
X = X / float(len(chars))
y = tf.keras.utils.to_categorical(data_y)

# Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# Generate text
start = np.random.randint(0, len(data_X)-1)
pattern = data_X[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# Generate characters
for i in range(50):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(result, end='')
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")
